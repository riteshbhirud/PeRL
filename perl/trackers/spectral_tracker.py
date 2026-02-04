# Copyright (c) 2025
# Spectral tracking for PEFT adapter weights during RLVR training
# Extension for mechanistic analysis of PEFT methods in RL

import torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpectralMetrics:
    """Container for spectral metrics of a single layer at a single step."""
    step: int
    layer_name: str
    peft_type: str

    # Core spectral metrics
    singular_values: np.ndarray  # Full array of singular values
    spectral_gap: float  # σ_1 - σ_r
    condition_number: float  # σ_1 / σ_r
    effective_rank: float  # (Σσ_i)^2 / Σ(σ_i)^2
    nuclear_norm: float  # Σσ_i (trace norm)
    frobenius_norm: float  # sqrt(Σσ_i^2)

    # Energy distribution metrics
    top1_sv_ratio: float  # σ_1 / Σσ_i
    top5_sv_ratio: float  # Σσ_{1:5} / Σσ_i
    top10_sv_ratio: float  # Σσ_{1:10} / Σσ_i

    # Shape info
    matrix_shape: Tuple[int, int]
    rank: int  # Number of non-zero singular values

    # DoRA-specific (optional)
    magnitude_norm: Optional[float] = None
    direction_norm: Optional[float] = None
    magnitude_direction_ratio: Optional[float] = None

    # AdaLoRA-specific (optional)
    importance_scores: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert numpy arrays to lists
        d['singular_values'] = self.singular_values.tolist()
        d['matrix_shape'] = list(self.matrix_shape)
        if self.importance_scores is not None:
            d['importance_scores'] = self.importance_scores.tolist()
        return d


class SpectralTracker:
    """
    Track singular value decomposition of PEFT adapter weights during training.

    Supports all PEFT methods in PeRL:
    - LoRA, DoRA, PiSSA, MiLoRA, LoRA+, rsLoRA, LoRA-FA (all use lora_A/lora_B)
    - AdaLoRA (uses lora_E, lora_A, lora_B with importance scores)
    - VeRA (frozen random matrices with scaling vectors)
    - MiSS (mixture of sub-spaces)
    - IA3 (scaling vectors only - no SVD needed)
    - LayerNorm tuning (no adapters - skip)

    For each adapter layer at each logging step, computes:
    - Full SVD: W = U Σ V^T
    - Spectral metrics: gap, condition number, effective rank
    - Energy distribution: top-k singular value ratios
    - Method-specific metrics (DoRA magnitude, AdaLoRA importance)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        log_frequency: int = 100,
        save_dir: str = "./spectral_logs",
        peft_type: Optional[str] = None,
        track_gradients: bool = False,
        compute_full_svd: bool = True,
        max_layers_to_track: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize the SpectralTracker.

        Args:
            model: The model with PEFT adapters
            log_frequency: Log spectral metrics every N training steps
            save_dir: Directory to save spectral history
            peft_type: Override PEFT type detection (optional)
            track_gradients: Also track gradient norms (increases overhead)
            compute_full_svd: Compute full SVD (vs just top-k for speed)
            max_layers_to_track: Limit number of layers to track (for memory)
            verbose: Print progress messages
        """
        self.model = model
        self.log_frequency = log_frequency
        self.save_dir = save_dir
        self.track_gradients = track_gradients
        self.compute_full_svd = compute_full_svd
        self.max_layers_to_track = max_layers_to_track
        self.verbose = verbose

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Detect PEFT type from model
        self.peft_type = peft_type or self._detect_peft_type()

        # Discover adapter layers
        self.adapter_layers = self._discover_adapter_layers()

        # History storage: {layer_name: {metric_name: [values_over_steps]}}
        self.history: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self.steps_logged: List[int] = []

        # Timing stats
        self.total_logging_time = 0.0
        self.num_log_calls = 0

        if self.verbose:
            logger.info(f"[SpectralTracker] Initialized with PEFT type: {self.peft_type}")
            logger.info(f"[SpectralTracker] Found {len(self.adapter_layers)} adapter layers to track")
            logger.info(f"[SpectralTracker] Logging every {log_frequency} steps to {save_dir}")

    def _detect_peft_type(self) -> str:
        """Detect the PEFT method being used from model structure."""
        # Check for various PEFT signatures
        for name, module in self.model.named_modules():
            # DoRA check (has lora_magnitude_vector or use_dora flag)
            if hasattr(module, 'lora_magnitude_vector'):
                return "dora"
            if hasattr(module, 'use_dora') and module.use_dora:
                return "dora"

            # AdaLoRA check (has lora_E for importance)
            if hasattr(module, 'lora_E'):
                return "adalora"

            # VeRA check (has vera_lambda_b and vera_lambda_d)
            if hasattr(module, 'vera_lambda_b'):
                return "vera"

            # IA3 check (has ia3_l scaling)
            if hasattr(module, 'ia3_l'):
                return "ia3"

            # MiSS check
            if 'Miss' in type(module).__name__:
                return "miss"

            # Standard LoRA check (lora_A and lora_B)
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                return "lora"

        # Check for LayerNorm tuning (no adapters)
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'ln' in name.lower():
                return "layernorm"

        return "unknown"

    def _discover_adapter_layers(self) -> Dict[str, torch.nn.Module]:
        """Find all adapter layers in the model."""
        adapter_layers = {}

        for name, module in self.model.named_modules():
            is_adapter = False

            # Check for LoRA-style adapters
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                is_adapter = True

            # Check for AdaLoRA
            elif hasattr(module, 'lora_E'):
                is_adapter = True

            # Check for VeRA
            elif hasattr(module, 'vera_lambda_b'):
                is_adapter = True

            # Check for IA3
            elif hasattr(module, 'ia3_l'):
                is_adapter = True

            # Check for MiSS
            elif 'Miss' in type(module).__name__ and hasattr(module, 'weight'):
                is_adapter = True

            if is_adapter:
                adapter_layers[name] = module

        # Optionally limit number of layers
        if self.max_layers_to_track and len(adapter_layers) > self.max_layers_to_track:
            # Keep evenly spaced layers
            keys = list(adapter_layers.keys())
            step = len(keys) // self.max_layers_to_track
            selected_keys = keys[::step][:self.max_layers_to_track]
            adapter_layers = {k: adapter_layers[k] for k in selected_keys}
            if self.verbose:
                logger.info(f"[SpectralTracker] Limited to {len(adapter_layers)} layers")

        return adapter_layers

    def _get_effective_weight_matrix(self, module: torch.nn.Module, name: str) -> Optional[torch.Tensor]:
        """
        Get the effective weight update matrix for different PEFT types.

        For LoRA: returns B @ A (the low-rank update)
        For AdaLoRA: returns P @ diag(Λ) @ Q
        For VeRA: returns diag(λ_b) @ B @ diag(λ_d) @ A
        """
        try:
            # Standard LoRA/DoRA/PiSSA/MiLoRA/LoRA+/rsLoRA
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Handle ModuleDict or dict-style access (PEFT library uses ModuleDict)
                lora_A = module.lora_A
                lora_B = module.lora_B

                if isinstance(lora_A, (dict, torch.nn.ModuleDict)):
                    # Get default adapter
                    adapter_name = list(lora_A.keys())[0] if lora_A else 'default'
                    A = lora_A[adapter_name].weight  # (r, in_features)
                    B = lora_B[adapter_name].weight  # (out_features, r)
                elif hasattr(lora_A, 'weight'):
                    A = lora_A.weight
                    B = lora_B.weight
                else:
                    return None

                # Compute effective update: B @ A
                return B @ A

            # AdaLoRA with importance scores
            elif hasattr(module, 'lora_E'):
                lora_A = module.lora_A
                lora_B = module.lora_B
                lora_E = module.lora_E

                if isinstance(lora_A, (dict, torch.nn.ModuleDict)):
                    adapter_name = list(lora_A.keys())[0] if lora_A else 'default'
                    A = lora_A[adapter_name].weight
                    B = lora_B[adapter_name].weight
                    if isinstance(lora_E, (dict, torch.nn.ModuleDict, torch.nn.ParameterDict)):
                        E = lora_E[adapter_name]
                    else:
                        E = lora_E
                elif hasattr(lora_A, 'weight'):
                    A = lora_A.weight
                    B = lora_B.weight
                    E = lora_E if isinstance(lora_E, torch.Tensor) else lora_E.weight
                else:
                    return None

                # Handle E being a Parameter or Tensor
                if hasattr(E, 'weight'):
                    E = E.weight
                E = E.squeeze()

                # AdaLoRA: B @ diag(E) @ A
                return B @ torch.diag(E) @ A

            # VeRA with scaling vectors
            elif hasattr(module, 'vera_lambda_b'):
                vera_A = module.vera_A  # Frozen random (r, in_features)
                vera_B = module.vera_B  # Frozen random (out_features, r)
                lambda_b = module.vera_lambda_b  # Trainable scaling (out_features,)
                lambda_d = module.vera_lambda_d  # Trainable scaling (r,)

                # Handle if these are Parameters
                if hasattr(lambda_b, 'data'):
                    lambda_b = lambda_b.data
                if hasattr(lambda_d, 'data'):
                    lambda_d = lambda_d.data

                # VeRA: diag(λ_b) @ B @ diag(λ_d) @ A
                # = (out, out) @ (out, r) @ (r, r) @ (r, in)
                # Simplified: (λ_b[:, None] * B) @ (λ_d[:, None] * A)
                scaled_A = lambda_d.unsqueeze(1) * vera_A  # (r, 1) * (r, in) = (r, in)
                scaled_B = lambda_b.unsqueeze(1) * vera_B  # (out, 1) * (out, r) = (out, r)
                return scaled_B @ scaled_A  # (out, r) @ (r, in) = (out, in)

            return None

        except Exception as e:
            if self.verbose:
                logger.warning(f"[SpectralTracker] Error getting weight matrix for {name}: {e}")
            return None

    def _compute_spectral_metrics(
        self,
        weight_matrix: torch.Tensor,
        step: int,
        layer_name: str,
        module: torch.nn.Module,
    ) -> Optional[SpectralMetrics]:
        """Compute all spectral metrics for a weight matrix."""
        try:
            # Move to CPU and convert to float32 for stable SVD
            W = weight_matrix.detach().float().cpu()

            # Compute SVD
            if self.compute_full_svd:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            else:
                # For very large matrices, use truncated SVD (top 100)
                k = min(100, min(W.shape))
                U, S, Vh = torch.svd_lowrank(W, q=k)

            # Convert to numpy for metrics
            S_np = S.numpy()

            # Handle edge cases for zero/near-zero matrices
            S_max = S_np[0] if len(S_np) > 0 else 0.0
            S_min = S_np[-1] if len(S_np) > 0 else 0.0
            S_sum = S_np.sum()

            # Avoid division by zero with sensible defaults
            if S_sum < 1e-10:
                S_sum = 1e-10
            if S_min < 1e-10:
                S_min = 1e-10

            # Compute metrics
            spectral_gap = float(S_max - S_np[-1]) if len(S_np) > 1 else 0.0
            # Condition number: if max singular value is 0, matrix is zero -> return 1.0 (well-conditioned trivially)
            if S_max < 1e-10:
                condition_number = 1.0
            else:
                condition_number = float(S_max / S_min)

            # Effective rank: (Σσ_i)^2 / Σ(σ_i)^2
            S_sq_sum = (S_np ** 2).sum()
            if S_sq_sum < 1e-10:
                S_sq_sum = 1e-10
            effective_rank = float((S_sum ** 2) / S_sq_sum)

            # Norms
            nuclear_norm = float(S_sum)
            frobenius_norm = float(np.sqrt(S_sq_sum))

            # Energy ratios
            top1_ratio = float(S_np[0] / S_sum) if len(S_np) > 0 else 0.0
            top5_ratio = float(S_np[:5].sum() / S_sum) if len(S_np) >= 5 else float(S_sum / S_sum)
            top10_ratio = float(S_np[:10].sum() / S_sum) if len(S_np) >= 10 else float(S_sum / S_sum)

            # Count non-zero singular values
            rank = int((S_np > 1e-10).sum())

            metrics = SpectralMetrics(
                step=step,
                layer_name=layer_name,
                peft_type=self.peft_type,
                singular_values=S_np,
                spectral_gap=spectral_gap,
                condition_number=condition_number,
                effective_rank=effective_rank,
                nuclear_norm=nuclear_norm,
                frobenius_norm=frobenius_norm,
                top1_sv_ratio=top1_ratio,
                top5_sv_ratio=top5_ratio,
                top10_sv_ratio=top10_ratio,
                matrix_shape=tuple(W.shape),
                rank=rank,
            )

            # DoRA-specific metrics
            if self.peft_type == "dora" and hasattr(module, 'lora_magnitude_vector'):
                mag_vector = module.lora_magnitude_vector
                if isinstance(mag_vector, (dict, torch.nn.ModuleDict, torch.nn.ParameterDict)):
                    adapter_name = list(mag_vector.keys())[0]
                    mag = mag_vector[adapter_name]
                else:
                    mag = mag_vector

                # Handle Parameter vs Tensor
                if hasattr(mag, 'data'):
                    mag = mag.data

                metrics.magnitude_norm = float(torch.norm(mag).item())

                # Direction is the normalized weight update
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A
                    lora_B = module.lora_B
                    if isinstance(lora_A, (dict, torch.nn.ModuleDict)):
                        adapter_name = list(lora_A.keys())[0]
                        A = lora_A[adapter_name].weight
                        B = lora_B[adapter_name].weight
                    elif hasattr(lora_A, 'weight'):
                        A = lora_A.weight
                        B = lora_B.weight
                    else:
                        A = B = None

                    if A is not None and B is not None:
                        direction = B @ A
                        metrics.direction_norm = float(torch.norm(direction).item())
                        if metrics.direction_norm > 1e-10:
                            metrics.magnitude_direction_ratio = metrics.magnitude_norm / metrics.direction_norm

            # AdaLoRA importance scores
            if self.peft_type == "adalora" and hasattr(module, 'lora_E'):
                lora_E = module.lora_E
                if isinstance(lora_E, (dict, torch.nn.ModuleDict, torch.nn.ParameterDict)):
                    adapter_name = list(lora_E.keys())[0]
                    E = lora_E[adapter_name]
                else:
                    E = lora_E

                # Handle Parameter vs Tensor vs Module
                if hasattr(E, 'weight'):
                    E = E.weight
                elif hasattr(E, 'data'):
                    E = E.data

                metrics.importance_scores = E.detach().cpu().numpy().flatten()

            return metrics

        except Exception as e:
            if self.verbose:
                logger.warning(f"[SpectralTracker] Error computing metrics for {layer_name}: {e}")
            return None

    def log_step(self, step: int) -> bool:
        """
        Log spectral metrics if step is a multiple of log_frequency.

        Args:
            step: Current training step

        Returns:
            True if logging occurred, False otherwise
        """
        if step % self.log_frequency != 0:
            return False

        # Only log on main process in distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return False

        start_time = time.time()

        if self.verbose:
            logger.info(f"[SpectralTracker] Logging spectral metrics at step {step}")

        self.steps_logged.append(step)

        # Process each adapter layer
        with torch.no_grad():
            for layer_name, module in self.adapter_layers.items():
                # Get effective weight matrix
                W = self._get_effective_weight_matrix(module, layer_name)
                if W is None:
                    continue

                # Compute spectral metrics
                metrics = self._compute_spectral_metrics(W, step, layer_name, module)
                if metrics is None:
                    continue

                # Store in history
                metrics_dict = metrics.to_dict()
                for metric_name, value in metrics_dict.items():
                    if metric_name not in ['step', 'layer_name', 'peft_type']:
                        self.history[layer_name][metric_name].append(value)

                # Store step separately
                self.history[layer_name]['steps'].append(step)

        # Track timing
        elapsed = time.time() - start_time
        self.total_logging_time += elapsed
        self.num_log_calls += 1

        if self.verbose:
            logger.info(f"[SpectralTracker] Logged {len(self.adapter_layers)} layers in {elapsed:.2f}s")

        return True

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save all tracked data to disk.

        Args:
            filepath: Path to save file. If None, uses save_dir/spectral_history.pt

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, "spectral_history.pt")

        # Prepare data for saving
        save_data = {
            'peft_type': self.peft_type,
            'log_frequency': self.log_frequency,
            'steps_logged': self.steps_logged,
            'layer_names': list(self.adapter_layers.keys()),
            'history': dict(self.history),
            'timing': {
                'total_logging_time': self.total_logging_time,
                'num_log_calls': self.num_log_calls,
                'avg_time_per_call': self.total_logging_time / max(1, self.num_log_calls),
            },
        }

        # Save as PyTorch file (handles numpy arrays well)
        torch.save(save_data, filepath)

        # Also save a JSON summary (without large arrays)
        summary_path = filepath.replace('.pt', '_summary.json')
        summary = {
            'peft_type': self.peft_type,
            'log_frequency': self.log_frequency,
            'num_steps_logged': len(self.steps_logged),
            'steps_range': [min(self.steps_logged), max(self.steps_logged)] if self.steps_logged else [],
            'num_layers_tracked': len(self.adapter_layers),
            'layer_names': list(self.adapter_layers.keys()),
            'timing': save_data['timing'],
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            logger.info(f"[SpectralTracker] Saved history to {filepath}")
            logger.info(f"[SpectralTracker] Saved summary to {summary_path}")

        return filepath

    def get_layer_history(self, layer_name: str) -> Dict[str, List]:
        """
        Retrieve history for a specific layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Dictionary of {metric_name: [values_over_time]}
        """
        if layer_name not in self.history:
            raise KeyError(f"Layer '{layer_name}' not found. Available: {list(self.history.keys())}")
        return dict(self.history[layer_name])

    def get_metric_across_layers(self, metric_name: str, step_idx: int = -1) -> Dict[str, Any]:
        """
        Get a specific metric across all layers at a given step.

        Args:
            metric_name: Name of the metric (e.g., 'effective_rank')
            step_idx: Index into steps_logged (-1 for latest)

        Returns:
            Dictionary of {layer_name: metric_value}
        """
        result = {}
        for layer_name, metrics in self.history.items():
            if metric_name in metrics and len(metrics[metric_name]) > abs(step_idx):
                result[layer_name] = metrics[metric_name][step_idx]
        return result

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all tracking."""
        if not self.steps_logged:
            return {'error': 'No steps logged yet'}

        summary = {
            'peft_type': self.peft_type,
            'total_steps_logged': len(self.steps_logged),
            'step_range': (min(self.steps_logged), max(self.steps_logged)),
            'num_layers': len(self.adapter_layers),
            'timing': {
                'total_time': self.total_logging_time,
                'avg_per_call': self.total_logging_time / max(1, self.num_log_calls),
            },
        }

        # Aggregate stats across layers for latest step
        for metric in ['effective_rank', 'condition_number', 'spectral_gap', 'frobenius_norm']:
            values = self.get_metric_across_layers(metric, step_idx=-1)
            if values:
                vals = list(values.values())
                summary[f'{metric}_latest'] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                }

        return summary

    @staticmethod
    def load(filepath: str) -> 'SpectralTracker':
        """
        Load a saved SpectralTracker from disk.

        Note: This creates a tracker without a model reference.
        Use for analysis only, not for continued tracking.
        """
        # weights_only=False needed for defaultdict and numpy arrays
        data = torch.load(filepath, weights_only=False)

        # Create a minimal tracker for analysis
        tracker = object.__new__(SpectralTracker)
        tracker.model = None
        tracker.log_frequency = data['log_frequency']
        tracker.save_dir = os.path.dirname(filepath)
        tracker.peft_type = data['peft_type']
        tracker.adapter_layers = {name: None for name in data['layer_names']}
        tracker.history = defaultdict(lambda: defaultdict(list), data['history'])
        tracker.steps_logged = data['steps_logged']
        tracker.total_logging_time = data['timing']['total_logging_time']
        tracker.num_log_calls = data['timing']['num_log_calls']
        tracker.verbose = False
        tracker.track_gradients = False
        tracker.compute_full_svd = True
        tracker.max_layers_to_track = None

        return tracker


def create_spectral_tracker(
    model: torch.nn.Module,
    args: Any,
    save_dir: Optional[str] = None,
) -> SpectralTracker:
    """
    Factory function to create SpectralTracker from training args.

    Args:
        model: The PEFT model
        args: TrainConfig object with tracker settings
        save_dir: Override save directory

    Returns:
        Configured SpectralTracker instance
    """
    # Get tracker config if available
    tracker_config = getattr(args, 'tracker', None)

    if tracker_config is None:
        # Use defaults
        log_frequency = 100
        peft_type = getattr(args.peft, 'type', None)
    else:
        log_frequency = getattr(tracker_config, 'spectral_log_frequency', 100)
        peft_type = getattr(args.peft, 'type', None)

    if save_dir is None:
        save_dir = os.path.join(args.training.output_dir, 'spectral_logs')

    return SpectralTracker(
        model=model,
        log_frequency=log_frequency,
        save_dir=save_dir,
        peft_type=peft_type,
        verbose=not getattr(args.common, 'debug', False),
    )
