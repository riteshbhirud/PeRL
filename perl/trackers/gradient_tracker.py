# Copyright (c) 2025
# Gradient flow tracking for PEFT adapter weights during RLVR training
# Extension for mechanistic analysis of PEFT methods in RL

import torch
import numpy as np
import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientSnapshot:
    """Container for gradient data at a single step."""
    step: int
    gradients: Dict[str, float]  # {param_name: grad_norm}
    layer_gradients: Dict[int, Dict[str, float]]  # {layer_idx: {component: grad_norm}}
    total_grad_norm: float
    num_params_with_grads: int
    timestamp: float


class GradientFlowTracker:
    """
    Track gradient magnitudes across layers during training.

    For each layer, tracks gradient norms of:
    - lora_A parameters
    - lora_B parameters
    - magnitude parameters (DoRA only)
    - Other adapter parameters (vera_lambda, ia3_l, etc.)

    Organizes data to create heatmaps: layers (rows) × training steps (cols)

    Integration note: For accurate gradient capture, this tracker should log
    AFTER backward() but BEFORE optimizer.step(). In HuggingFace Trainer,
    use the callback's on_step_end with gradient accumulation, or call
    log_step() manually in a custom training loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        log_frequency: int = 100,
        save_dir: str = "./gradient_logs",
        track_adapter_only: bool = True,
        peft_type: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the GradientFlowTracker.

        Args:
            model: The model with PEFT adapters
            log_frequency: Log gradient norms every N training steps
            save_dir: Directory to save gradient history
            track_adapter_only: Only track adapter parameters (not base model)
            peft_type: PEFT type (for component-specific tracking)
            verbose: Print progress messages
        """
        self.model = model
        self.log_frequency = log_frequency
        self.save_dir = save_dir
        self.track_adapter_only = track_adapter_only
        self.peft_type = peft_type
        self.verbose = verbose

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Discover adapter parameters
        self.adapter_params = self._discover_adapter_params()

        # History storage
        # gradient_flow[step] = {param_name: grad_norm}
        self.gradient_flow: Dict[int, Dict[str, float]] = {}

        # Structured by layer and component
        # layer_flow[step][layer_idx][component] = grad_norm
        self.layer_flow: Dict[int, Dict[int, Dict[str, float]]] = {}

        # Steps logged
        self.steps_logged: List[int] = []

        # Track layer indices for heatmap dimensions
        self.layer_indices: Set[int] = set()
        self.components: Set[str] = set()

        # Timing stats
        self.total_logging_time = 0.0
        self.num_log_calls = 0

        # Track warnings to avoid spam
        self._warned_no_grads = False

        if self.verbose:
            logger.info(f"[GradientFlowTracker] Initialized with {len(self.adapter_params)} adapter parameters")
            logger.info(f"[GradientFlowTracker] Logging every {log_frequency} steps to {save_dir}")

    def _discover_adapter_params(self) -> Dict[str, torch.nn.Parameter]:
        """Find all adapter parameters in the model."""
        adapter_params = {}

        # Keywords that identify adapter parameters
        adapter_keywords = [
            'lora_a', 'lora_b', 'lora_e',  # LoRA, AdaLoRA
            'magnitude', 'lora_magnitude',  # DoRA
            'vera_lambda', 'vera_a', 'vera_b',  # VeRA
            'ia3_l',  # IA3
            'adapter',  # Generic adapter
        ]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()

            if self.track_adapter_only:
                # Check if this is an adapter parameter
                is_adapter = any(kw in name_lower for kw in adapter_keywords)
                if is_adapter:
                    adapter_params[name] = param
            else:
                # Track all parameters
                adapter_params[name] = param

        return adapter_params

    def _extract_layer_idx(self, param_name: str) -> int:
        """
        Extract layer index from parameter name.

        Examples:
        - "model.layers.10.self_attn.q_proj.lora_A" -> 10
        - "transformer.h.5.mlp.lora_B" -> 5
        - "base_model.model.model.layers.15.mlp.down_proj.lora_A.default.weight" -> 15
        """
        # Try various patterns for layer indices
        patterns = [
            r'layers?\.(\d+)',  # layers.10 or layer.10
            r'\.h\.(\d+)',  # transformer.h.5
            r'block\.(\d+)',  # block.5
            r'encoder\.(\d+)',  # encoder.5
            r'decoder\.(\d+)',  # decoder.5
        ]

        for pattern in patterns:
            match = re.search(pattern, param_name, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return -1  # Unknown layer

    def _extract_component(self, param_name: str) -> str:
        """
        Extract component type from parameter name.

        Examples:
        - "...lora_A.weight" -> "A"
        - "...lora_B.default.weight" -> "B"
        - "...lora_magnitude_vector" -> "magnitude"
        - "...lora_E" -> "E"
        - "...vera_lambda_b" -> "lambda_b"
        - "...vera_lambda_d" -> "lambda_d"
        """
        name_lower = param_name.lower()

        if 'lora_a' in name_lower:
            return 'A'
        elif 'lora_b' in name_lower:
            return 'B'
        elif 'lora_e' in name_lower:
            return 'E'  # AdaLoRA importance
        elif 'magnitude' in name_lower:
            return 'magnitude'
        elif 'vera_lambda_b' in name_lower:
            return 'lambda_b'
        elif 'vera_lambda_d' in name_lower:
            return 'lambda_d'
        elif 'ia3' in name_lower:
            return 'ia3'
        else:
            return 'other'

    def _extract_module_type(self, param_name: str) -> str:
        """
        Extract module type (q_proj, k_proj, v_proj, etc.) from parameter name.
        """
        name_lower = param_name.lower()

        module_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'up_proj', 'down_proj', 'gate_proj',
                       'fc1', 'fc2', 'mlp', 'attention']

        for mt in module_types:
            if mt in name_lower:
                return mt

        return 'unknown'

    def log_step(self, step: int) -> bool:
        """
        Log gradient norms at the current step.

        IMPORTANT: Call this AFTER backward() but BEFORE optimizer.step()
        to ensure gradients are available.

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
            logger.info(f"[GradientFlowTracker] Logging gradients at step {step}")

        self.steps_logged.append(step)

        # Initialize storage for this step
        step_grads = {}
        step_layer_grads: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        num_with_grads = 0
        total_grad_norm_sq = 0.0

        # Collect gradients
        with torch.no_grad():
            for name, param in self.adapter_params.items():
                if param.grad is None:
                    continue

                num_with_grads += 1

                # Compute gradient norm
                grad_norm = param.grad.norm().item()
                total_grad_norm_sq += grad_norm ** 2

                # Store by parameter name
                step_grads[name] = grad_norm

                # Extract layer and component
                layer_idx = self._extract_layer_idx(name)
                component = self._extract_component(name)
                module_type = self._extract_module_type(name)

                # Create composite key for layer-component-module
                key = f"{component}_{module_type}" if module_type != 'unknown' else component

                # Aggregate by layer (sum if multiple params in same layer/component)
                if layer_idx >= 0:
                    self.layer_indices.add(layer_idx)
                    self.components.add(component)

                    # Store individual gradient
                    step_layer_grads[layer_idx][key] = grad_norm

                    # Also store just by component (aggregated across modules in layer)
                    if component not in step_layer_grads[layer_idx]:
                        step_layer_grads[layer_idx][component] = 0.0
                    step_layer_grads[layer_idx][component] = max(
                        step_layer_grads[layer_idx][component], grad_norm
                    )

        # Warn if no gradients found
        if num_with_grads == 0:
            if not self._warned_no_grads:
                logger.warning(
                    "[GradientFlowTracker] No gradients found! "
                    "Make sure to call log_step() AFTER backward() but BEFORE optimizer.step()"
                )
                self._warned_no_grads = True
            return False

        # Store in history
        self.gradient_flow[step] = step_grads
        self.layer_flow[step] = dict(step_layer_grads)

        # Track timing
        elapsed = time.time() - start_time
        self.total_logging_time += elapsed
        self.num_log_calls += 1

        if self.verbose:
            total_norm = np.sqrt(total_grad_norm_sq)
            logger.info(
                f"[GradientFlowTracker] Logged {num_with_grads} params, "
                f"total grad norm: {total_norm:.4f}, time: {elapsed:.3f}s"
            )

        return True

    def get_gradient_matrix(
        self,
        component: str = 'B',
        module_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Get gradient data as a matrix for heatmap visualization.

        Args:
            component: Which component ('A', 'B', 'magnitude', 'E', etc.)
            module_type: Optional filter by module type ('q_proj', 'mlp', etc.)

        Returns:
            Tuple of (matrix, layer_indices, steps)
            - matrix: 2D numpy array (num_layers × num_steps)
            - layer_indices: List of layer indices (row labels)
            - steps: List of step numbers (column labels)
        """
        if not self.layer_flow:
            return np.array([[]]), [], []

        # Get sorted layer indices and steps
        layer_indices = sorted(self.layer_indices)
        steps = sorted(self.steps_logged)

        # Create mapping for quick lookup
        layer_to_row = {layer: i for i, layer in enumerate(layer_indices)}

        # Create matrix
        matrix = np.zeros((len(layer_indices), len(steps)))

        for step_idx, step in enumerate(steps):
            if step not in self.layer_flow:
                continue

            for layer_idx, layer_data in self.layer_flow[step].items():
                if layer_idx not in layer_to_row:
                    continue

                row = layer_to_row[layer_idx]

                # Find matching key
                if module_type:
                    key = f"{component}_{module_type}"
                    if key in layer_data:
                        matrix[row, step_idx] = layer_data[key]
                else:
                    # Use component directly (aggregated)
                    if component in layer_data:
                        matrix[row, step_idx] = layer_data[component]

        return matrix, layer_indices, steps

    def create_heatmap(
        self,
        component: str = 'B',
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        cmap: str = 'hot',
        log_scale: bool = True,
    ) -> Optional[Any]:
        """
        Create heatmap showing gradient flow across layers and time.

        Args:
            component: Which component to visualize ('A', 'B', 'magnitude')
            save_path: Where to save the figure (None to not save)
            title: Custom title for the plot
            figsize: Figure size in inches
            cmap: Colormap name
            log_scale: Use log scale for colors (recommended for gradients)

        Returns:
            matplotlib figure object, or None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            logger.warning("[GradientFlowTracker] matplotlib not installed, cannot create heatmap")
            return None

        # Get gradient matrix
        matrix, layer_indices, steps = self.get_gradient_matrix(component)

        if matrix.size == 0:
            logger.warning(f"[GradientFlowTracker] No data for component '{component}'")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Handle log scale
        if log_scale:
            # Add small epsilon to avoid log(0)
            matrix_plot = np.log10(matrix + 1e-10)
            cbar_label = 'Log10(Gradient Norm)'
        else:
            matrix_plot = matrix
            cbar_label = 'Gradient Norm'

        # Create heatmap
        try:
            import seaborn as sns
            sns.heatmap(
                matrix_plot,
                ax=ax,
                cmap=cmap,
                cbar_kws={'label': cbar_label},
                xticklabels=50,  # Show every 50th step label
                yticklabels=True,
            )
        except ImportError:
            # Fallback to matplotlib imshow
            im = ax.imshow(matrix_plot, aspect='auto', cmap=cmap, origin='upper')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)

        # Labels
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Layer Index')

        # Set y-tick labels to actual layer indices
        ax.set_yticks(range(len(layer_indices)))
        ax.set_yticklabels(layer_indices)

        # Set x-tick labels to step numbers (sparse)
        step_interval = max(1, len(steps) // 10)
        x_ticks = range(0, len(steps), step_interval)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([steps[i] for i in x_ticks], rotation=45)

        # Title
        if title is None:
            title = f'Gradient Flow - {component} Component'
            if self.peft_type:
                title += f' ({self.peft_type.upper()})'
        ax.set_title(title)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                logger.info(f"[GradientFlowTracker] Saved heatmap to {save_path}")

        return fig

    def create_layer_comparison_plot(
        self,
        layers: Optional[List[int]] = None,
        component: str = 'B',
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create line plot comparing gradient norms across selected layers over time.

        Args:
            layers: List of layer indices to compare (None = first, middle, last)
            component: Which component ('A', 'B', etc.)
            save_path: Where to save the figure

        Returns:
            matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[GradientFlowTracker] matplotlib not installed")
            return None

        matrix, layer_indices, steps = self.get_gradient_matrix(component)

        if matrix.size == 0:
            return None

        # Select layers to plot
        if layers is None:
            # Default: first, middle, last
            if len(layer_indices) >= 3:
                layers = [layer_indices[0], layer_indices[len(layer_indices)//2], layer_indices[-1]]
            else:
                layers = layer_indices

        # Create mapping
        layer_to_row = {layer: i for i, layer in enumerate(layer_indices)}

        fig, ax = plt.subplots(figsize=(12, 6))

        for layer in layers:
            if layer not in layer_to_row:
                continue
            row = layer_to_row[layer]
            ax.plot(steps, matrix[row, :], label=f'Layer {layer}', linewidth=2)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'Gradient Flow Over Time - {component} Component')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of gradient flow."""
        if not self.gradient_flow:
            return {'error': 'No gradients logged yet'}

        # Compute stats across all logged steps
        all_grads = []
        for step_data in self.gradient_flow.values():
            all_grads.extend(step_data.values())

        all_grads = np.array(all_grads)

        # Per-component stats at final step
        component_stats = {}
        if self.steps_logged:
            final_step = max(self.steps_logged)
            if final_step in self.layer_flow:
                for layer_idx, layer_data in self.layer_flow[final_step].items():
                    for comp, grad in layer_data.items():
                        if comp not in component_stats:
                            component_stats[comp] = []
                        component_stats[comp].append(grad)

        return {
            'peft_type': self.peft_type,
            'total_steps_logged': len(self.steps_logged),
            'step_range': (min(self.steps_logged), max(self.steps_logged)) if self.steps_logged else None,
            'num_layers': len(self.layer_indices),
            'layer_range': (min(self.layer_indices), max(self.layer_indices)) if self.layer_indices else None,
            'components_tracked': list(self.components),
            'overall_stats': {
                'mean': float(np.mean(all_grads)) if len(all_grads) > 0 else 0,
                'std': float(np.std(all_grads)) if len(all_grads) > 0 else 0,
                'min': float(np.min(all_grads)) if len(all_grads) > 0 else 0,
                'max': float(np.max(all_grads)) if len(all_grads) > 0 else 0,
            },
            'component_stats_final': {
                comp: {
                    'mean': float(np.mean(vals)),
                    'max': float(np.max(vals)),
                }
                for comp, vals in component_stats.items()
            },
            'timing': {
                'total_time': self.total_logging_time,
                'avg_per_call': self.total_logging_time / max(1, self.num_log_calls),
            },
        }

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save gradient flow data to disk.

        Args:
            filepath: Path to save file. If None, uses save_dir/gradient_flow.pt

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, "gradient_flow.pt")

        # Prepare data for saving
        save_data = {
            'peft_type': self.peft_type,
            'log_frequency': self.log_frequency,
            'steps_logged': self.steps_logged,
            'layer_indices': sorted(self.layer_indices),
            'components': sorted(self.components),
            'gradient_flow': self.gradient_flow,
            'layer_flow': self.layer_flow,
            'timing': {
                'total_logging_time': self.total_logging_time,
                'num_log_calls': self.num_log_calls,
            },
        }

        # Save as PyTorch file
        torch.save(save_data, filepath)

        # Also save JSON summary
        summary_path = filepath.replace('.pt', '_summary.json')
        summary = self.get_summary_stats()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            logger.info(f"[GradientFlowTracker] Saved to {filepath}")
            logger.info(f"[GradientFlowTracker] Summary saved to {summary_path}")

        return filepath

    @staticmethod
    def load(filepath: str) -> 'GradientFlowTracker':
        """
        Load a saved GradientFlowTracker from disk.

        Note: Creates a tracker without a model reference.
        Use for analysis only, not for continued tracking.
        """
        data = torch.load(filepath, weights_only=False)

        # Create minimal tracker for analysis
        tracker = object.__new__(GradientFlowTracker)
        tracker.model = None
        tracker.log_frequency = data['log_frequency']
        tracker.save_dir = os.path.dirname(filepath)
        tracker.peft_type = data['peft_type']
        tracker.track_adapter_only = True
        tracker.verbose = False
        tracker.adapter_params = {}
        tracker.gradient_flow = data['gradient_flow']
        tracker.layer_flow = data['layer_flow']
        tracker.steps_logged = data['steps_logged']
        tracker.layer_indices = set(data['layer_indices'])
        tracker.components = set(data['components'])
        tracker.total_logging_time = data['timing']['total_logging_time']
        tracker.num_log_calls = data['timing']['num_log_calls']
        tracker._warned_no_grads = False

        return tracker

    def get_layer_history(self, layer_idx: int) -> Dict[str, List[float]]:
        """
        Get gradient history for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Dict of {component: [grad_norms_over_time]}
        """
        history: Dict[str, List[float]] = defaultdict(list)

        for step in sorted(self.steps_logged):
            if step in self.layer_flow and layer_idx in self.layer_flow[step]:
                layer_data = self.layer_flow[step][layer_idx]
                for comp, grad in layer_data.items():
                    history[comp].append(grad)
            else:
                # Fill with 0 if no data
                for comp in self.components:
                    history[comp].append(0.0)

        history['steps'] = list(sorted(self.steps_logged))
        return dict(history)


def create_gradient_tracker(
    model: torch.nn.Module,
    args: Any,
    save_dir: Optional[str] = None,
) -> GradientFlowTracker:
    """
    Factory function to create GradientFlowTracker from training args.

    Args:
        model: The PEFT model
        args: TrainConfig object with tracker settings
        save_dir: Override save directory

    Returns:
        Configured GradientFlowTracker instance
    """
    tracker_config = getattr(args, 'tracker', None)

    if tracker_config is None:
        log_frequency = 100
        track_adapter_only = True
    else:
        log_frequency = getattr(tracker_config, 'gradient_log_frequency', 100)
        track_adapter_only = getattr(tracker_config, 'track_adapter_only', True)

    if save_dir is None:
        save_dir = os.path.join(args.training.output_dir, 'gradient_logs')

    peft_type = getattr(args.peft, 'type', None)

    return GradientFlowTracker(
        model=model,
        log_frequency=log_frequency,
        save_dir=save_dir,
        track_adapter_only=track_adapter_only,
        peft_type=peft_type,
        verbose=not getattr(args.common, 'debug', False),
    )
