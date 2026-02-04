# Copyright (c) 2025
# Training callbacks for mechanistic analysis during RLVR training

import os
import torch
import logging
from typing import Optional, Any, Dict
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .spectral_tracker import SpectralTracker

logger = logging.getLogger(__name__)


class SpectralTrackingCallback(TrainerCallback):
    """
    Hugging Face Trainer callback for spectral tracking during training.

    This callback integrates SpectralTracker with TRL's GRPOTrainer,
    automatically logging spectral metrics at specified intervals.

    Usage:
        tracker = SpectralTracker(model, log_frequency=100)
        callback = SpectralTrackingCallback(tracker)
        trainer = GRPOTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        tracker: Optional[SpectralTracker] = None,
        log_frequency: int = 100,
        save_dir: Optional[str] = None,
        save_on_train_end: bool = True,
        log_to_wandb: bool = True,
    ):
        """
        Initialize the callback.

        Args:
            tracker: Pre-configured SpectralTracker. If None, will be created on_train_begin.
            log_frequency: How often to log (only used if tracker is None)
            save_dir: Where to save spectral logs (only used if tracker is None)
            save_on_train_end: Whether to save history when training ends
            log_to_wandb: Whether to log summary metrics to wandb/trackio
        """
        super().__init__()
        self.tracker = tracker
        self.log_frequency = log_frequency
        self.save_dir = save_dir
        self.save_on_train_end = save_on_train_end
        self.log_to_wandb = log_to_wandb
        self._wandb = None
        self._trackio = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """Initialize tracker if not provided, set up logging."""
        # Create tracker if not provided
        if self.tracker is None and model is not None:
            save_dir = self.save_dir or os.path.join(args.output_dir, 'spectral_logs')
            self.tracker = SpectralTracker(
                model=model,
                log_frequency=self.log_frequency,
                save_dir=save_dir,
            )
            logger.info(f"[SpectralTrackingCallback] Created SpectralTracker with {len(self.tracker.adapter_layers)} layers")

        # Set up wandb/trackio logging
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    self._wandb = wandb
                    logger.info("[SpectralTrackingCallback] WandB logging enabled")
            except ImportError:
                pass

            try:
                import trackio
                if hasattr(trackio, 'is_initialized') and trackio.is_initialized():
                    self._trackio = trackio
                    logger.info("[SpectralTrackingCallback] Trackio logging enabled")
            except ImportError:
                pass

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """Log spectral metrics at the end of each step."""
        if self.tracker is None:
            return

        # Log spectral metrics
        logged = self.tracker.log_step(state.global_step)

        # Log summary to wandb/trackio
        if logged and (self._wandb or self._trackio):
            self._log_to_experiment_tracker(state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save spectral history when training ends."""
        if self.tracker is None:
            return

        if self.save_on_train_end:
            save_path = self.tracker.save()
            logger.info(f"[SpectralTrackingCallback] Saved spectral history to {save_path}")

        # Log final summary
        summary = self.tracker.get_summary_stats()
        logger.info(f"[SpectralTrackingCallback] Final summary: {summary}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Optionally save spectral data with each checkpoint."""
        if self.tracker is None:
            return

        # Save intermediate spectral history
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            spectral_path = os.path.join(checkpoint_dir, "spectral_history.pt")
            self.tracker.save(spectral_path)

    def _log_to_experiment_tracker(self, step: int):
        """Log summary metrics to wandb/trackio."""
        if self.tracker is None:
            return

        # Get latest metrics across layers
        metrics_to_log = {}

        for metric_name in ['effective_rank', 'condition_number', 'spectral_gap', 'frobenius_norm']:
            values = self.tracker.get_metric_across_layers(metric_name, step_idx=-1)
            if values:
                vals = list(values.values())
                metrics_to_log[f'spectral/{metric_name}_mean'] = sum(vals) / len(vals)
                metrics_to_log[f'spectral/{metric_name}_max'] = max(vals)
                metrics_to_log[f'spectral/{metric_name}_min'] = min(vals)

        # Log DoRA-specific metrics if available
        if self.tracker.peft_type == 'dora':
            mag_norms = self.tracker.get_metric_across_layers('magnitude_norm', step_idx=-1)
            if mag_norms:
                vals = list(mag_norms.values())
                metrics_to_log['spectral/dora_magnitude_mean'] = sum(vals) / len(vals)

        # Log to wandb
        if self._wandb:
            self._wandb.log(metrics_to_log, step=step)

        # Log to trackio
        if self._trackio:
            self._trackio.log(metrics_to_log, step=step)


class GradientFlowCallback(TrainerCallback):
    """
    Callback to track gradient norms by layer during training using GradientFlowTracker.

    This complements SpectralTracker by providing gradient flow analysis.
    Creates heatmap visualizations showing gradient magnitude across layers and time.

    Usage:
        from perl.trackers import GradientFlowTracker, GradientFlowCallback
        tracker = GradientFlowTracker(model, log_frequency=100)
        callback = GradientFlowCallback(tracker)
        trainer = GRPOTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        tracker: Optional['GradientFlowTracker'] = None,
        log_frequency: int = 100,
        save_dir: Optional[str] = None,
        track_adapter_only: bool = True,
        peft_type: Optional[str] = None,
        save_on_train_end: bool = True,
        log_to_wandb: bool = True,
        create_heatmaps: bool = True,
    ):
        """
        Initialize gradient flow tracking callback.

        Args:
            tracker: Pre-configured GradientFlowTracker. If None, created on_train_begin.
            log_frequency: How often to log gradient norms (only used if tracker is None)
            save_dir: Where to save gradient history (only used if tracker is None)
            track_adapter_only: Only track gradients of adapter parameters
            peft_type: PEFT type for tracker (only used if tracker is None)
            save_on_train_end: Whether to save history when training ends
            log_to_wandb: Whether to log summary metrics to wandb/trackio
            create_heatmaps: Whether to create heatmap visualizations on train end
        """
        super().__init__()
        self.tracker = tracker
        self.log_frequency = log_frequency
        self.save_dir = save_dir
        self.track_adapter_only = track_adapter_only
        self.peft_type = peft_type
        self.save_on_train_end = save_on_train_end
        self.log_to_wandb = log_to_wandb
        self.create_heatmaps = create_heatmaps
        self._wandb = None
        self._trackio = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """Initialize tracker if not provided, set up logging."""
        # Import here to avoid circular imports
        from .gradient_tracker import GradientFlowTracker

        # Create tracker if not provided
        if self.tracker is None and model is not None:
            save_dir = self.save_dir or os.path.join(args.output_dir, 'gradient_logs')
            self.tracker = GradientFlowTracker(
                model=model,
                log_frequency=self.log_frequency,
                save_dir=save_dir,
                track_adapter_only=self.track_adapter_only,
                peft_type=self.peft_type,
            )
            logger.info(f"[GradientFlowCallback] Created GradientFlowTracker with {len(self.tracker.adapter_params)} parameters")

        # Set up wandb/trackio logging
        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    self._wandb = wandb
                    logger.info("[GradientFlowCallback] WandB logging enabled")
            except ImportError:
                pass

            try:
                import trackio
                if hasattr(trackio, 'is_initialized') and trackio.is_initialized():
                    self._trackio = trackio
                    logger.info("[GradientFlowCallback] Trackio logging enabled")
            except ImportError:
                pass

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """Log gradient norms at the end of each step."""
        if self.tracker is None:
            return

        # Log gradient norms (tracker handles frequency check internally)
        logged = self.tracker.log_step(state.global_step)

        # Log summary to wandb/trackio
        if logged and (self._wandb or self._trackio):
            self._log_to_experiment_tracker(state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save gradient history and create visualizations when training ends."""
        if self.tracker is None:
            return

        if self.save_on_train_end:
            save_path = self.tracker.save()
            logger.info(f"[GradientFlowCallback] Saved gradient history to {save_path}")

        # Create heatmap visualizations
        if self.create_heatmaps:
            self._create_visualizations()

        # Log final summary
        summary = self.tracker.get_summary_stats()
        logger.info(f"[GradientFlowCallback] Final summary: {summary}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Optionally save gradient data with each checkpoint."""
        if self.tracker is None:
            return

        # Save intermediate gradient history
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            gradient_path = os.path.join(checkpoint_dir, "gradient_flow.pt")
            self.tracker.save(gradient_path)

    def _log_to_experiment_tracker(self, step: int):
        """Log summary metrics to wandb/trackio."""
        if self.tracker is None:
            return

        # Get latest metrics
        metrics_to_log = {}
        summary = self.tracker.get_summary_stats()

        # Overall gradient statistics
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            metrics_to_log['gradient/mean'] = stats.get('mean', 0)
            metrics_to_log['gradient/max'] = stats.get('max', 0)

        # Per-component stats (e.g., A vs B gradients)
        if 'component_stats_final' in summary:
            for comp, comp_stats in summary['component_stats_final'].items():
                if isinstance(comp_stats, dict):
                    metrics_to_log[f'gradient/{comp}_mean'] = comp_stats.get('mean', 0)
                    metrics_to_log[f'gradient/{comp}_max'] = comp_stats.get('max', 0)

        # Log to wandb
        if self._wandb and metrics_to_log:
            self._wandb.log(metrics_to_log, step=step)

        # Log to trackio
        if self._trackio and metrics_to_log:
            self._trackio.log(metrics_to_log, step=step)

    def _create_visualizations(self):
        """Create heatmap visualizations for gradient flow."""
        if self.tracker is None:
            return

        save_dir = self.tracker.save_dir

        # Create heatmaps for main components
        for component in ['A', 'B']:
            try:
                heatmap_path = os.path.join(save_dir, f'gradient_heatmap_{component}.png')
                fig = self.tracker.create_heatmap(
                    component=component,
                    save_path=heatmap_path,
                )
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    logger.info(f"[GradientFlowCallback] Created heatmap: {heatmap_path}")
            except Exception as e:
                logger.warning(f"[GradientFlowCallback] Failed to create {component} heatmap: {e}")

        # DoRA-specific: magnitude gradient heatmap
        if self.tracker.peft_type == 'dora':
            try:
                heatmap_path = os.path.join(save_dir, 'gradient_heatmap_magnitude.png')
                fig = self.tracker.create_heatmap(
                    component='magnitude',
                    save_path=heatmap_path,
                )
                if fig is not None:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    logger.info(f"[GradientFlowCallback] Created DoRA magnitude heatmap: {heatmap_path}")
            except Exception as e:
                logger.warning(f"[GradientFlowCallback] Failed to create magnitude heatmap: {e}")

        # Create layer comparison plot
        try:
            comparison_path = os.path.join(save_dir, 'gradient_layer_comparison.png')
            fig = self.tracker.create_layer_comparison_plot(
                save_path=comparison_path,
            )
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.close(fig)
                logger.info(f"[GradientFlowCallback] Created layer comparison plot: {comparison_path}")
        except Exception as e:
            logger.warning(f"[GradientFlowCallback] Failed to create layer comparison: {e}")


def create_tracking_callbacks(
    model: torch.nn.Module,
    args: Any,
    enable_spectral: bool = True,
    enable_gradient: bool = False,
    log_to_wandb: bool = True,
) -> list:
    """
    Factory function to create all tracking callbacks.

    Args:
        model: The PEFT model
        args: TrainConfig with tracker settings
        enable_spectral: Enable spectral tracking
        enable_gradient: Enable gradient flow tracking
        log_to_wandb: Whether to log metrics to wandb/trackio

    Returns:
        List of configured callbacks
    """
    callbacks = []

    # Get tracker config
    tracker_config = getattr(args, 'tracker', None)
    peft_type = getattr(args.peft, 'type', None)
    track_adapter_only = True
    if tracker_config:
        track_adapter_only = getattr(tracker_config, 'track_adapter_only', True)

    if enable_spectral:
        log_freq = 100
        if tracker_config:
            log_freq = getattr(tracker_config, 'spectral_log_frequency', 100)

        save_dir = os.path.join(args.training.output_dir, 'spectral_logs')

        tracker = SpectralTracker(
            model=model,
            log_frequency=log_freq,
            save_dir=save_dir,
            peft_type=peft_type,
        )
        callbacks.append(SpectralTrackingCallback(
            tracker=tracker,
            log_to_wandb=log_to_wandb,
        ))

    if enable_gradient:
        from .gradient_tracker import GradientFlowTracker

        log_freq = 100
        if tracker_config:
            log_freq = getattr(tracker_config, 'gradient_log_frequency', 100)

        save_dir = os.path.join(args.training.output_dir, 'gradient_logs')

        tracker = GradientFlowTracker(
            model=model,
            log_frequency=log_freq,
            save_dir=save_dir,
            track_adapter_only=track_adapter_only,
            peft_type=peft_type,
        )
        callbacks.append(GradientFlowCallback(
            tracker=tracker,
            log_to_wandb=log_to_wandb,
            create_heatmaps=True,
        ))

    return callbacks
