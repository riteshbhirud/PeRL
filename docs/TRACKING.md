# Mechanistic Tracking for PEFT Methods

This document describes the spectral and gradient flow tracking extensions added in Phase 1 of the PeRL mechanistic analysis project.

## Quick Start

### Enable tracking with convenience flags:
```bash
python run.py \
    --enable_tracking \
    --tracking_frequency 100 \
    --config.model.model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --config.dataset.dataset_name_or_path Jiayi-Pan/Countdown-Tasks-3to4 \
    --config.peft.type lora \
    --config.training.output_dir output_with_tracking
```

### Enable only spectral tracking:
```bash
python run.py \
    --enable_spectral_tracking \
    --tracking_frequency 50 \
    ...
```

### Enable only gradient tracking:
```bash
python run.py \
    --enable_gradient_tracking \
    --tracking_frequency 50 \
    ...
```

### Using standard config flags:
```bash
python run.py \
    --config.tracker.enable_spectral_tracking true \
    --config.tracker.enable_gradient_tracking true \
    --config.tracker.spectral_log_frequency 100 \
    --config.tracker.gradient_log_frequency 100 \
    ...
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--enable_tracking` | Enable both spectral and gradient tracking |
| `--enable_spectral_tracking` | Enable only spectral tracking |
| `--enable_gradient_tracking` | Enable only gradient tracking |
| `--tracking_frequency N` | Log every N steps (default: 100) |
| `--tracking_output_dir DIR` | Custom output directory for tracking data |

## Config Options

All tracking options are in `config.tracker.*`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_spectral_tracking` | bool | False | Enable spectral analysis of adapter weights |
| `enable_gradient_tracking` | bool | False | Enable gradient flow tracking |
| `spectral_log_frequency` | int | 100 | Log spectral metrics every N steps |
| `gradient_log_frequency` | int | 100 | Log gradient norms every N steps |
| `track_adapter_only` | bool | True | Only track adapter parameters |
| `compute_full_svd` | bool | True | Compute full SVD (False = truncated) |
| `max_layers_to_track` | int | None | Limit layers to track (None = all) |
| `save_with_checkpoints` | bool | True | Save tracking data with checkpoints |
| `tracking_output_dir` | str | None | Custom output directory |

## Output Files

Training with tracking enabled produces:

```
output_dir/
├── spectral_logs/
│   ├── spectral_history.pt        # Full spectral data
│   └── spectral_history_summary.json
├── gradient_logs/
│   ├── gradient_flow.pt           # Full gradient data
│   ├── gradient_flow_summary.json
│   ├── gradient_heatmap_A.png     # Heatmap for lora_A gradients
│   ├── gradient_heatmap_B.png     # Heatmap for lora_B gradients
│   └── gradient_layer_comparison.png
├── spectral_history_final.pt      # Final spectral snapshot
└── gradient_flow_final.pt         # Final gradient snapshot
```

## Analyzing Results

### Load spectral data:
```python
from perl.trackers import SpectralTracker

tracker = SpectralTracker.load('output_dir/spectral_history_final.pt')
summary = tracker.get_summary_stats()
print(summary)

# Get layer-specific history
layer_history = tracker.get_layer_history('model.layers.0.self_attn.q_proj')
```

### Load gradient data:
```python
from perl.trackers import GradientFlowTracker

tracker = GradientFlowTracker.load('output_dir/gradient_flow_final.pt')
summary = tracker.get_summary_stats()
print(summary)

# Create heatmap
tracker.create_heatmap(component='B', save_path='gradient_heatmap.png')
```

## Metrics Tracked

### Spectral Metrics (per layer, per step)
- `singular_values`: Full spectrum of adapter weight matrix
- `spectral_gap`: Difference between largest and smallest singular values
- `condition_number`: Ratio of max to min singular values
- `effective_rank`: Measure of how many dimensions are "active"
- `nuclear_norm`: Sum of singular values
- `frobenius_norm`: L2 norm of weight matrix
- `top1_sv_ratio`, `top5_sv_ratio`, `top10_sv_ratio`: Energy in top-k singular values

### DoRA-Specific Metrics
- `magnitude_norm`: Norm of magnitude vector
- `direction_norm`: Norm of direction component
- `magnitude_direction_ratio`: Ratio of the two

### AdaLoRA-Specific Metrics
- `importance_scores`: Per-rank importance scores

### Gradient Metrics (per layer, per step)
- Gradient norms for `lora_A`, `lora_B` components
- Gradient norms for `magnitude` (DoRA)
- Per-module breakdown (q_proj, k_proj, v_proj, etc.)

## Attention Implementation

If you encounter flash-attention compatibility issues, you can use SDPA instead:

```bash
python run.py \
    --config.model.attn_implementation sdpa \
    ...
```

Options: `flash_attention_2` (default), `sdpa`, `eager`

The training script will automatically fallback to SDPA if flash_attention_2 fails to load.

## Performance Overhead

Expected overhead with tracking enabled:
- ~5-10% for typical training runs
- Overhead scales with number of tracked layers
- Use `--tracking_frequency 100` or higher to minimize impact

## Supported PEFT Methods

Tracking works with all PEFT methods in PeRL:
- LoRA, DoRA, AdaLoRA
- PiSSA, MiLoRA, MiLoRA+
- LoRA+, rsLoRA, LoRA-FA
- VeRA, IA3, LayerNorm Tuning

## Testing

Validate setup without GPU:
```bash
python scripts/validate_tracking_setup.py
```

Run integration tests:
```bash
python -m perl.trackers.test_integration
```

Full training test (requires GPU):
```bash
bash scripts/test_tracking.sh
```
