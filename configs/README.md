# PeRL Experiment Configurations

This directory contains YAML configuration files for running mechanistic analysis experiments with PeRL.

## Quick Start

```bash
# Validate all configs
python scripts/validate_configs.py

# Generate/regenerate configs
python scripts/generate_experiment_configs.py

# List experiments in a config
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --list

# Dry run (preview without executing)
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --dry_run

# Run specific experiments
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
    --only "lora_r16_s42,dora_r16_s42"
```

## Directory Structure

```
configs/
├── README.md                    # This file
├── experiments/
│   ├── core_1.5B.yaml          # 24 experiments: 8 methods × 3 seeds (1.5B model)
│   ├── core_7B.yaml            # 15 experiments: 5 methods × 3 seeds (7B model)
│   ├── stress_rank.yaml        # 32 experiments: 4 methods × 4 ranks × 2 seeds
│   └── stress_data.yaml        # 32 experiments: 4 methods × 4 data sizes × 2 seeds
└── math.py                     # Evaluation benchmark configs
```

## Experiment Summary

| Config File | Model | Methods | Variables | Experiments |
|------------|-------|---------|-----------|-------------|
| core_1.5B.yaml | 1.5B | 8 | 3 seeds | 24 |
| core_7B.yaml | 7B | 5 | 3 seeds | 15 |
| stress_rank.yaml | 1.5B | 4 | 4 ranks × 2 seeds | 32 |
| stress_data.yaml | 1.5B | 4 | 4 data sizes × 2 seeds | 32 |
| **Total** | | | | **103** |

## Config Format

Each YAML config file has two sections:

### Base Section
Shared settings inherited by all experiments:

```yaml
base:
  description: "Human-readable description"
  model:
    model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    dtype: bfloat16
    attn_implementation: sdpa  # or flash_attention_2, eager
  dataset:
    dataset_name_or_path: Jiayi-Pan/Countdown-Tasks-3to4
  training:
    max_steps: 1000
    save_steps: 100
    logging_steps: 10
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 8
    num_generations: 8
    use_vllm: false
    use_liger_kernel: false
    report_to:
      - wandb
  tracker:
    enable_spectral_tracking: true
    enable_gradient_tracking: true
    spectral_log_frequency: 100
    gradient_log_frequency: 100
  wandb:
    use_wandb: true
    project: peft-rlvr-mechanistic
    log_spectral_images: true
    log_gradient_images: true
```

### Experiments Section
List of experiments that override base settings:

```yaml
experiments:
  - name: lora_r16_s42
    peft:
      type: lora
      r: 16
      lora_alpha: 32
    training:
      learning_rate: 1e-05
    common:
      seed: 42
```

## PEFT Methods

The following PEFT methods are supported:

| Method | Type | Description |
|--------|------|-------------|
| LoRA | `lora` | Standard Low-Rank Adaptation |
| DoRA | `dora` | Weight-Decomposed LoRA |
| AdaLoRA | `adalora` | Adaptive rank allocation |
| PiSSA | `pissa` | Principal Singular values init |
| MiLoRA | `milora` | Minor components init |
| VeRA | `vera` | Vector-based Random Matrix Adaptation |
| LoRA+ | `lora_plus` | Different LR for A/B matrices |
| MiSS | `miss` | MiSS adapter |

## Experiment Variables

### Seeds
- **Core experiments**: 42, 123, 456 (3 seeds)
- **Stress tests**: 42, 123 (2 seeds)

### Ranks (stress_rank.yaml)
- 4, 16, 64, 256

### Data Sizes (stress_data.yaml)
- 1k, 4k, 8.5k, 17k examples

## Default Parameters

Based on Yin et al. paper specifications:

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 (2 × rank) |
| Learning Rate | 1e-5 (1e-4 for VeRA) |
| Max Steps | 1000 |
| Batch Size (1.5B) | 4 |
| Batch Size (7B) | 2 |
| Gradient Accumulation | 8 |
| Tracking Frequency | 100 steps |

## Validation

Run the validation script to check configs:

```bash
# Validate all configs
python scripts/validate_configs.py

# Validate specific config
python scripts/validate_configs.py --config configs/experiments/core_1.5B.yaml

# Verbose output
python scripts/validate_configs.py --verbose
```

The validator checks:
- Required fields presence
- Valid PEFT method types
- Parameter ranges (rank, alpha, learning rate)
- Unique experiment names
- YAML syntax correctness

## Regenerating Configs

To regenerate all config files:

```bash
# Generate with defaults
python scripts/generate_experiment_configs.py

# Preview without writing
python scripts/generate_experiment_configs.py --dry-run

# Custom output directory
python scripts/generate_experiment_configs.py --output-dir my_configs/
```

## Running Experiments

Use the `run_experiments.py` script to execute experiments from config files:

```bash
# List all experiments in a config
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --list

# Dry run (preview commands without executing)
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --dry_run

# Run all experiments sequentially
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml

# Run specific experiments only
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
    --only "lora_r16_s42,dora_r16_s42"

# Exclude specific experiments
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
    --exclude "vera_r16_s42,vera_r16_s123"

# Resume from failures (skip completed experiments)
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
    --skip_completed

# Parallel execution on multiple GPUs
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
    --parallel 4 --gpu_ids "0,1,2,3"

# Custom output directory
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml \
    --output_dir /path/to/outputs
```

### Runner Features

- **Progress tracking**: Saves status to `run_status.json` after each experiment
- **Resume capability**: Use `--skip_completed` to resume from failures
- **Parallel execution**: Run multiple experiments concurrently with `--parallel N`
- **GPU assignment**: Specify GPUs for parallel runs with `--gpu_ids`
- **Logging**: Each experiment logs to `output/{config_name}/logs/{experiment_name}.log`
- **Graceful interruption**: Ctrl+C saves progress before exiting

## WandB Integration

All experiment configs include Weights & Biases (WandB) integration for real-time monitoring of experiments running on a remote cluster.

### Setup

```bash
# One-time setup (interactive login)
python scripts/setup_wandb.py

# Check if already logged in
python scripts/setup_wandb.py --check

# Setup for offline mode (sync later)
python scripts/setup_wandb.py --offline
```

### WandB Config in YAML

The base section includes WandB settings:

```yaml
base:
  training:
    report_to:
      - wandb
  wandb:
    use_wandb: true
    project: peft-rlvr-mechanistic
    log_spectral_images: true
    log_gradient_images: true
```

### Auto-Generated Tags and Groups

Experiments are automatically tagged and grouped for easy filtering:

**Auto-generated tags:**
- PEFT method: `lora`, `dora`, `adalora`, etc.
- Seed: `seed42`, `seed123`, etc.
- Model size: `1.5B`, `7B`
- Rank: `r4`, `r16`, `r64`, etc.
- Special conditions: `ultra_low_rank`, `high_rank`, `data_scarcity`, `spectral_tracking`

**Auto-generated groups:**
- Format: `{model_size}_{peft_type}` (e.g., `1.5B_lora`, `7B_dora`)

### Logged Metrics

**Training metrics (every step):**
- Loss, rewards, learning rate
- Gradient norms, GPU memory

**Spectral metrics (every 100 steps):**
- Singular value distributions per layer
- Effective rank, condition numbers
- SVD analysis images

**Gradient metrics (every 100 steps):**
- Gradient norms by layer (A/B matrices)
- Gradient flow statistics
- Gradient heatmap images

### Remote Monitoring

Monitor experiments from your Mac while they run on a Linux cluster:

1. **Dashboard URL:** `https://wandb.ai/{entity}/peft-rlvr-mechanistic`
2. **Filter by tags:** Click Filters → Select tags (e.g., `lora`, `seed42`)
3. **Group runs:** Runs auto-grouped by `{model_size}_{peft_type}`
4. **Compare metrics:** Select multiple runs → Click Compare

### Offline Mode

For unreliable network connections:

```bash
# Enable offline mode
export WANDB_MODE=offline

# Or in config:
# wandb:
#   offline: true

# Run experiments (data saved locally)
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml

# Sync when back online
wandb sync output/*/wandb/
```

### Disabling WandB

For local testing without WandB:

```bash
# Via debug mode (disables all reporting)
python run.py ... --config.common.debug true

# Or override report_to
python run.py ... --config.training.report_to []
```

## Output Structure

The runner creates outputs organized by config file:

```
output/{config_name}/
├── run_status.json             # Overall run status and progress
├── logs/                       # Experiment log files
│   ├── lora_r16_s42.log
│   ├── dora_r16_s42.log
│   └── ...
├── lora_r16_s42/               # Individual experiment output
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── spectral_logs/
│   │   └── spectral_history.pt
│   ├── gradient_logs/
│   │   ├── gradient_flow.pt
│   │   ├── gradient_heatmap_A.png
│   │   ├── gradient_heatmap_B.png
│   │   └── gradient_layer_comparison.png
│   ├── spectral_history_final.pt
│   └── gradient_flow_final.pt
└── dora_r16_s42/
    └── ...
```

## Customization

To add custom experiments:

1. Edit the appropriate YAML file in `configs/experiments/`
2. Add a new entry to the `experiments` list
3. Run validation: `python scripts/validate_configs.py`

Example custom experiment:

```yaml
experiments:
  # ... existing experiments ...

  # Custom experiment
  - name: custom_lora_r32_s42
    peft:
      type: lora
      r: 32
      lora_alpha: 64
    training:
      learning_rate: 5e-6
      max_steps: 2000
    common:
      seed: 42
```

## Monitoring Tools

PeRL includes several tools for monitoring experiment progress without logging into the cluster.

### Quick Status Check (Bash)

```bash
# Quick overview of experiment status
bash scripts/quick_status.sh output/core_1.5B

# Output includes:
# - Total/completed/running/failed counts
# - Progress bar
# - Recently active experiments
# - GPU status (if available)
# - Disk usage
```

### Detailed Status Check (Python)

```bash
# Full status report
python scripts/check_experiment_status.py --output_dir output/core_1.5B

# Verbose output (show all experiments)
python scripts/check_experiment_status.py --output_dir output/core_1.5B --verbose

# Watch mode (refresh every 30 seconds)
python scripts/check_experiment_status.py --output_dir output/core_1.5B --watch

# JSON output for scripting
python scripts/check_experiment_status.py --output_dir output/core_1.5B --json
```

Output example:
```
========================================
Experiment Status: output/core_1.5B
========================================
Total experiments: 24
  Completed: 3 ✓
  Running:   1 ⧗
  Failed:    0 ✗
  Not started: 20 ○

Completed:
  ✓ lora_r16_s42    (45.2 min, 2.8 GB)
  ✓ dora_r16_s42    (52.1 min, 3.1 GB)

Running:
  ⧗ pissa_r16_s42   (28 min elapsed, ~17 min remaining)

Progress: [###-------] 12.5%
Estimated completion: 2026-02-07 14:30
```

### Failure Diagnosis

```bash
# Diagnose failed experiments
python scripts/diagnose_failures.py --output_dir output/core_1.5B

# Show more log lines
python scripts/diagnose_failures.py --output_dir output/core_1.5B --lines 50

# Verbose mode (show tracebacks)
python scripts/diagnose_failures.py --output_dir output/core_1.5B --verbose
```

Features:
- Identifies error types (OOM, connection, file not found, etc.)
- Shows relevant log lines
- Provides suggested fixes for common errors
- Groups failures by error type

### Summary Report Generation

```bash
# Generate markdown summary report
python scripts/generate_summary_report.py --output_dir output/core_1.5B

# Print to stdout only
python scripts/generate_summary_report.py --output_dir output/core_1.5B --stdout
```

Generates `SUMMARY.md` with:
- Progress overview and completion percentage
- Table of completed experiments with durations and losses
- Results grouped by PEFT method
- Storage usage statistics
- Issues and failures summary

### Progress Bar in Batch Runner

The batch runner includes tqdm progress bars:

```bash
# Default: progress bar enabled
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml

# Disable progress bar (verbose output)
python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml --no_progress
```

Progress bar shows:
- Current experiment name
- Completion count and percentage
- Elapsed time and ETA
- Per-experiment completion messages

### Monitoring Workflow

1. **Start experiments** on cluster:
   ```bash
   python scripts/run_experiments.py --config configs/experiments/core_1.5B.yaml
   ```

2. **Quick check** from local machine:
   ```bash
   bash scripts/quick_status.sh output/core_1.5B
   ```

3. **Watch progress** continuously:
   ```bash
   python scripts/check_experiment_status.py -o output/core_1.5B --watch
   ```

4. **Check for failures** if progress stalls:
   ```bash
   python scripts/diagnose_failures.py -o output/core_1.5B
   ```

5. **Generate report** when complete:
   ```bash
   python scripts/generate_summary_report.py -o output/core_1.5B
   ```
