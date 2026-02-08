#!/bin/bash
# ==============================================================================
# TEST 7: Pre-Production Validation Run
# ==============================================================================
#
# Purpose: Full-scale validation before production runs
# - Full 1024-step training (not toy 10-50 steps)
# - Real DAPO-Math-17k dataset (not Countdown)
# - All tracking enabled (spectral + gradient)
# - Expected: ~3-4 hours on 1x A100
#
# This test catches issues that only appear in long runs and validates
# tracking overhead is acceptable at scale.
#
# Usage:
#   bash scripts/run_preproduction_test.sh
#
# Requirements:
#   - CUDA-capable GPU (A100 recommended)
#   - ~40GB GPU memory
#   - WandB account configured
#
# ==============================================================================

set -e  # Exit on error

echo "============================================================"
echo "TEST 7: Pre-Production Validation Run"
echo "============================================================"
echo "Date: $(date)"
echo ""

# Configuration
CONFIG_FILE="configs/experiments/test_preproduction.yaml"
EXPERIMENT_NAME="dora_preproduction"
OUTPUT_DIR="output/preproduction/${EXPERIMENT_NAME}"

# Check requirements
echo "1. Checking requirements..."

# Check CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  ⚠️  WARNING: CUDA not available. This test requires a GPU."
    echo "     On Mac, you can validate the config but not run training."
    echo ""
    read -p "Continue with validation only? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    VALIDATE_ONLY=true
else
    echo "  ✓ CUDA available"
    python -c "import torch; print(f'    GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
    VALIDATE_ONLY=false
fi

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "  ✗ Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "  ✓ Config file exists"

# Check WandB
if ! python -c "import wandb" 2>/dev/null; then
    echo "  ⚠️  WARNING: wandb not installed"
else
    echo "  ✓ wandb available"
fi

echo ""

# Validate configuration
echo "2. Validating configuration..."
python scripts/validate_configs.py

echo ""
echo "3. Pre-production test configuration:"
echo "   Config: $CONFIG_FILE"
echo "   Experiment: $EXPERIMENT_NAME"
echo "   Output: $OUTPUT_DIR"
echo ""
echo "   Settings:"
echo "   - Model: DeepSeek-R1-Distill-Qwen-1.5B"
echo "   - Dataset: DAPO-Math-17k"
echo "   - Max steps: 1024"
echo "   - Batch size: 4 × 8 = 32 effective"
echo "   - Rank: 32, Alpha: 64"
echo "   - Tracking: Spectral + Gradient every 100 steps"
echo "   - Checkpoints: Every 256 steps (4 total)"
echo ""
echo "   Estimated time: ~3-4 hours on A100"
echo ""

if [ "$VALIDATE_ONLY" = true ]; then
    echo "============================================================"
    echo "VALIDATION ONLY MODE (no GPU)"
    echo "============================================================"
    echo ""
    echo "Config validated successfully!"
    echo ""
    echo "To run on your Linux A100 machine:"
    echo ""
    echo "  # 1. Transfer the code"
    echo "  rsync -avz --exclude 'output' --exclude '.git' \\"
    echo "      /path/to/PeRL user@linux-server:/path/to/PeRL"
    echo ""
    echo "  # 2. SSH to server"
    echo "  ssh user@linux-server"
    echo ""
    echo "  # 3. Run the test"
    echo "  cd /path/to/PeRL"
    echo "  bash scripts/run_preproduction_test.sh"
    echo ""
    echo "============================================================"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run pre-production test
echo "4. Starting pre-production training..."
echo ""
echo "   This will take ~3-4 hours on A100."
echo "   Monitor progress at: https://wandb.ai"
echo ""

# Parse the config and run
# Note: This assumes you have a run_experiments.py or similar script
# that can parse YAML configs and run training

# Option A: Using the existing PeRL training infrastructure
if [ -f "run.py" ]; then
    echo "   Using run.py..."
    python run.py \
        --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        --config.model.dtype "bfloat16" \
        --config.model.attn_implementation "flash_attention_2" \
        --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
        --config.peft.use_peft true \
        --config.peft.type "dora" \
        --config.peft.task_type "CAUSAL_LM" \
        --config.peft.r 32 \
        --config.peft.lora_alpha 64 \
        --config.peft.lora_dropout 0.05 \
        --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
        --config.training.max_steps 1024 \
        --config.training.per_device_train_batch_size 4 \
        --config.training.gradient_accumulation_steps 8 \
        --config.training.learning_rate 1e-5 \
        --config.training.warmup_ratio 0.1 \
        --config.training.save_strategy "steps" \
        --config.training.save_steps 256 \
        --config.training.logging_steps 10 \
        --config.training.max_completion_length 16384 \
        --config.training.num_generations 8 \
        --config.training.max_prompt_length 512 \
        --config.training.use_vllm true \
        --config.training.vllm_mode "colocate" \
        --config.training.vllm_gpu_memory_utilization 0.4 \
        --config.training.use_liger_kernel false \
        --config.training.loss_type "dapo" \
        --config.training.lr_scheduler_type "cosine" \
        --config.training.run_name "$EXPERIMENT_NAME" \
        --config.training.output_dir "$OUTPUT_DIR" \
        --config.training.report_to '["wandb"]' \
        --config.tracker.enable_spectral_tracking true \
        --config.tracker.enable_gradient_tracking true \
        --config.tracker.spectral_log_frequency 100 \
        --config.tracker.gradient_log_frequency 100 \
        --config.wandb.project "peft-rlvr-preproduction" \
        --config.common.seed 42

# Option B: Using accelerate with a custom training script
elif [ -f "scripts/run_experiments.py" ]; then
    echo "   Using scripts/run_experiments.py..."
    python scripts/run_experiments.py \
        --config "$CONFIG_FILE" \
        --experiment "$EXPERIMENT_NAME"

else
    echo "   ⚠️  No training script found."
    echo "   Please run manually with:"
    echo ""
    echo "   accelerate launch perl/train.py \\"
    echo "       --config.model.model_name_or_path 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' \\"
    echo "       --config.peft.type 'dora' \\"
    echo "       --config.training.max_steps 1024 \\"
    echo "       ..."
    exit 1
fi

echo ""
echo "============================================================"
echo "Pre-production test completed!"
echo "============================================================"
echo ""
echo "5. Post-test verification:"
echo ""

# Check outputs
echo "   Checking outputs..."

if [ -d "$OUTPUT_DIR" ]; then
    CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -type d -name "checkpoint-*" | wc -l | tr -d ' ')
    echo "   ✓ Checkpoints saved: $CHECKPOINT_COUNT"
else
    echo "   ✗ Output directory not found"
fi

# Check WandB
echo "   ✓ Check WandB for training curves"

echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "If this test succeeded:"
echo "  1. Review WandB logs for any anomalies"
echo "  2. Check tracking data was saved correctly"
echo "  3. Verify checkpoints are loadable"
echo "  4. You're ready for production runs!"
echo ""
echo "If this test failed:"
echo "  1. Check the error messages above"
echo "  2. Review GPU memory usage"
echo "  3. Check WandB for partial logs"
echo "  4. Fix issues before production"
echo ""
echo "============================================================"
