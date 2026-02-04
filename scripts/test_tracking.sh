#!/bin/bash
# Phase 1C: Test training with mechanistic tracking enabled
# Run on GPU machine (A100/H100) to verify tracking works with real training

set -e

# Configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR="test_tracking_output"
MAX_STEPS=100
TRACKING_FREQUENCY=25

# Attention implementation: "flash_attention_2", "sdpa", or "eager"
# Use "sdpa" if flash-attention has compatibility issues (common on some systems)
# The training script will auto-fallback to sdpa if flash_attention_2 fails
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

echo "============================================================"
echo "Phase 1C: Training with Mechanistic Tracking"
echo "============================================================"
echo ""
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Max steps: $MAX_STEPS"
echo "Tracking frequency: $TRACKING_FREQUENCY"
echo "Attention implementation: $ATTN_IMPL"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Clean up previous test output
rm -rf $OUTPUT_DIR

# Run training with tracking enabled
python run.py \
    --enable_tracking \
    --tracking_frequency $TRACKING_FREQUENCY \
    --config.model.model_name_or_path "$MODEL" \
    --config.model.attn_implementation "$ATTN_IMPL" \
    --config.dataset.dataset_name_or_path "$DATASET" \
    --config.peft.type lora \
    --config.peft.r 16 \
    --config.peft.lora_alpha 32 \
    --config.training.max_steps $MAX_STEPS \
    --config.training.output_dir "$OUTPUT_DIR" \
    --config.training.save_steps 50 \
    --config.training.logging_steps 10 \
    --config.training.per_device_train_batch_size 1 \
    --config.training.gradient_accumulation_steps 8 \
    --config.training.num_generations 8 \
    --config.training.use_vllm false \
    --config.training.use_liger_kernel false \
    --config.common.debug true

echo ""
echo "============================================================"
echo "Verifying tracking output..."
echo "============================================================"

# Check for spectral tracking files
if [ -f "$OUTPUT_DIR/spectral_history_final.pt" ]; then
    echo "✓ Spectral history saved: $OUTPUT_DIR/spectral_history_final.pt"
else
    echo "✗ Spectral history NOT found!"
    exit 1
fi

# Check for gradient tracking files
if [ -f "$OUTPUT_DIR/gradient_flow_final.pt" ]; then
    echo "✓ Gradient flow saved: $OUTPUT_DIR/gradient_flow_final.pt"
else
    echo "✗ Gradient flow NOT found!"
    exit 1
fi

# Check for logs directories
if [ -d "$OUTPUT_DIR/spectral_logs" ]; then
    echo "✓ Spectral logs directory exists"
    ls -la "$OUTPUT_DIR/spectral_logs/"
else
    echo "✗ Spectral logs directory NOT found!"
fi

if [ -d "$OUTPUT_DIR/gradient_logs" ]; then
    echo "✓ Gradient logs directory exists"
    ls -la "$OUTPUT_DIR/gradient_logs/"
else
    echo "✗ Gradient logs directory NOT found!"
fi

echo ""
echo "============================================================"
echo "Validating saved data..."
echo "============================================================"

# Validate the saved files with Python
python -c "
import torch
import os

# Load spectral data
spectral_path = '$OUTPUT_DIR/spectral_history_final.pt'
spectral_data = torch.load(spectral_path, weights_only=False)
print(f'Spectral data:')
print(f'  - Steps logged: {len(spectral_data.get(\"steps_logged\", []))}')
print(f'  - Layers tracked: {len(spectral_data.get(\"layer_names\", []))}')
print(f'  - PEFT type: {spectral_data.get(\"peft_type\", \"unknown\")}')

# Load gradient data
gradient_path = '$OUTPUT_DIR/gradient_flow_final.pt'
gradient_data = torch.load(gradient_path, weights_only=False)
print(f'')
print(f'Gradient data:')
print(f'  - Steps logged: {len(gradient_data.get(\"steps_logged\", []))}')
print(f'  - Layers tracked: {len(gradient_data.get(\"layer_indices\", []))}')
print(f'  - Components: {gradient_data.get(\"components\", [])}')

print('')
print('All tracking data validated successfully!')
"

echo ""
echo "============================================================"
echo "Phase 1C Test Complete!"
echo "============================================================"
echo ""
echo "Tracking files are saved in: $OUTPUT_DIR"
echo ""
echo "To analyze the results, you can use:"
echo "  python -c \"from perl.trackers import SpectralTracker; t = SpectralTracker.load('$OUTPUT_DIR/spectral_history_final.pt'); print(t.get_summary_stats())\""
echo ""
