#!/bin/bash
# Setup script for PeRL on Linux with A100/H100 GPUs
# Run this script after cloning the repository

set -e

echo "============================================================"
echo "PeRL Environment Setup for Linux (A100/H100)"
echo "============================================================"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: CUDA not found. Make sure CUDA is installed."
    echo "Expected: CUDA 12.x for best compatibility"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support (for A100/H100)
echo ""
echo "Installing PyTorch with CUDA 12.x support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention (optional, can use SDPA as fallback)
echo ""
echo "Installing flash-attention..."
echo "Note: If this fails, you can skip it and use --config.model.attn_implementation sdpa"
pip install flash-attn --no-build-isolation || {
    echo "WARNING: flash-attention installation failed. Using SDPA instead."
    echo "Set --config.model.attn_implementation sdpa when running training."
}

# Install main dependencies
echo ""
echo "Installing main dependencies..."
pip install accelerate datasets deepspeed fire huggingface-hub
pip install math-verify transformers peft wandb

# Install TRL with vLLM support
echo ""
echo "Installing TRL with vLLM..."
pip install "trl[vllm]"

# Install additional dependencies for tracking
echo ""
echo "Installing tracking dependencies..."
pip install matplotlib seaborn numpy

# Install optional but recommended packages
echo ""
echo "Installing optional packages..."
pip install liger-kernel  # For optimized kernels
pip install trackio  # For experiment tracking (optional)

# Verify installation
echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

import transformers
print(f'Transformers version: {transformers.__version__}')

import peft
print(f'PEFT version: {peft.__version__}')

import trl
print(f'TRL version: {trl.__version__}')

# Check tracking modules
from perl.trackers import SpectralTracker, GradientFlowTracker
print('Tracking modules: OK')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source env/bin/activate"
echo ""
echo "To validate tracking setup (no GPU needed):"
echo "  python scripts/validate_tracking_setup.py"
echo ""
echo "To run a test training with tracking:"
echo "  bash scripts/test_tracking.sh"
echo ""
