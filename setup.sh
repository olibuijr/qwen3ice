#!/bin/bash

# Setup script for Qwen3-4B Icelandic training on RTX 3080

echo "========================================="
echo "Qwen3-4B Icelandic Training Setup"
echo "========================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected. Please ensure CUDA is installed."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Unsloth (optimized for memory-efficient training)
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install flash-attention if compatible (optional, may fail on some systems)
echo "Attempting to install Flash Attention (optional)..."
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed (optional)"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/icelandic
mkdir -p models
mkdir -p logs
mkdir -p checkpoints

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Prepare the dataset: python prepare_icelandic_dataset.py"
echo "3. Start training: python train_qwen_icelandic.py"
echo ""
echo "For testing with a small sample:"
echo "  python train_qwen_icelandic.py --max-seq-length 1024"
echo ""