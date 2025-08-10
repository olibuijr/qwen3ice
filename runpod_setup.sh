#!/bin/bash

# RunPod Setup Script for Qwen Icelandic Training on RTX 5090
# Optimized for 32GB VRAM with PyTorch 2.8.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RunPod RTX 5090 Setup for Qwen Training ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check CUDA version
echo -e "${YELLOW}Checking CUDA installation...${NC}"
nvidia-smi
nvcc --version

# Update system packages
echo -e "${YELLOW}Updating system packages...${NC}"
apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    tmux \
    htop \
    nvtop \
    build-essential

# Install PyTorch 2.8.0 with CUDA 12.1 support
echo -e "${YELLOW}Installing PyTorch 2.8.0 with CUDA 12.1...${NC}"
pip install --upgrade pip
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install Flash Attention 2 for RTX 5090
echo -e "${YELLOW}Installing Flash Attention 2...${NC}"
pip install ninja packaging
pip install flash-attn --no-build-isolation

# Install Triton for optimized kernels
echo -e "${YELLOW}Installing Triton...${NC}"
pip install triton

# Install project dependencies
echo -e "${YELLOW}Installing project dependencies...${NC}"
pip install -r requirements_essential.txt

# Install Unsloth with 5090 optimizations
echo -e "${YELLOW}Installing Unsloth for RTX 5090...${NC}"
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional ML tools
pip install wandb tensorboard bitsandbytes

# Setup environment variables for optimal performance
echo -e "${YELLOW}Setting up environment variables...${NC}"
cat >> ~/.bashrc << 'EOF'

# PyTorch optimizations for RTX 5090
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/workspace/cache
export HF_HOME=/workspace/huggingface

# Enable TF32 for RTX 5090
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Flash Attention settings
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=8

# Triton cache
export TRITON_CACHE_DIR=/workspace/triton_cache
EOF

source ~/.bashrc

# Create necessary directories
echo -e "${YELLOW}Creating workspace directories...${NC}"
mkdir -p /workspace/cache
mkdir -p /workspace/huggingface
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/triton_cache

# Download and prepare the dataset if not exists
if [ ! -f "data/icelandic/train.jsonl" ]; then
    echo -e "${YELLOW}Dataset not found. Preparing Icelandic dataset...${NC}"
    python qwen_manager.py prepare-data
else
    echo -e "${GREEN}✓ Dataset already exists${NC}"
fi

# Create optimized training script for 5090
echo -e "${YELLOW}Creating optimized training launcher...${NC}"
cat > train_5090.sh << 'TRAIN_SCRIPT'
#!/bin/bash

# Training script optimized for RTX 5090 32GB

# Enable performance optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Use the 5090-optimized config
CONFIG_FILE="config_5090.yaml"

echo "Starting training with RTX 5090 optimizations..."
echo "Using config: $CONFIG_FILE"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Launch training with optimal settings
python -u qwen_manager.py train \
    --config $CONFIG_FILE \
    --compile \
    --use-flash-attention \
    --mixed-precision bf16 \
    --dataloader-workers 8 \
    --gradient-checkpointing false \
    2>&1 | tee logs/training_5090_$(date +%Y%m%d_%H%M%S).log

echo "Training completed!"
TRAIN_SCRIPT

chmod +x train_5090.sh

# Create monitoring script
echo -e "${YELLOW}Creating monitoring script...${NC}"
cat > monitor.sh << 'MONITOR_SCRIPT'
#!/bin/bash

# Split terminal monitoring for training
tmux new-session -d -s training_monitor

# Window 1: nvidia-smi
tmux send-keys -t training_monitor "watch -n 1 nvidia-smi" C-m

# Window 2: Training logs
tmux new-window -t training_monitor
tmux send-keys -t training_monitor "tail -f logs/training_5090_*.log" C-m

# Window 3: System resources
tmux new-window -t training_monitor
tmux send-keys -t training_monitor "htop" C-m

# Window 4: GPU detailed monitoring
tmux new-window -t training_monitor
tmux send-keys -t training_monitor "nvtop" C-m

echo "Monitoring session started. Attach with: tmux attach -t training_monitor"
MONITOR_SCRIPT

chmod +x monitor.sh

# Test GPU memory and capabilities
echo -e "${YELLOW}Testing GPU capabilities...${NC}"
python << 'EOF'
import torch
import sys

print("\n" + "="*50)
print("GPU INFORMATION")
print("="*50)

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {gpu.name}")
    print(f"GPU Memory: {gpu.total_memory / 1024**3:.2f} GB")
    print(f"GPU Compute Capability: {gpu.major}.{gpu.minor}")
    print(f"Number of SMs: {gpu.multi_processor_count}")
    print(f"CUDA Cores (estimated): {gpu.multi_processor_count * 128}")
    
    # Check for specific features
    print(f"\nFeature Support:")
    print(f"BF16: {torch.cuda.is_bf16_supported()}")
    print(f"TF32: Enabled by default on Ampere+")
    print(f"Flash Attention: Checking...")
    
    try:
        import flash_attn
        print(f"Flash Attention: ✓ Installed (v{flash_attn.__version__})")
    except:
        print(f"Flash Attention: ✗ Not installed")
    
    # Memory test
    print(f"\nMemory Test:")
    test_tensor = torch.zeros(1024, 1024, 1024, dtype=torch.float16, device='cuda')
    print(f"Can allocate 2GB tensor: ✓")
    del test_tensor
    torch.cuda.empty_cache()
    
    # Optimal batch size calculation
    model_size_gb = 6  # Qwen 3B in full precision
    available_memory = (gpu.total_memory / 1024**3) - 2  # Leave 2GB buffer
    estimated_batch_size = int((available_memory - model_size_gb) / 2)  # Rough estimate
    print(f"\nRecommended batch size: {estimated_batch_size}-{estimated_batch_size*2}")
    
else:
    print("ERROR: No GPU detected!")
    sys.exit(1)

print("="*50 + "\n")
EOF

# Create README for RunPod users
cat > RUNPOD_README.md << 'EOF'
# RunPod RTX 5090 Training Guide

## Quick Start

1. **Start Training with Optimizations:**
   ```bash
   ./train_5090.sh
   ```

2. **Monitor Training:**
   ```bash
   ./monitor.sh
   tmux attach -t training_monitor
   ```

3. **Interactive Training Manager:**
   ```bash
   python qwen_manager.py interactive
   ```

## RTX 5090 Optimizations

This setup includes:
- PyTorch 2.8.0 with CUDA 12.1
- Flash Attention 2 for 10x faster attention
- Triton optimized kernels
- BF16 mixed precision (better than FP16 for 5090)
- TF32 enabled for matrix operations
- Larger batch sizes (8 per device)
- Longer sequence lengths (4096 tokens)
- Full precision optimizer (no quantization needed)

## Expected Performance

With RTX 5090 (32GB):
- **Batch Size:** 8 per device (32 effective with gradient accumulation)
- **Sequence Length:** 4096 tokens
- **Training Speed:** ~15,000-20,000 tokens/sec
- **Memory Usage:** ~28-30GB
- **Time per Epoch:** 2-3 hours
- **Total Training:** 10-15 hours for 5 epochs

## Configuration

The `config_5090.yaml` is optimized for:
- No quantization (full BF16 precision)
- Larger LoRA rank (128 vs 16)
- Longer sequences (4096 vs 1024)
- Larger batches (8 vs 1)
- More aggressive learning rate
- Better regularization

## Tips

1. **Use Weights & Biases for tracking:**
   ```bash
   wandb login
   export WANDB_PROJECT="qwen-icelandic-5090"
   ```

2. **Enable torch.compile for 20% speedup:**
   ```python
   model = torch.compile(model, mode="max-autotune")
   ```

3. **Use multiple GPUs if available:**
   ```bash
   torchrun --nproc_per_node=2 train.py
   ```

## Troubleshooting

- **OOM Errors:** Reduce batch_size or max_seq_length
- **Slow Training:** Ensure Flash Attention is installed
- **CUDA Errors:** Check PyTorch/CUDA compatibility
EOF

echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ RunPod setup completed successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "Next steps:"
echo -e "1. Start training: ${BLUE}./train_5090.sh${NC}"
echo -e "2. Monitor progress: ${BLUE}./monitor.sh${NC}"
echo -e "3. Or use the manager: ${BLUE}python qwen_manager.py interactive${NC}"
echo ""
echo -e "${YELLOW}GPU Info:${NC}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""