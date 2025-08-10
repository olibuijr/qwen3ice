#!/bin/bash

# Qwen Icelandic Manager Runner
# Convenient wrapper for the unified manager system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Qwen Icelandic Training Manager    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to check and install package
check_and_install() {
    local package=$1
    local import_name=${2:-$1}
    python -c "import ${import_name}" 2>/dev/null || {
        echo -e "${YELLOW}Installing ${package}...${NC}"
        pip install "${package}" || echo -e "${YELLOW}${package} installation failed (optional)${NC}"
    }
}

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
MIN_VERSION="3.9"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo -e "${RED}Error: Python $MIN_VERSION or higher required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch first (with CUDA if available)
    echo -e "${YELLOW}Installing PyTorch...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        # Install PyTorch with CUDA support
        pip install torch>=2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        # CPU-only version
        pip install torch>=2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install core ML dependencies
    echo -e "${YELLOW}Installing core ML libraries...${NC}"
    pip install transformers>=4.51.0 datasets>=2.14.0 accelerate>=0.24.0
    pip install peft>=0.7.0 trl>=0.7.4 bitsandbytes>=0.41.0
    
    # Install tokenizer dependencies
    echo -e "${YELLOW}Installing tokenizer libraries...${NC}"
    pip install tokenizers>=0.15.0 sentencepiece>=0.1.99 protobuf>=3.20.0 || {
        echo -e "${YELLOW}Some tokenizer libraries failed, trying alternatives...${NC}"
        pip install tokenizers>=0.15.0 protobuf>=3.20.0
    }
    
    # Install CLI and UI dependencies
    echo -e "${YELLOW}Installing CLI/UI libraries...${NC}"
    pip install click>=8.1.0 rich>=14.0.0 plotext>=5.2.0 textual>=0.40.0
    
    # Install monitoring tools
    echo -e "${YELLOW}Installing monitoring tools...${NC}"
    pip install psutil>=5.9.0 GPUtil>=1.4.0 nvidia-ml-py>=12.0.0 || {
        echo -e "${YELLOW}GPU monitoring tools optional, continuing...${NC}"
        pip install psutil>=5.9.0
    }
    
    # Install data processing libraries
    echo -e "${YELLOW}Installing data processing libraries...${NC}"
    pip install pandas>=2.0.0 numpy>=1.24.0 pyyaml>=6.0 tqdm>=4.65.0
    
    # Install additional ML tools
    echo -e "${YELLOW}Installing additional tools...${NC}"
    pip install huggingface_hub>=0.19.0 safetensors>=0.4.0
    
    # Try to install Unsloth (optional but recommended)
    echo -e "${YELLOW}Attempting to install Unsloth optimizations...${NC}"
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || {
        echo -e "${YELLOW}Unsloth installation failed (optional optimization)${NC}"
    }
    
    echo -e "${GREEN}✓ Virtual environment created and dependencies installed${NC}"
else
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    
    # Check and install missing critical dependencies
    echo -e "${YELLOW}Checking for missing dependencies...${NC}"
    
    # Core ML libraries
    check_and_install "torch>=2.8.0" "torch"
    check_and_install "transformers>=4.51.0" "transformers"
    check_and_install "datasets>=2.14.0" "datasets"
    check_and_install "accelerate>=0.24.0" "accelerate"
    check_and_install "peft>=0.7.0" "peft"
    check_and_install "trl>=0.7.4" "trl"
    
    # CLI/UI libraries
    check_and_install "click>=8.1.0" "click"
    check_and_install "rich>=14.0.0" "rich"
    check_and_install "plotext>=5.2.0" "plotext"
    check_and_install "textual>=0.40.0" "textual"
    
    # Monitoring
    check_and_install "psutil>=5.9.0" "psutil"
    check_and_install "GPUtil>=1.4.0" "GPUtil"
    
    # Data processing
    check_and_install "pandas>=2.0.0" "pandas"
    check_and_install "numpy>=1.24.0" "numpy"
    check_and_install "pyyaml>=6.0" "yaml"
    check_and_install "tqdm>=4.65.0" "tqdm"
fi

# Verify critical imports
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "
import sys
errors = []
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA: {torch.version.cuda}')
except ImportError:
    errors.append('torch')
    
try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except ImportError:
    errors.append('transformers')
    
try:
    import datasets
    print('✓ Datasets')
except ImportError:
    errors.append('datasets')
    
try:
    import click
    print('✓ Click')
except ImportError:
    errors.append('click')
    
try:
    import rich
    print('✓ Rich')
except ImportError:
    errors.append('rich')

if errors:
    print(f'\\n❌ Missing critical packages: {errors}')
    print('Run: pip install ' + ' '.join(errors))
    sys.exit(1)
else:
    print('\\n✅ All critical dependencies installed!')
" || {
    echo -e "${RED}Some critical dependencies are missing!${NC}"
    echo -e "${YELLOW}Try running: pip install -r requirements_essential.txt${NC}"
    exit 1
}

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ No GPU detected - will use CPU${NC}"
fi

# Main menu if no arguments
if [ $# -eq 0 ]; then
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "1) Run complete pipeline (recommended for first time)"
    echo "2) Prepare datasets only"
    echo "3) Train model"
    echo "4) Test trained model (interactive chat)"
    echo "5) Monitor system resources"
    echo "6) Launch interactive UI"
    echo "7) View project info"
    echo "8) Exit"
    echo ""
    read -p "Enter choice [1-8]: " choice
    
    case $choice in
        1)
            python qwen_manager.py full-pipeline
            ;;
        2)
            python qwen_manager.py prepare-data
            ;;
        3)
            python qwen_manager.py train
            ;;
        4)
            python qwen_manager.py inference ./models/qwen-icelandic
            ;;
        5)
            python qwen_manager.py monitor
            ;;
        6)
            python qwen_manager.py interactive
            ;;
        7)
            python qwen_manager.py info
            ;;
        8)
            echo -e "${YELLOW}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
else
    # Pass arguments directly to manager
    python qwen_manager.py "$@"
fi