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

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements_full.txt
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

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