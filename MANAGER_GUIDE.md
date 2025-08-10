# 🚀 Qwen Icelandic Manager - Complete Guide

## Overview

The Qwen Manager (`qwen_manager.py`) is a comprehensive, unified system that handles all aspects of training Qwen models for Icelandic. It replaces all individual scripts with a single, powerful entry point.

## Features

### 🎯 Core Capabilities
- **Dataset Management**: Download, prepare, and manage Icelandic datasets
- **Model Training**: Full training pipeline with QLoRA and memory optimizations
- **Inference Engine**: Interactive chat and text generation
- **System Monitoring**: Real-time GPU, CPU, and memory tracking
- **Configuration Management**: YAML-based configuration system
- **Log Analysis**: Training metrics extraction and visualization
- **Interactive UI**: Rich terminal interface for monitoring

### 🛠️ Integrated Components
- Dataset preparation (Wikipedia, IC3, Wiki QA)
- Model training with Unsloth optimizations
- Evaluation and metrics tracking
- Model export and conversion
- Interactive inference and testing

## Installation

```bash
# Install all dependencies
pip install -r requirements_full.txt

# Install Unsloth (optional but recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Make executable
chmod +x qwen_manager.py
```

## Quick Start

### 1. Complete Pipeline (Recommended)
```bash
# Run the complete training pipeline
python qwen_manager.py full-pipeline

# This will:
# 1. Setup environment and check dependencies
# 2. Download and prepare datasets
# 3. Train the model
# 4. Evaluate performance
# 5. Export the model
# 6. Offer to test with interactive chat
```

### 2. Step-by-Step Training
```bash
# Step 1: Setup environment
python qwen_manager.py setup

# Step 2: Prepare datasets
python qwen_manager.py prepare-data

# Step 3: Train model
python qwen_manager.py train

# Step 4: Test the model
python qwen_manager.py inference ./models/qwen-icelandic
```

## Command Reference

### Core Commands

#### `setup`
Check dependencies and prepare environment:
```bash
python qwen_manager.py setup
```

#### `prepare-data`
Download and prepare Icelandic datasets:
```bash
python qwen_manager.py prepare-data --sources wikipedia ic3 wikiqa
```

#### `train`
Train the model:
```bash
python qwen_manager.py train

# Resume from checkpoint
python qwen_manager.py train --resume checkpoint-500
```

#### `inference`
Interactive chat with trained model:
```bash
python qwen_manager.py inference ./models/qwen-icelandic
```

#### `generate`
Generate text from a prompt:
```bash
python qwen_manager.py generate ./models/qwen-icelandic "Hvað er Ísland?"
```

### Monitoring Commands

#### `monitor`
Real-time system resource monitoring:
```bash
python qwen_manager.py monitor
```

#### `analyze-logs`
Analyze training logs:
```bash
python qwen_manager.py analyze-logs logs/training_20250810.log
```

### Configuration Commands

#### `config`
Edit configuration interactively:
```bash
python qwen_manager.py config
```

#### `info`
Display project information:
```bash
python qwen_manager.py info
```

### Advanced Commands

#### `interactive`
Launch the full interactive UI (requires terminal with TUI support):
```bash
python qwen_manager.py interactive
```

#### `full-pipeline`
Run complete pipeline automatically:
```bash
python qwen_manager.py full-pipeline
```

## Configuration

The system uses `config.yaml` for all settings:

```yaml
project_name: qwen-icelandic
model_name: Qwen/Qwen2.5-3B-Instruct

# Dataset settings
dataset_dir: ./data/icelandic
use_all_data: true

# Training settings
output_dir: ./models/qwen-icelandic
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
max_seq_length: 2048

# Optimization settings
use_4bit: true
use_unsloth: true
fp16: true
gradient_checkpointing: true

# LoRA settings
lora_r: 32
lora_alpha: 32
lora_dropout: 0.0
```

## Usage Examples

### Example 1: Quick Training
```bash
# Prepare data and train in one command
python qwen_manager.py train
```

### Example 2: Custom Dataset Sources
```bash
# Use only Wikipedia data
python qwen_manager.py prepare-data --sources wikipedia

# Use all available sources
python qwen_manager.py prepare-data --sources wikipedia ic3 wikiqa
```

### Example 3: Monitor Training
```bash
# Terminal 1: Start training
python qwen_manager.py train

# Terminal 2: Monitor system
python qwen_manager.py monitor

# Terminal 3: Watch logs
tail -f logs/training_*.log
```

### Example 4: Test Model
```bash
# Interactive chat
python qwen_manager.py inference ./models/qwen-icelandic

# Single generation
python qwen_manager.py generate ./models/qwen-icelandic \
    "Segðu mér sögu um tröll" --max-length 500
```

## Interactive UI

Launch the full-featured terminal UI:

```bash
python qwen_manager.py interactive
```

### UI Features:
- **Real-time Metrics**: Loss, learning rate, throughput
- **System Monitoring**: GPU, CPU, memory usage
- **Log Viewer**: Filtered, searchable training logs
- **Configuration Editor**: Edit settings without leaving UI
- **Dataset Browser**: Preview training examples
- **Training Control**: Start, stop, pause, resume

### Keyboard Shortcuts:
- `s` - Start Training
- `x` - Stop Training
- `p` - Pause Training
- `r` - Resume Training
- `l` - Show Logs
- `m` - Show Metrics
- `c` - Show Config
- `d` - Show Dataset
- `h` - Help
- `q` - Quit

## Workflow Examples

### Workflow 1: First Time Setup
```bash
# 1. Check environment
python qwen_manager.py setup

# 2. Review configuration
python qwen_manager.py config

# 3. Run full pipeline
python qwen_manager.py full-pipeline
```

### Workflow 2: Experiment with Settings
```bash
# 1. Edit config
python qwen_manager.py config

# 2. Prepare smaller dataset
python qwen_manager.py prepare-data --sources wikipedia

# 3. Quick training test
python qwen_manager.py train

# 4. Evaluate results
python qwen_manager.py inference ./models/qwen-icelandic
```

### Workflow 3: Production Training
```bash
# 1. Prepare full dataset
python qwen_manager.py prepare-data

# 2. Launch interactive UI for monitoring
python qwen_manager.py interactive

# 3. Start training from UI
# 4. Monitor progress in real-time
# 5. Save checkpoints periodically
```

## Tips and Best Practices

### Memory Management
- Use `use_4bit: true` for RTX 3080 (10GB)
- Enable gradient checkpointing for larger models
- Reduce `max_seq_length` if OOM errors occur

### Dataset Optimization
- Start with smaller datasets for testing
- Use `--sources wikipedia` for quick tests
- Full dataset training takes 4-8 hours on RTX 3080

### Monitoring
- Always run `monitor` in separate terminal
- Check GPU temperature to avoid throttling
- Save logs for later analysis

### Checkpointing
- Checkpoints saved every 100 steps by default
- Use `--resume` to continue from checkpoint
- Keep best 3 checkpoints to save space

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in config
per_device_train_batch_size: 1

# Reduce sequence length
max_seq_length: 1024

# Enable more aggressive optimization
use_4bit: true
gradient_checkpointing: true
```

### Slow Training
```bash
# Check GPU utilization
python qwen_manager.py monitor

# Ensure using GPU
nvidia-smi

# Enable Unsloth
use_unsloth: true
```

### Dataset Issues
```bash
# Clear cache and re-download
rm -rf data/icelandic/*
python qwen_manager.py prepare-data
```

## Advanced Features

### Custom Training Loop
```python
from qwen_manager import QwenManager, TrainingStage

manager = QwenManager()
manager.initialize("config.yaml")

# Run specific stages
manager.run_pipeline([
    TrainingStage.SETUP,
    TrainingStage.DATASET,
    TrainingStage.TRAINING
])
```

### Programmatic Usage
```python
from qwen_manager import InferenceEngine

# Load and use model
engine = InferenceEngine("./models/qwen-icelandic")
engine.load_model()
response = engine.generate("Hvað er Ísland?")
print(response)
```

## Support

For issues or questions:
- Check logs in `./logs/` directory
- Run `python qwen_manager.py info` for system details
- Use `python qwen_manager.py setup` to verify dependencies

## Next Steps

1. **Train Your Model**: Run `python qwen_manager.py full-pipeline`
2. **Test Results**: Use interactive chat to test quality
3. **Fine-tune**: Adjust config and retrain as needed
4. **Deploy**: Export model for production use

---

*Built for efficient Icelandic language model training on consumer hardware*