# Qwen3-4B Icelandic Fine-tuning

This project fine-tunes the Qwen3-4B-Thinking model for Icelandic language generation, optimized for RTX 3080 (10GB VRAM) using QLoRA and Unsloth optimizations.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (creates venv, installs dependencies)
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Prepare Dataset

```bash
# Download and prepare Icelandic datasets
python prepare_icelandic_dataset.py
```

This will download and combine multiple Icelandic datasets:
- Icelandic Gigaword Corpus (IGC)
- Icelandic Wikipedia
- OSCAR Icelandic subset
- GreynirCorpus
- Icelandic Common Crawl (IC3)

### 3. Start Training

```bash
# Full training with default settings
python train_qwen_icelandic.py

# Or with custom settings
python train_qwen_icelandic.py \
    --max-seq-length 1024 \
    --output ./my-model
```

## 📊 Dataset Sources

| Dataset | Size | Quality | License |
|---------|------|---------|---------|
| IGC-2024 | ~2.4B words | High | Mixed (50% CC BY 4.0) |
| Wikipedia | ~56MB | High | CC BY-SA |
| OSCAR | Variable | Medium | Various |
| GreynirCorpus | ~140M words | High | CC BY 4.0 |
| IC3 | Variable | Medium-High | Various |

## 🎯 Training Optimizations for RTX 3080

### Memory Optimizations
- **QLoRA**: 4-bit quantization reduces memory by ~75%
- **Gradient Checkpointing**: 30% memory reduction via Unsloth
- **8-bit Optimizer**: AdamW in 8-bit precision
- **FP16 Training**: Mixed precision for faster training
- **Batch Size**: 1 with gradient accumulation (effective batch=8)

### Expected Performance
- **Memory Usage**: ~8-9GB out of 10GB
- **Training Speed**: ~2-5x faster than standard fine-tuning
- **Training Time**: ~4-8 hours for 100k examples
- **Model Quality**: 99%+ of full precision performance

## 🔧 Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  max_seq_length: 2048  # Reduce to 1024 if OOM
  
training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

## 💾 Output Files

After training, you'll find:

```
qwen-icelandic-4b/
├── lora/              # LoRA adapters only
├── merged/            # Full merged model (if memory allows)
└── gguf/              # GGUF format for llama.cpp
```

## 🧪 Testing the Model

```bash
# Test with existing model
python train_qwen_icelandic.py --test-only

# Or use in Python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./qwen-icelandic-4b/merged")
tokenizer = AutoTokenizer.from_pretrained("./qwen-icelandic-4b/merged")

# Generate Icelandic text
prompt = "Hvað er Ísland?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## 🚨 Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce sequence length:
   ```bash
   python train_qwen_icelandic.py --max-seq-length 1024
   ```

2. Close other GPU applications

3. Use system RAM for optimizer offloading (edit config)

### Slow Training

1. Ensure CUDA is properly installed:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Check that Unsloth is installed correctly

3. Monitor GPU utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

## 📈 Monitoring Training

```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/training.log

# Use tensorboard (if enabled)
tensorboard --logdir ./qwen-icelandic-4b
```

## 🤝 Contributing

Feel free to open issues or submit PRs to improve the training pipeline or add more Icelandic datasets.

## 📄 License

This project uses various open-source components. Please check individual dataset licenses before commercial use.

## 🙏 Acknowledgments

- Qwen team for the base model
- Unsloth for memory optimizations
- Icelandic NLP community for datasets
- HuggingFace for infrastructure