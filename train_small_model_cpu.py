#!/usr/bin/env python3
"""
Train a much smaller model (125M-350M params) on CPU
This is feasible with 32GB RAM but will still be slow
"""

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

def train_small_model():
    print("Training small model on CPU (this will be slow but feasible)")
    
    # Use a MUCH smaller model
    model_name = "microsoft/phi-1_5"  # 1.3B params - borderline for 32GB
    # Alternative: "EleutherAI/pythia-160m"  # 160M params - definitely fits
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU needs float32
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load your Icelandic dataset
    print("Loading dataset...")
    dataset = load_dataset('json', data_files={
        'train': 'data/icelandic/train.jsonl',
        'validation': 'data/icelandic/validation.jsonl'
    })
    
    # Tokenize function
    def tokenize_function(examples):
        # Extract text from messages
        texts = []
        for msg_list in examples['messages']:
            text = " ".join([m['content'] for m in msg_list])
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=512  # Short context for CPU
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir="./small-icelandic-model",
        
        # Very small batch size for CPU
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Simulate larger batch
        
        # Training settings
        num_train_epochs=1,  # Just 1 epoch due to speed
        learning_rate=5e-5,
        warmup_steps=100,
        
        # CPU optimizations
        fp16=False,  # CPU doesn't support fp16 well
        dataloader_num_workers=4,  # Use multiple CPU cores
        
        # Logging
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        
        # Memory saving
        gradient_checkpointing=True,
        
        # Disable things that need GPU
        ddp_backend=None,
        
        # Save disk space
        save_total_limit=2,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
    )
    
    print("\n" + "="*60)
    print("⚠️ WARNING: CPU training will be VERY slow!")
    print("Estimated time: 24-48+ hours for 1 epoch")
    print("Consider using Google Colab (free GPU) instead")
    print("="*60 + "\n")
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model("./small-icelandic-model-final")
    print("Training complete!")

if __name__ == "__main__":
    train_small_model()