#!/usr/bin/env python3
"""
Qwen3-4B Icelandic Fine-tuning Script
Optimized for RTX 3080 (10GB VRAM) using QLoRA and Unsloth
"""

import os
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict

from datasets import load_from_disk, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

warnings.filterwarnings("ignore")


class QwenIcelandicTrainer:
    """Fine-tune Qwen3-4B for Icelandic using memory-efficient techniques"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        output_dir: str = "./qwen-icelandic-4b",
        max_seq_length: int = 2048,
        use_4bit: bool = True
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length = max_seq_length
        self.use_4bit = use_4bit
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {gpu_memory:.2f} GB")
            
            if gpu_memory < 10:
                print("WARNING: Less than 10GB VRAM detected. Using aggressive memory optimization.")
                self.max_seq_length = min(1024, max_seq_length)
        
    def load_model_and_tokenizer(self):
        """Load Qwen3-4B with QLoRA configuration optimized for RTX 3080"""
        print(f"Loading model: {self.model_name}")
        
        # Load model with 4-bit quantization for memory efficiency
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect (will use float16 for RTX 3080)
            load_in_4bit=self.use_4bit,  # Essential for 10GB VRAM
        )
        
        # Configure LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=32,  # LoRA rank - good balance for 4B model
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=32,
            lora_dropout=0,  # Optimized for Unsloth
            bias="none",
            use_gradient_checkpointing="unsloth",  # 30% memory reduction
            random_state=42,
            use_rslora=False,  # Use standard LoRA
            loftq_config=None,
        )
        
        # Set up chat template for Qwen
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen-3",  # Qwen3 chat template
            mapping={"role": "role", "content": "content"}
        )
        
        print(f"Model loaded with LoRA adapters")
        print(f"Trainable parameters: {self.count_parameters()}")
        
    def count_parameters(self) -> str:
        """Count trainable and total parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        percentage = 100 * trainable / total
        return f"{trainable:,} / {total:,} ({percentage:.2f}%)"
    
    def load_dataset(self, dataset_path: Optional[str] = None):
        """Load the prepared Icelandic dataset"""
        if dataset_path and Path(dataset_path).exists():
            print(f"Loading dataset from {dataset_path}")
            self.dataset = load_from_disk(dataset_path)
        else:
            # Try to load from default location
            default_path = Path("./data/icelandic/icelandic_training_data")
            if default_path.exists():
                print(f"Loading dataset from {default_path}")
                self.dataset = load_from_disk(str(default_path))
            else:
                print("No local dataset found. Loading a sample dataset...")
                # Load a small sample for testing
                self.dataset = self.create_sample_dataset()
        
        print(f"Dataset loaded:")
        print(f"  Training examples: {len(self.dataset['train'])}")
        print(f"  Validation examples: {len(self.dataset['test'])}")
    
    def create_sample_dataset(self):
        """Create a small sample dataset for testing"""
        from datasets import Dataset, DatasetDict
        
        sample_data = [
            {
                "messages": [
                    {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
                    {"role": "user", "content": "Hvað er höfuðborg Íslands?"},
                    {"role": "assistant", "content": "Höfuðborg Íslands er Reykjavík. Hún er stærsta borg landsins og þar búa um 140.000 manns."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
                    {"role": "user", "content": "Segðu mér frá veðrinu á Íslandi."},
                    {"role": "assistant", "content": "Veðrið á Íslandi er mjög breytilegt. Það getur breyst hratt og oft er sagt að hægt sé að upplifa allar fjórar árstíðirnar á einum degi."}
                ]
            }
        ] * 50  # Repeat for testing
        
        train_data = Dataset.from_list(sample_data[:80])
        test_data = Dataset.from_list(sample_data[80:])
        
        return DatasetDict({"train": train_data, "test": test_data})
    
    def get_training_arguments(self) -> SFTConfig:
        """Get optimized training arguments for RTX 3080"""
        return SFTConfig(
            output_dir=str(self.output_dir),
            
            # Training hyperparameters
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Keep minimal for 10GB VRAM
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Effective batch size of 8
            
            # Learning rate settings
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            
            # Optimizer settings
            optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
            
            # Memory optimizations
            fp16=not is_bfloat16_supported(),  # Use FP16 on RTX 3080
            bf16=is_bfloat16_supported(),
            gradient_checkpointing=True,  # Essential for memory
            max_grad_norm=0.3,
            
            # Logging and saving
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Other settings
            seed=42,
            report_to="none",  # Disable wandb/tensorboard for simplicity
            remove_unused_columns=False,
            dataset_text_field="",  # We'll handle formatting
            max_seq_length=self.max_seq_length,
            packing=False,  # Don't pack sequences for clarity
        )
    
    def formatting_func(self, examples):
        """Format examples for training"""
        texts = []
        for messages in examples["messages"]:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts
    
    def train(self):
        """Execute the training loop"""
        print("\nStarting training...")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Output directory: {self.output_dir}")
        
        # Get training arguments
        training_args = self.get_training_arguments()
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            formatting_func=self.formatting_func,
        )
        
        # Start training
        trainer.train()
        
        print("\nTraining completed!")
        
        # Save the model
        self.save_model()
    
    def save_model(self):
        """Save the trained model and tokenizer"""
        print(f"\nSaving model to {self.output_dir}")
        
        # Save model in different formats
        self.model.save_pretrained(str(self.output_dir / "lora"))  # LoRA adapters only
        self.tokenizer.save_pretrained(str(self.output_dir / "lora"))
        
        # Optionally merge and save full model (requires more memory)
        try:
            print("Attempting to merge and save full model...")
            self.model.save_pretrained_merged(
                str(self.output_dir / "merged"),
                self.tokenizer,
                save_method="merged_16bit"  # Save in 16-bit to save space
            )
            print("Full model saved successfully!")
        except Exception as e:
            print(f"Could not save merged model (memory constraints): {e}")
            print("LoRA adapters saved successfully. You can merge them later.")
        
        # Save as GGUF for llama.cpp (optional)
        try:
            print("Attempting to save GGUF format...")
            self.model.save_pretrained_gguf(
                str(self.output_dir / "gguf"),
                self.tokenizer,
                quantization_method="q4_k_m"  # 4-bit quantization
            )
            print("GGUF model saved successfully!")
        except Exception as e:
            print(f"Could not save GGUF format: {e}")
    
    def run_inference_test(self):
        """Test the trained model with Icelandic prompts"""
        print("\nTesting model with Icelandic prompts...")
        
        FastLanguageModel.for_inference(self.model)  # Enable inference mode
        
        test_prompts = [
            "Hvað er Ísland?",
            "Segðu mér sögu um tröll.",
            "Hvernig er veðrið á Íslandi?",
        ]
        
        for prompt in test_prompts:
            messages = [
                {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
                {"role": "user", "content": prompt}
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                use_cache=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Qwen3-4B on Icelandic data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model name or path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to prepared dataset")
    parser.add_argument("--output", type=str, default="./qwen-icelandic-4b",
                        help="Output directory for trained model")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run inference test with existing model")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = QwenIcelandicTrainer(
        model_name=args.model,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        use_4bit=True  # Always use 4-bit for RTX 3080
    )
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    if args.test_only:
        # Just run inference test
        trainer.run_inference_test()
    else:
        # Full training pipeline
        trainer.load_dataset(args.dataset)
        trainer.train()
        trainer.run_inference_test()
    
    print("\nPipeline completed successfully!")
    print(f"Model saved to: {args.output}")
    print("\nNext steps:")
    print("1. Test the model: python train_qwen_icelandic.py --test-only")
    print("2. Use the model: Load from the output directory")
    print("3. Convert to GGUF for llama.cpp if needed")


if __name__ == "__main__":
    main()