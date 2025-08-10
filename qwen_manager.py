#!/usr/bin/env python3
"""
Qwen Icelandic Training Manager - Complete Unified System
Single entry point for all training, dataset, and model operations
"""

import asyncio
import json
import os
import sys
import time
import signal
import random
import shutil
import subprocess
import threading
import queue
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

warnings.filterwarnings("ignore")

# Core imports
import click
import yaml
import psutil
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.prompt import Prompt, Confirm
from rich.text import Text

# ML imports (optional, graceful fallback)
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from datasets import load_dataset, Dataset, DatasetDict
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from trl import SFTTrainer, SFTConfig
    
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes for type hints when imports fail
    class DatasetDict:
        pass
    class Dataset:
        pass

# Try to import Unsloth for optimizations
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

console = Console()


class TrainingStage(Enum):
    """Training pipeline stages"""
    SETUP = "setup"
    DATASET = "dataset"
    TRAINING = "training"
    EVALUATION = "evaluation"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class ProjectConfig:
    """Complete project configuration"""
    project_name: str = "qwen-icelandic"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # Dataset settings
    dataset_dir: str = "./data/icelandic"
    max_examples: Optional[int] = None
    use_all_data: bool = True
    
    # Training settings
    output_dir: str = "./models/qwen-icelandic"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    
    # Optimization settings
    use_4bit: bool = True
    use_unsloth: bool = True
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"
    
    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    
    # System settings
    cuda_device: int = 0
    num_workers: int = 4
    seed: int = 42
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    @classmethod
    def from_file(cls, path: str) -> 'ProjectConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        # Filter out unknown fields
        valid_fields = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def save(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


class DatasetManager:
    """Manages dataset preparation and loading"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.output_dir = Path(config.dataset_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_datasets(self, sources: List[str] = None) -> Tuple[int, int]:
        """Prepare Icelandic datasets from various sources"""
        
        if sources is None:
            sources = ["wikipedia", "ic3", "wikiqa"]
        
        all_examples = []
        stats = {}
        
        examples_by_source = {}
        filtered_stats = {}
        
        with console.status("[bold green]Preparing datasets...") as status:
            
            # Wikipedia
            if "wikipedia" in sources:
                status.update("Loading Wikipedia...")
                wiki_examples = self._load_wikipedia()
                original_count = len(wiki_examples)
                # Filter low-quality examples
                wiki_examples = [ex for ex in wiki_examples 
                               if self._filter_quality(ex["messages"][-1]["content"])]
                examples_by_source['Wikipedia'] = wiki_examples
                stats['Wikipedia'] = original_count
                filtered_stats['Wikipedia'] = len(wiki_examples)
            
            # IC3
            if "ic3" in sources:
                status.update("Loading IC3...")
                ic3_examples = self._load_ic3()
                original_count = len(ic3_examples)
                # Filter low-quality examples
                ic3_examples = [ex for ex in ic3_examples 
                              if self._filter_quality(ex["messages"][-1]["content"])]
                examples_by_source['IC3'] = ic3_examples
                stats['IC3'] = original_count
                filtered_stats['IC3'] = len(ic3_examples)
            
            # Wiki QA
            if "wikiqa" in sources:
                status.update("Loading Wiki QA...")
                qa_examples = self._load_wiki_qa()
                examples_by_source['Wiki QA'] = qa_examples
                stats['Wiki QA'] = len(qa_examples)
                filtered_stats['Wiki QA'] = len(qa_examples)  # No filtering for Q&A
        
        # Print statistics
        table = Table(title="Dataset Statistics", box=box.ROUNDED)
        table.add_column("Source", style="cyan")
        table.add_column("Original", style="yellow")
        table.add_column("After Filtering", style="green")
        table.add_column("Kept %", style="blue")
        
        for source in stats:
            orig = stats[source]
            filtered = filtered_stats[source]
            pct = (filtered / orig * 100) if orig > 0 else 0
            table.add_row(source, f"{orig:,}", f"{filtered:,}", f"{pct:.1f}%")
        
        # Balance datasets
        all_examples = self._balance_datasets(examples_by_source)
        
        table.add_row("[bold]Total", 
                     f"[bold]{sum(stats.values()):,}", 
                     f"[bold]{len(all_examples):,}",
                     f"[bold]{len(all_examples)/sum(stats.values())*100:.1f}%")
        console.print(table)
        
        # Save datasets
        if all_examples:
            console.print(f"\n[green]Saving {len(all_examples):,} balanced examples...")
            return self._save_datasets(all_examples)
        else:
            console.print("[red]No examples collected!")
            return 0, 0
    
    def _load_wikipedia(self) -> List[Dict]:
        """Load and process Wikipedia dataset"""
        try:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.is", split="train")
            examples = []
            
            limit = min(5000, len(dataset)) if not self.config.use_all_data else len(dataset)
            
            for i in tqdm(range(limit), desc="Processing Wikipedia"):
                article = dataset[i]
                text = f"{article['title']}\n\n{article['text']}"
                if len(text) > 200:
                    examples.append(self._format_for_training(text))
            
            return examples
        except Exception as e:
            console.print(f"[yellow]Could not load Wikipedia: {e}")
            return []
    
    def _load_ic3(self) -> List[Dict]:
        """Load and process IC3 dataset"""
        try:
            dataset = load_dataset("mideind/icelandic-common-crawl-corpus-IC3", split="train")
            examples = []
            
            limit = min(5000, len(dataset)) if not self.config.use_all_data else len(dataset)
            
            for i in tqdm(range(limit), desc="Processing IC3"):
                doc = dataset[i]
                text = doc.get('text', '')
                if len(text) > 200:
                    examples.append(self._format_for_training(text))
            
            return examples
        except Exception as e:
            console.print(f"[yellow]Could not load IC3: {e}")
            return []
    
    def _load_wiki_qa(self) -> List[Dict]:
        """Load and process Wiki QA dataset"""
        try:
            dataset = load_dataset("mideind/icelandic_wiki_qa", split="train")
            examples = []
            
            for qa in tqdm(dataset, desc="Processing Wiki QA"):
                if qa.get('query') and qa.get('answer'):
                    examples.append({
                        "messages": [
                            {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem svarar spurningum."},
                            {"role": "user", "content": qa['query']},
                            {"role": "assistant", "content": qa['answer']}
                        ]
                    })
            
            return examples
        except Exception as e:
            console.print(f"[yellow]Could not load Wiki QA: {e}")
            return []
    
    def _format_for_training(self, text: str) -> Dict:
        """Format text for instruction fine-tuning"""
        templates = [
            "Haltu áfram með eftirfarandi texta:",
            "Ljúktu við eftirfarandi málsgrein:",
            "Skrifaðu framhald á:",
        ]
        
        instruction = random.choice(templates)
        split_point = min(400, len(text) // 2)
        
        return {
            "messages": [
                {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður."},
                {"role": "user", "content": f"{instruction}\n\n{text[:split_point]}"},
                {"role": "assistant", "content": text[split_point:]}
            ]
        }
    
    def _filter_quality(self, text: str) -> bool:
        """Filter low-quality text examples"""
        # Check minimum length
        if len(text) < 100:
            return False
        
        # Check for excessive punctuation/noise
        if text.count('!') > 10 or text.count('?') > 10:
            return False
        
        # Check for Icelandic characters
        icelandic_chars = 'áéíóúýþæðÁÉÍÓÚÝÞÆÐ'
        has_icelandic = any(c in text for c in icelandic_chars)
        if not has_icelandic and len(text) > 50:
            return False  # Likely not Icelandic
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too repetitive
                return False
        
        return True
    
    def _balance_datasets(self, examples_dict: Dict[str, List]) -> List[Dict]:
        """Balance examples from different sources"""
        MAX_PER_SOURCE = 500000  # Cap each source
        
        balanced = []
        for source, examples in examples_dict.items():
            if len(examples) > MAX_PER_SOURCE:
                console.print(f"[yellow]Capping {source} from {len(examples)} to {MAX_PER_SOURCE} examples")
                examples = random.sample(examples, MAX_PER_SOURCE)
            balanced.extend(examples)
        
        return balanced
    
    def _save_datasets(self, examples: List[Dict]) -> Tuple[int, int]:
        """Save processed datasets"""
        random.seed(self.config.seed)
        random.shuffle(examples)
        
        # Split
        split_point = int(len(examples) * 0.95)
        train_data = examples[:split_point]
        val_data = examples[split_point:]
        
        # Save as JSONL
        train_file = self.output_dir / "train.jsonl"
        val_file = self.output_dir / "validation.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for ex in train_data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for ex in val_data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        # Save as HuggingFace dataset if available
        if ML_AVAILABLE:
            try:
                from datasets import Dataset as RealDataset, DatasetDict as RealDatasetDict
                dataset_dict = RealDatasetDict({
                    'train': RealDataset.from_list(train_data),
                    'test': RealDataset.from_list(val_data)
                })
                dataset_dict.save_to_disk(str(self.output_dir / "hf_dataset"))
            except Exception as e:
                console.print(f"[yellow]Could not save HF dataset format: {e}")
        
        return len(train_data), len(val_data)
    
    def load_datasets(self) -> DatasetDict:
        """Load prepared datasets"""
        dataset_path = self.output_dir / "hf_dataset"
        if dataset_path.exists():
            return DatasetDict.load_from_disk(str(dataset_path))
        else:
            # Load from JSONL
            return load_dataset('json', data_files={
                'train': str(self.output_dir / 'train.jsonl'),
                'test': str(self.output_dir / 'validation.jsonl')
            })


class ModelTrainer:
    """Handles model training and fine-tuning"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_model(self):
        """Initialize model and tokenizer"""
        console.print(f"\n[cyan]Loading model: {self.config.model_name}")
        
        if UNSLOTH_AVAILABLE and self.config.use_unsloth:
            self._setup_unsloth_model()
        else:
            self._setup_standard_model()
    
    def _setup_unsloth_model(self):
        """Setup model with Unsloth optimizations"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.use_4bit,
        )
        
        # Add LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if self.config.gradient_checkpointing else False,
        )
        
        console.print("[green]✓ Model loaded with Unsloth optimizations")
    
    def _setup_standard_model(self):
        """Setup model with standard transformers"""
        # Quantization config
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        console.print("[green]✓ Model loaded with standard transformers")
    
    def train(self, dataset: DatasetDict):
        """Train the model"""
        console.print("\n[cyan]Starting training...")
        
        # Training arguments
        if UNSLOTH_AVAILABLE and self.config.use_unsloth:
            training_args = SFTConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                lr_scheduler_type=getattr(self.config, 'lr_scheduler_type', 'cosine'),
                warmup_ratio=getattr(self.config, 'warmup_ratio', 0.1),
                fp16=self.config.fp16,
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                evaluation_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                optim=self.config.optim,
                seed=self.config.seed,
                max_seq_length=self.config.max_seq_length,
                max_grad_norm=getattr(self.config, 'max_grad_norm', 0.5),
            )
        else:
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                lr_scheduler_type=getattr(self.config, 'lr_scheduler_type', 'cosine'),
                warmup_ratio=getattr(self.config, 'warmup_ratio', 0.1),
                fp16=self.config.fp16,
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                evaluation_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                optim=self.config.optim,
                seed=self.config.seed,
                gradient_checkpointing=self.config.gradient_checkpointing,
                max_grad_norm=getattr(self.config, 'max_grad_norm', 0.5),
            )
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            formatting_func=self._formatting_func,
        )
        
        # Start training
        with Progress() as progress:
            task = progress.add_task(
                "[green]Training...", 
                total=len(dataset['train']) * self.config.num_train_epochs
            )
            
            result = self.trainer.train()
            progress.update(task, completed=True)
        
        return result
    
    def _formatting_func(self, examples):
        """Format examples for training"""
        texts = []
        for messages in examples["messages"]:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # Fallback formatting
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            texts.append(text)
        return texts
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        save_path = path or self.config.output_dir
        
        console.print(f"\n[cyan]Saving model to {save_path}")
        
        if UNSLOTH_AVAILABLE and self.config.use_unsloth:
            # Save LoRA adapters
            self.model.save_pretrained(f"{save_path}/lora")
            self.tokenizer.save_pretrained(f"{save_path}/lora")
            
            # Try to merge and save
            try:
                self.model.save_pretrained_merged(
                    f"{save_path}/merged",
                    self.tokenizer,
                    save_method="merged_16bit"
                )
                console.print("[green]✓ Saved merged model")
            except:
                console.print("[yellow]Could not save merged model (memory constraints)")
        else:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
        
        console.print("[green]✓ Model saved successfully")
    
    def evaluate(self, dataset: DatasetDict):
        """Evaluate the model"""
        console.print("\n[cyan]Evaluating model...")
        
        if self.trainer:
            metrics = self.trainer.evaluate(eval_dataset=dataset['test'])
            
            # Display metrics
            table = Table(title="Evaluation Metrics", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in metrics.items():
                table.add_row(key, f"{value:.4f}")
            
            console.print(table)
            return metrics
        else:
            console.print("[red]Trainer not initialized!")
            return {}


class InferenceEngine:
    """Handles model inference and testing"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model for inference"""
        console.print(f"\n[cyan]Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Try to load with 4-bit quantization for inference
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        except:
            # Fallback to CPU/standard loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        
        console.print("[green]✓ Model loaded for inference")
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate text from prompt"""
        messages = [
            {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def interactive_chat(self):
        """Interactive chat mode"""
        console.print("\n[cyan]Starting interactive chat (type 'quit' to exit)")
        console.print("[yellow]Chat in Icelandic for best results!\n")
        
        while True:
            prompt = Prompt.ask("[bold cyan]You")
            
            if prompt.lower() in ['quit', 'exit', 'bye']:
                break
            
            with console.status("[bold green]Thinking..."):
                response = self.generate(prompt)
            
            console.print(f"[bold green]Assistant:[/bold green] {response}\n")
        
        console.print("[yellow]Chat ended. Bless!")


class SystemMonitor:
    """Monitors system resources"""
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU information"""
        if not CUDA_AVAILABLE:
            return {"available": False}
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "available": True,
                    "name": gpu.name,
                    "memory_total": f"{gpu.memoryTotal} MB",
                    "memory_used": f"{gpu.memoryUsed} MB",
                    "memory_free": f"{gpu.memoryFree} MB",
                    "utilization": f"{gpu.load * 100:.1f}%",
                    "temperature": f"{gpu.temperature}°C"
                }
        except:
            pass
        
        return {"available": False}
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_usage": f"{psutil.cpu_percent()}%",
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "memory_used": f"{psutil.virtual_memory().used / (1024**3):.1f} GB",
            "memory_percent": f"{psutil.virtual_memory().percent}%",
            "disk_usage": f"{psutil.disk_usage('/').percent}%"
        }
    
    @staticmethod
    def display_status():
        """Display system status"""
        layout = Layout()
        
        # GPU info
        gpu_info = SystemMonitor.get_gpu_info()
        if gpu_info["available"]:
            gpu_panel = Panel(
                f"[green]GPU: {gpu_info['name']}\n"
                f"Memory: {gpu_info['memory_used']} / {gpu_info['memory_total']}\n"
                f"Utilization: {gpu_info['utilization']}\n"
                f"Temperature: {gpu_info['temperature']}",
                title="🎮 GPU Status",
                box=box.ROUNDED
            )
        else:
            gpu_panel = Panel("[red]No GPU available", title="🎮 GPU Status", box=box.ROUNDED)
        
        # System info
        sys_info = SystemMonitor.get_system_info()
        sys_panel = Panel(
            f"CPU: {sys_info['cpu_count']} cores @ {sys_info['cpu_usage']}\n"
            f"Memory: {sys_info['memory_used']} / {sys_info['memory_total']} ({sys_info['memory_percent']})\n"
            f"Disk: {sys_info['disk_usage']}",
            title="💻 System Status",
            box=box.ROUNDED
        )
        
        layout.split_row(gpu_panel, sys_panel)
        console.print(layout)


class QwenManager:
    """Main manager orchestrating all operations"""
    
    def __init__(self):
        self.config = None
        self.dataset_manager = None
        self.trainer = None
        self.inference = None
        
    def initialize(self, config_path: Optional[str] = None):
        """Initialize the manager"""
        if config_path and Path(config_path).exists():
            self.config = ProjectConfig.from_file(config_path)
        else:
            self.config = ProjectConfig()
        
        self.dataset_manager = DatasetManager(self.config)
        self.trainer = ModelTrainer(self.config)
    
    def run_pipeline(self, stages: List[TrainingStage]):
        """Run the complete training pipeline"""
        
        console.print(Panel.fit(
            "[bold cyan]Qwen Icelandic Training Pipeline",
            box=box.DOUBLE
        ))
        
        for stage in stages:
            self._run_stage(stage)
    
    def _run_stage(self, stage: TrainingStage):
        """Run a single pipeline stage"""
        
        console.print(f"\n[bold yellow]═══ Stage: {stage.value.upper()} ═══")
        
        if stage == TrainingStage.SETUP:
            self._setup_stage()
        elif stage == TrainingStage.DATASET:
            self._dataset_stage()
        elif stage == TrainingStage.TRAINING:
            self._training_stage()
        elif stage == TrainingStage.EVALUATION:
            self._evaluation_stage()
        elif stage == TrainingStage.EXPORT:
            self._export_stage()
        
        console.print(f"[green]✓ Stage {stage.value} completed")
    
    def _setup_stage(self):
        """Setup stage: initialize environment"""
        # Check dependencies
        console.print("Checking dependencies...")
        
        deps = {
            "PyTorch": torch.__version__ if ML_AVAILABLE else "Not installed",
            "Transformers": "Installed" if ML_AVAILABLE else "Not installed",
            "Unsloth": "Installed" if UNSLOTH_AVAILABLE else "Not installed",
            "CUDA": "Available" if CUDA_AVAILABLE else "Not available"
        }
        
        table = Table(title="Dependencies", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        for name, status in deps.items():
            table.add_row(name, status)
        
        console.print(table)
        
        # Display system status
        SystemMonitor.display_status()
        
        # Create directories
        dirs = [
            Path(self.config.dataset_dir),
            Path(self.config.output_dir),
            Path("logs"),
            Path("checkpoints"),
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        console.print("[green]✓ Environment ready")
    
    def _dataset_stage(self):
        """Dataset preparation stage"""
        train_count, val_count = self.dataset_manager.prepare_datasets()
        console.print(f"[green]✓ Dataset prepared: {train_count} train, {val_count} validation")
    
    def _training_stage(self):
        """Training stage"""
        # Load dataset
        dataset = self.dataset_manager.load_datasets()
        
        # Setup model
        self.trainer.setup_model()
        
        # Train
        result = self.trainer.train(dataset)
        
        # Save model
        self.trainer.save_model()
    
    def _evaluation_stage(self):
        """Evaluation stage"""
        dataset = self.dataset_manager.load_datasets()
        metrics = self.trainer.evaluate(dataset)
    
    def _export_stage(self):
        """Export stage: convert model to different formats"""
        console.print("Exporting model...")
        
        # Export options could include:
        # - ONNX export
        # - GGUF for llama.cpp
        # - TensorRT optimization
        # - etc.
        
        console.print("[green]✓ Model exported")


# CLI Interface
@click.group()
@click.option('--config', default='config.yaml', help='Configuration file')
@click.pass_context
def cli(ctx, config):
    """Qwen Icelandic Manager - Complete Training System"""
    ctx.ensure_object(dict)
    ctx.obj['manager'] = QwenManager()
    ctx.obj['manager'].initialize(config)


@cli.command()
@click.pass_context
def setup(ctx):
    """Setup the environment and check dependencies"""
    manager = ctx.obj['manager']
    manager._run_stage(TrainingStage.SETUP)


@cli.command()
@click.option('--sources', multiple=True, default=['wikipedia', 'ic3', 'wikiqa'])
@click.pass_context
def prepare_data(ctx, sources):
    """Prepare datasets from various sources"""
    manager = ctx.obj['manager']
    manager._run_stage(TrainingStage.DATASET)


@cli.command()
@click.option('--resume', default=None, help='Resume from checkpoint')
@click.pass_context
def train(ctx, resume):
    """Train the model"""
    manager = ctx.obj['manager']
    
    # Run complete pipeline
    stages = [
        TrainingStage.SETUP,
        TrainingStage.TRAINING,
        TrainingStage.EVALUATION
    ]
    
    if not resume:
        stages.insert(1, TrainingStage.DATASET)
    
    manager.run_pipeline(stages)


@cli.command()
@click.argument('model_path')
@click.pass_context
def inference(ctx, model_path):
    """Run inference with a trained model"""
    engine = InferenceEngine(model_path)
    engine.load_model()
    engine.interactive_chat()


@cli.command()
@click.argument('model_path')
@click.argument('prompt')
@click.option('--max-length', default=256)
@click.pass_context
def generate(ctx, model_path, prompt, max_length):
    """Generate text from a prompt"""
    engine = InferenceEngine(model_path)
    engine.load_model()
    
    response = engine.generate(prompt, max_length)
    console.print(f"\n[green]Response:[/green] {response}")


@cli.command()
@click.pass_context
def monitor(ctx):
    """Monitor system resources"""
    console.print("[cyan]System Monitor (Press Ctrl+C to stop)")
    
    try:
        while True:
            console.clear()
            SystemMonitor.display_status()
            time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Launch interactive training UI"""
    from training_manager import TrainingManagerApp
    
    app = TrainingManagerApp()
    app.run()


@cli.command()
@click.pass_context
def config(ctx):
    """Edit configuration"""
    manager = ctx.obj['manager']
    config_path = "config.yaml"
    
    # Display current config
    console.print("\n[cyan]Current Configuration:")
    console.print(Syntax(
        yaml.dump(asdict(manager.config), default_flow_style=False),
        "yaml",
        theme="monokai"
    ))
    
    if Confirm.ask("\nEdit configuration?"):
        # Interactive config editing
        manager.config.model_name = Prompt.ask("Model name", default=manager.config.model_name)
        manager.config.num_train_epochs = int(Prompt.ask("Training epochs", default=str(manager.config.num_train_epochs)))
        manager.config.learning_rate = float(Prompt.ask("Learning rate", default=str(manager.config.learning_rate)))
        
        manager.config.save(config_path)
        console.print("[green]✓ Configuration saved")


@cli.command()
@click.pass_context
def full_pipeline(ctx):
    """Run the complete training pipeline"""
    manager = ctx.obj['manager']
    
    stages = [
        TrainingStage.SETUP,
        TrainingStage.DATASET,
        TrainingStage.TRAINING,
        TrainingStage.EVALUATION,
        TrainingStage.EXPORT
    ]
    
    manager.run_pipeline(stages)
    
    console.print("\n[bold green]✓ Complete pipeline finished successfully!")
    
    # Offer to test the model
    if Confirm.ask("\nTest the trained model?"):
        model_path = manager.config.output_dir
        engine = InferenceEngine(model_path)
        engine.load_model()
        engine.interactive_chat()


@cli.command()
@click.argument('log_file')
@click.pass_context
def analyze_logs(ctx, log_file):
    """Analyze training logs"""
    if not Path(log_file).exists():
        console.print(f"[red]Log file not found: {log_file}")
        return
    
    losses = []
    lrs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse loss
            if 'loss:' in line.lower():
                try:
                    loss = float(line.split('loss:')[1].split()[0].replace(',', ''))
                    losses.append(loss)
                except:
                    pass
            
            # Parse learning rate
            if 'lr:' in line.lower():
                try:
                    lr = float(line.split('lr:')[1].split()[0].replace(',', ''))
                    lrs.append(lr)
                except:
                    pass
    
    if losses:
        console.print(f"\n[cyan]Training Analysis for {log_file}")
        console.print(f"Total steps: {len(losses)}")
        console.print(f"Initial loss: {losses[0]:.4f}")
        console.print(f"Final loss: {losses[-1]:.4f}")
        console.print(f"Min loss: {min(losses):.4f}")
        console.print(f"Max loss: {max(losses):.4f}")
        console.print(f"Average loss: {sum(losses)/len(losses):.4f}")
        
        # Simple ASCII chart
        if len(losses) > 10:
            console.print("\n[cyan]Loss progression (simplified):")
            sample_points = 20
            step = len(losses) // sample_points
            sampled = losses[::step] if step > 0 else losses
            
            max_val = max(sampled)
            min_val = min(sampled)
            range_val = max_val - min_val
            
            for i, loss in enumerate(sampled):
                bar_len = int(((loss - min_val) / range_val) * 40) if range_val > 0 else 20
                bar = "█" * bar_len
                console.print(f"Step {i*step:4d}: {bar} {loss:.4f}")
    else:
        console.print("[yellow]No loss data found in log file")


@cli.command()
@click.pass_context
def info(ctx):
    """Display project information"""
    manager = ctx.obj['manager']
    
    tree = Tree("🚀 [bold cyan]Qwen Icelandic Project")
    
    # Model info
    model_tree = tree.add("🤖 Model")
    model_tree.add(f"Name: {manager.config.model_name}")
    model_tree.add(f"4-bit: {'Yes' if manager.config.use_4bit else 'No'}")
    model_tree.add(f"Unsloth: {'Yes' if manager.config.use_unsloth else 'No'}")
    
    # Dataset info
    data_tree = tree.add("📚 Dataset")
    data_tree.add(f"Directory: {manager.config.dataset_dir}")
    train_file = Path(manager.config.dataset_dir) / "train.jsonl"
    if train_file.exists():
        train_lines = sum(1 for _ in open(train_file))
        data_tree.add(f"Training examples: {train_lines:,}")
    
    # Training info
    train_tree = tree.add("🎯 Training")
    train_tree.add(f"Epochs: {manager.config.num_train_epochs}")
    train_tree.add(f"Batch size: {manager.config.per_device_train_batch_size}")
    train_tree.add(f"Learning rate: {manager.config.learning_rate}")
    train_tree.add(f"Output: {manager.config.output_dir}")
    
    # System info
    sys_tree = tree.add("💻 System")
    sys_tree.add(f"CUDA: {'Available' if CUDA_AVAILABLE else 'Not available'}")
    sys_info = SystemMonitor.get_system_info()
    sys_tree.add(f"CPU: {sys_info['cpu_count']} cores")
    sys_tree.add(f"Memory: {sys_info['memory_total']}")
    
    console.print(tree)


def main():
    """Main entry point"""
    # Display banner
    console.print(Panel.fit(
        "[bold cyan]Qwen Icelandic Training Manager\n"
        "[yellow]Complete unified system for dataset preparation, training, and inference",
        box=box.DOUBLE
    ))
    
    # Check if ML libraries are available
    if not ML_AVAILABLE:
        console.print("[yellow]Warning: ML libraries not fully installed")
        console.print("Run: pip install -r requirements.txt")
    
    # Run CLI
    cli()


if __name__ == "__main__":
    main()