#!/usr/bin/env python3
"""
Icelandic Dataset Preparation - Final Optimized Version
Efficiently prepares all available Icelandic datasets for Qwen3-4B training
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import time


class IcelandicDatasetPreparer:
    """Prepare and combine Icelandic datasets for fine-tuning"""
    
    def __init__(self, output_dir: str = "./data/icelandic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        self.all_examples = []
        self.stats = {}
        
    def format_text_for_training(self, text: str, max_length: int = 2048) -> Optional[Dict]:
        """Format text for instruction fine-tuning"""
        if not text or len(text.strip()) < 100:
            return None
            
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # Create varied instructions
        templates = [
            ("Haltu áfram með eftirfarandi texta:", 400),
            ("Ljúktu við eftirfarandi málsgrein:", 300),
            ("Skrifaðu framhald á:", 500),
            ("Hvað kemur næst?", 200),
        ]
        
        instruction, split_point = random.choice(templates)
        split_point = min(split_point, len(text) // 2)
        
        return {
            "messages": [
                {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
                {"role": "user", "content": f"{instruction}\n\n{text[:split_point]}"},
                {"role": "assistant", "content": text[split_point:]}
            ]
        }
    
    def process_wikipedia(self, limit: Optional[int] = None):
        """Process Wikipedia articles"""
        print("\n[1/4] Processing Wikipedia...")
        try:
            print("  Loading dataset...")
            dataset = load_dataset(
                "wikimedia/wikipedia", 
                "20231101.is",
                split="train",
                streaming=False
            )
            
            total = len(dataset)
            if limit:
                total = min(total, limit)
                dataset = dataset.select(range(total))
            
            print(f"  Processing {total:,} articles...")
            processed = 0
            
            # Process in batches for efficiency
            batch_size = 1000
            for i in tqdm(range(0, total, batch_size), desc="  Wikipedia batches"):
                batch_end = min(i + batch_size, total)
                batch = dataset[i:batch_end]
                
                for j in range(len(batch['title'])):
                    text = f"{batch['title'][j]}\n\n{batch['text'][j]}"
                    formatted = self.format_text_for_training(text)
                    if formatted:
                        self.all_examples.append(formatted)
                        processed += 1
            
            self.stats['Wikipedia'] = processed
            print(f"  ✓ Processed {processed:,} examples")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats['Wikipedia'] = 0
    
    def process_ic3(self, limit: Optional[int] = None):
        """Process Icelandic Common Crawl"""
        print("\n[2/4] Processing Icelandic Common Crawl (IC3)...")
        try:
            print("  Loading dataset...")
            dataset = load_dataset(
                "mideind/icelandic-common-crawl-corpus-IC3",
                split="train",
                streaming=False
            )
            
            total = len(dataset)
            if limit:
                total = min(total, limit)
                dataset = dataset.select(range(total))
            
            print(f"  Processing {total:,} documents...")
            processed = 0
            
            # Process in batches
            batch_size = 1000
            for i in tqdm(range(0, total, batch_size), desc="  IC3 batches"):
                batch_end = min(i + batch_size, total)
                batch = dataset[i:batch_end]
                
                for text in batch['text']:
                    formatted = self.format_text_for_training(text)
                    if formatted:
                        self.all_examples.append(formatted)
                        processed += 1
            
            self.stats['IC3'] = processed
            print(f"  ✓ Processed {processed:,} examples")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats['IC3'] = 0
    
    def process_wiki_qa(self):
        """Process Wiki QA dataset"""
        print("\n[3/4] Processing Icelandic Wiki QA...")
        try:
            print("  Loading dataset...")
            dataset = load_dataset(
                "mideind/icelandic_wiki_qa",
                split="train",
                streaming=False
            )
            
            total = len(dataset)
            print(f"  Processing {total:,} Q&A pairs...")
            processed = 0
            
            for example in tqdm(dataset, desc="  Wiki QA"):
                if example.get('query') and example.get('answer'):
                    formatted = {
                        "messages": [
                            {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem svarar spurningum."},
                            {"role": "user", "content": example['query']},
                            {"role": "assistant", "content": example['answer']}
                        ]
                    }
                    self.all_examples.append(formatted)
                    processed += 1
            
            self.stats['Wiki QA'] = processed
            print(f"  ✓ Processed {processed:,} examples")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats['Wiki QA'] = 0
    
    def process_cc100(self, limit: Optional[int] = 10000):
        """Process CC-100 Icelandic (limited due to size)"""
        print("\n[4/4] Processing CC-100 Icelandic...")
        try:
            print("  Loading dataset (streaming mode)...")
            dataset = load_dataset(
                "cc100",
                lang="is",
                split="train",
                streaming=True  # Stream to avoid memory issues
            )
            
            print(f"  Processing up to {limit:,} documents...")
            processed = 0
            
            for example in tqdm(dataset, desc="  CC-100", total=limit):
                if processed >= limit:
                    break
                    
                text = example.get('text', '')
                formatted = self.format_text_for_training(text)
                if formatted:
                    self.all_examples.append(formatted)
                    processed += 1
            
            self.stats['CC-100'] = processed
            print(f"  ✓ Processed {processed:,} examples")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats['CC-100'] = 0
    
    def save_dataset(self):
        """Save the prepared dataset"""
        if not self.all_examples:
            raise ValueError("No examples were processed!")
        
        print(f"\n[Saving] Total examples: {len(self.all_examples):,}")
        
        # Shuffle
        print("  Shuffling dataset...")
        random.seed(42)
        random.shuffle(self.all_examples)
        
        # Create dataset
        print("  Creating HuggingFace dataset...")
        dataset = Dataset.from_list(self.all_examples)
        
        # Split
        print("  Splitting (95% train, 5% validation)...")
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        
        # Save to disk
        output_path = self.output_dir / "icelandic_training_data"
        print(f"  Saving to {output_path}...")
        split_dataset.save_to_disk(str(output_path))
        
        # Save as JSONL
        train_jsonl = self.output_dir / "train.jsonl"
        val_jsonl = self.output_dir / "validation.jsonl"
        
        print("  Saving JSONL files...")
        with open(train_jsonl, 'w', encoding='utf-8') as f:
            for example in split_dataset['train']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(val_jsonl, 'w', encoding='utf-8') as f:
            for example in split_dataset['test']:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save statistics
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            stats = {
                'datasets': self.stats,
                'total_examples': len(self.all_examples),
                'train_examples': len(split_dataset['train']),
                'val_examples': len(split_dataset['test']),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            json.dump(stats, f, indent=2)
        
        return split_dataset
    
    def run(self, use_all_data: bool = True):
        """Run the complete preparation pipeline"""
        print("\n" + "=" * 60)
        print("ICELANDIC DATASET PREPARATION - OPTIMIZED")
        print("=" * 60)
        
        start_time = time.time()
        
        # Process each dataset
        if use_all_data:
            self.process_wikipedia(limit=None)
            self.process_ic3(limit=50000)  # Limit IC3 as it can be very large
            self.process_wiki_qa()
            self.process_cc100(limit=20000)  # Limit CC-100 
        else:
            # Sample mode for testing
            self.process_wikipedia(limit=1000)
            self.process_ic3(limit=1000)
            self.process_wiki_qa()
            self.process_cc100(limit=1000)
        
        # Print statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        for name, count in self.stats.items():
            print(f"{name:15} : {count:,} examples")
        print("-" * 60)
        print(f"{'TOTAL':15} : {len(self.all_examples):,} examples")
        print("=" * 60)
        
        # Save dataset
        split_dataset = self.save_dataset()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("✓ PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Training examples   : {len(split_dataset['train']):,}")
        print(f"Validation examples : {len(split_dataset['test']):,}")
        print(f"Output directory    : {self.output_dir}")
        print(f"Time elapsed        : {elapsed:.1f} seconds")
        print("=" * 60)
        
        return split_dataset


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Icelandic datasets")
    parser.add_argument("--output-dir", type=str, default="./data/icelandic",
                        help="Output directory")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample mode (faster, less data)")
    
    args = parser.parse_args()
    
    preparer = IcelandicDatasetPreparer(output_dir=args.output_dir)
    
    try:
        dataset = preparer.run(use_all_data=not args.sample)
        
        print("\n✓ Success! Dataset is ready for training.")
        print("\nNext steps:")
        print(f"1. Check dataset: ls -la {args.output_dir}")
        print("2. Run training: python train_qwen_icelandic.py")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())