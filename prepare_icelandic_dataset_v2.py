#!/usr/bin/env python3
"""
Icelandic Dataset Preparation Pipeline for Qwen3-4B Fine-tuning
Version 2: Fixed for new HuggingFace datasets library
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm


class IcelandicDatasetPreparer:
    """Prepare and combine Icelandic datasets for fine-tuning"""
    
    def __init__(self, output_dir: str = "./data/icelandic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def load_wikipedia_icelandic(self) -> Optional[Dataset]:
        """Load Icelandic Wikipedia dataset using wikimedia/wikipedia"""
        print("\n[Wikipedia] Loading Icelandic Wikipedia...")
        try:
            # Use the new wikimedia/wikipedia dataset
            print("  Using wikimedia/wikipedia dataset...")
            dataset = load_dataset(
                "wikimedia/wikipedia", 
                "20231101.is",
                split="train",
                streaming=False  # Download full dataset
            )
            print(f"  ✓ Loaded Wikipedia with {len(dataset):,} articles")
            return dataset
        except Exception as e:
            print(f"  ✗ Could not load Wikipedia: {e}")
            return None
    
    def load_oscar_icelandic(self) -> Optional[Dataset]:
        """Load OSCAR Icelandic subset"""
        print("\n[OSCAR] Loading OSCAR Icelandic corpus...")
        print("  Note: OSCAR requires authentication. Trying alternative...")
        try:
            # Try OSCAR 2201 which might be available
            dataset = load_dataset("oscar-corpus/OSCAR-2201", "is", split="train", streaming=False)
            print(f"  ✓ Loaded OSCAR with {len(dataset):,} examples")
            return dataset
        except Exception as e:
            print(f"  ✗ OSCAR requires authentication: {e}")
            print("  Skipping OSCAR dataset (requires HuggingFace login)")
            return None
    
    def load_icelandic_cc(self) -> Optional[Dataset]:
        """Load Icelandic Common Crawl corpus"""
        print("\n[IC3] Loading Icelandic Common Crawl...")
        try:
            # Try different dataset names
            dataset = load_dataset("mideind/icelandic-common-crawl-corpus-IC3", split="train", streaming=False)
            print(f"  ✓ Loaded IC3 with {len(dataset):,} examples")
            return dataset
        except Exception as e:
            print(f"  ✗ Could not load IC3: {e}")
            return None
    
    def load_icelandic_wiki_qa(self) -> Optional[Dataset]:
        """Load Icelandic Wiki QA dataset"""
        print("\n[Wiki QA] Loading Icelandic Wiki QA...")
        try:
            dataset = load_dataset("mideind/icelandic_wiki_qa", split="train", streaming=False)
            print(f"  ✓ Loaded Wiki QA with {len(dataset):,} Q&A pairs")
            return dataset
        except Exception as e:
            print(f"  ✗ Could not load Wiki QA: {e}")
            return None
    
    def load_althingi_corpus(self) -> Optional[Dataset]:
        """Load Althingi Parliamentary Corpus if available"""
        print("\n[Althingi] Loading Icelandic Parliamentary Corpus...")
        try:
            dataset = load_dataset("language-and-voice-lab/althingi_asr", split="train", streaming=False)
            print(f"  ✓ Loaded Althingi with {len(dataset):,} examples")
            return dataset
        except Exception as e:
            print(f"  ✗ Could not load Althingi: {e}")
            return None
    
    def load_cc100_icelandic(self) -> Optional[Dataset]:
        """Load CC-100 Icelandic subset"""
        print("\n[CC-100] Loading CC-100 Icelandic...")
        try:
            # Try loading CC-100 Icelandic
            dataset = load_dataset("cc100", lang="is", split="train", streaming=False)
            print(f"  ✓ Loaded CC-100 with {len(dataset):,} examples")
            return dataset
        except Exception as e:
            print(f"  ✗ Could not load CC-100: {e}")
            return None
    
    def format_for_training(self, text: str, max_length: int = 2048) -> Dict[str, str]:
        """Format text for instruction fine-tuning"""
        # Create varied instruction templates
        instruction_templates = [
            "Haltu áfram með eftirfarandi texta:",
            "Ljúktu við eftirfarandi málsgrein:",
            "Skrifaðu framhald á:",
            "Hvað kemur næst í þessum texta?",
            "Útskýrðu og haltu áfram með:",
        ]
        
        instruction = random.choice(instruction_templates)
        
        # Truncate text if needed
        if len(text) > max_length:
            text = text[:max_length]
        
        # Split text for instruction format
        split_point = min(500, len(text) // 2)
        
        formatted = {
            "messages": [
                {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku af mikilli færni."},
                {"role": "user", "content": f"{instruction}\n\n{text[:split_point]}"},
                {"role": "assistant", "content": text[split_point:] if len(text) > split_point else text}
            ]
        }
        
        return formatted
    
    def process_dataset(self, dataset: Dataset, dataset_name: str, max_examples: Optional[int] = None) -> List[Dict]:
        """Process a single dataset into training format"""
        processed_examples = []
        
        if dataset is None:
            return processed_examples
        
        # Sample if needed
        if max_examples and len(dataset) > max_examples:
            print(f"  Sampling {max_examples:,} from {len(dataset):,} examples")
            dataset = dataset.shuffle(seed=42).select(range(max_examples))
        
        print(f"  Processing {len(dataset):,} examples from {dataset_name}...")
        
        for example in tqdm(dataset, desc=f"  {dataset_name}", leave=False):
            formatted = None
            
            # Handle Q&A format (Wiki QA)
            if "query" in example and "answer" in example:
                formatted = {
                    "messages": [
                        {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem svarar spurningum á íslensku."},
                        {"role": "user", "content": example.get("query", example.get("question", ""))},
                        {"role": "assistant", "content": example["answer"]}
                    ]
                }
            
            # Handle Wikipedia format
            elif "title" in example and "text" in example:
                text = f"{example['title']}\n\n{example['text']}"
                if len(text.strip()) > 100:
                    formatted = self.format_for_training(text)
            
            # Handle standard text format
            elif "text" in example:
                text = example["text"]
                if len(text.strip()) > 100:
                    formatted = self.format_for_training(text)
            
            # Handle other text fields
            elif "content" in example:
                text = example["content"]
                if len(text.strip()) > 100:
                    formatted = self.format_for_training(text)
            
            if formatted:
                processed_examples.append(formatted)
        
        print(f"  Processed {len(processed_examples):,} valid examples")
        return processed_examples
    
    def prepare_full_dataset(self, use_all_data: bool = True, max_per_dataset: Optional[int] = None):
        """Prepare the full Icelandic training dataset"""
        print("\n" + "=" * 60)
        print("ICELANDIC DATASET PREPARATION")
        print("=" * 60)
        print(f"Mode: {'Using ALL available data' if use_all_data else f'Limited to {max_per_dataset} per dataset'}")
        
        all_examples = []
        
        # Load and process each dataset
        datasets_config = [
            ("Wikipedia", self.load_wikipedia_icelandic),
            ("IC3", self.load_icelandic_cc),
            ("Wiki QA", self.load_icelandic_wiki_qa),
            ("CC-100", self.load_cc100_icelandic),
            ("OSCAR", self.load_oscar_icelandic),  # Try but might fail due to auth
            ("Althingi", self.load_althingi_corpus),
        ]
        
        dataset_stats = {}
        
        for name, loader_func in datasets_config:
            dataset = loader_func()
            if dataset:
                max_examples = None if use_all_data else max_per_dataset
                examples = self.process_dataset(dataset, name, max_examples)
                all_examples.extend(examples)
                dataset_stats[name] = len(examples)
        
        # Print statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        for name, count in dataset_stats.items():
            print(f"{name:15} : {count:,} examples")
        print("-" * 60)
        print(f"{'TOTAL':15} : {len(all_examples):,} examples")
        print("=" * 60)
        
        if not all_examples:
            raise ValueError("No examples could be prepared!")
        
        # Create dataset
        print("\nCreating final dataset...")
        combined_dataset = Dataset.from_list(all_examples)
        
        # Shuffle
        print("Shuffling dataset...")
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        # Split into train/validation
        print("Splitting into train/validation (95/5)...")
        split_dataset = combined_dataset.train_test_split(test_size=0.05, seed=42)
        
        # Save to disk
        output_path = self.output_dir / "icelandic_training_data"
        print(f"\nSaving to {output_path}...")
        split_dataset.save_to_disk(str(output_path))
        
        # Also save as JSONL
        train_jsonl = self.output_dir / "train.jsonl"
        val_jsonl = self.output_dir / "validation.jsonl"
        
        print(f"Saving JSONL files...")
        split_dataset["train"].to_json(str(train_jsonl), orient="records", lines=True, force_ascii=False)
        split_dataset["test"].to_json(str(val_jsonl), orient="records", lines=True, force_ascii=False)
        
        print("\n" + "=" * 60)
        print("PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Training examples   : {len(split_dataset['train']):,}")
        print(f"Validation examples : {len(split_dataset['test']):,}")
        print(f"Output directory    : {self.output_dir}")
        print("=" * 60)
        
        return split_dataset


def main():
    """Main dataset preparation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Icelandic datasets for Qwen training")
    parser.add_argument("--max-per-dataset", type=int, default=None,
                        help="Maximum examples per dataset (None for all)")
    parser.add_argument("--output-dir", type=str, default="./data/icelandic",
                        help="Output directory for processed datasets")
    parser.add_argument("--use-sample", action="store_true",
                        help="Use sampling mode (limit data)")
    
    args = parser.parse_args()
    
    preparer = IcelandicDatasetPreparer(output_dir=args.output_dir)
    
    # Determine mode
    use_all_data = not args.use_sample
    max_per_dataset = args.max_per_dataset if args.use_sample else None
    
    # Prepare dataset
    try:
        dataset = preparer.prepare_full_dataset(
            use_all_data=use_all_data,
            max_per_dataset=max_per_dataset
        )
        
        print("\n✓ Dataset preparation successful!")
        print("\nNext steps:")
        print("1. Review the dataset in:", args.output_dir)
        print("2. Run training: python train_qwen_icelandic.py")
        
    except Exception as e:
        print(f"\n✗ Error during preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())