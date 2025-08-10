#!/usr/bin/env python3
"""
Icelandic Dataset Preparation Pipeline for Qwen3-4B Fine-tuning
Prepares and combines multiple Icelandic datasets for LLM training
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm


class IcelandicDatasetPreparer:
    """Prepare and combine Icelandic datasets for fine-tuning"""
    
    def __init__(self, output_dir: str = "./data/icelandic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_igc_corpus(self) -> Optional[Dataset]:
        """Load Icelandic Gigaword Corpus from HuggingFace"""
        print("Loading Icelandic Gigaword Corpus (IGC)...")
        try:
            # Try to load the latest IGC corpus
            print("  Attempting to load arnastofnun/IGC-2024...")
            dataset = load_dataset("arnastofnun/IGC-2024", split="train", trust_remote_code=True)
            print(f"  ✓ Loaded IGC-2024 with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"  ✗ Could not load IGC-2024: {e}")
            try:
                # Fallback to older version
                print("  Attempting to load arnastofnun/IGC-2022-1...")
                dataset = load_dataset("arnastofnun/IGC-2022-1", split="train", trust_remote_code=True)
                print(f"  ✓ Loaded IGC-2022 with {len(dataset)} examples")
                return dataset
            except Exception as e2:
                print(f"  ✗ Could not load IGC corpus: {e2}")
                return None
    
    def load_wikipedia_icelandic(self) -> Optional[Dataset]:
        """Load Icelandic Wikipedia dataset"""
        print("Loading Icelandic Wikipedia...")
        try:
            # Load Wikipedia dataset
            dataset = load_dataset("wikipedia", "20231101.is", split="train")
            print(f"Loaded Wikipedia with {len(dataset)} articles")
            return dataset
        except Exception as e:
            print(f"Could not load Wikipedia: {e}")
            return None
    
    def load_oscar_icelandic(self) -> Optional[Dataset]:
        """Load OSCAR Icelandic subset"""
        print("Loading OSCAR Icelandic corpus...")
        try:
            # Load OSCAR corpus
            dataset = load_dataset("oscar-corpus/OSCAR-2301", "is", split="train")
            print(f"Loaded OSCAR with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Could not load OSCAR: {e}")
            return None
    
    def load_greynir_corpus(self) -> Optional[Dataset]:
        """Load GreynirCorpus if available"""
        print("Loading GreynirCorpus...")
        try:
            # Try to load from HuggingFace or local path
            dataset = load_dataset("mideind/greynir-corpus", split="train")
            print(f"Loaded GreynirCorpus with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Could not load GreynirCorpus: {e}")
            return None
    
    def load_icelandic_cc(self) -> Optional[Dataset]:
        """Load Icelandic Common Crawl corpus"""
        print("Loading Icelandic Common Crawl...")
        try:
            dataset = load_dataset("mideind/icelandic-common-crawl-corpus-IC3-v2", split="train")
            print(f"Loaded IC3 with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Could not load IC3: {e}")
            return None
    
    def load_icelandic_wiki_qa(self) -> Optional[Dataset]:
        """Load Icelandic Wiki QA dataset"""
        print("Loading Icelandic Wiki QA...")
        try:
            dataset = load_dataset("mideind/icelandic_wiki_qa", split="train")
            print(f"Loaded Wiki QA with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Could not load Wiki QA: {e}")
            return None
    
    def load_icelandic_error_corpus(self) -> Optional[Dataset]:
        """Load Icelandic Error Corpus for better language understanding"""
        print("Loading Icelandic Error Corpus...")
        try:
            dataset = load_dataset("mideind/icelandic-error-corpus-IceEC", split="train")
            print(f"Loaded IceEC with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Could not load IceEC: {e}")
            return None
    
    def load_cc100_icelandic(self) -> Optional[Dataset]:
        """Load CC-100 Icelandic subset"""
        print("Loading CC-100 Icelandic...")
        try:
            # CC-100 might be available through different sources
            dataset = load_dataset("cc100", lang="is", split="train")
            print(f"Loaded CC-100 with {len(dataset)} examples")
            return dataset
        except Exception as e:
            print(f"Could not load CC-100: {e}")
            return None
    
    def format_for_training(self, text: str, max_length: int = 2048) -> Dict[str, str]:
        """Format text for instruction fine-tuning"""
        # Create instruction-following format
        instruction_templates = [
            "Þýddu eftirfarandi texta á íslensku:",
            "Skrifaðu áframhald á eftirfarandi texta:",
            "Endurskrifaðu eftirfarandi texta:",
            "Útskýrðu eftirfarandi texta á íslensku:",
            "Dragðu saman eftirfarandi texta:",
        ]
        
        # Randomly select instruction type
        instruction = random.choice(instruction_templates)
        
        # Truncate text if needed
        if len(text) > max_length:
            text = text[:max_length]
        
        # Format as conversation
        formatted = {
            "messages": [
                {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
                {"role": "user", "content": instruction + "\n\n" + text[:500]},
                {"role": "assistant", "content": text[500:] if len(text) > 500 else text}
            ]
        }
        
        return formatted
    
    def combine_datasets(self, datasets: List[Dataset], sample_sizes: Optional[Dict[str, int]] = None, dataset_names: Optional[List[str]] = None) -> Dataset:
        """Combine multiple datasets with optional sampling"""
        combined_examples = []
        
        if dataset_names is None:
            dataset_names = []
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
                
            # Sample if specified
            if sample_sizes and i in sample_sizes:
                num_samples = min(sample_sizes[i], len(dataset))
                dataset = dataset.shuffle(seed=42).select(range(num_samples))
            
            # Extract text content
            for example in tqdm(dataset, desc=f"Processing {dataset_names[i] if i < len(dataset_names) else f'dataset {i+1}'}"):
                text = None
                formatted = None
                
                # Handle Q&A format datasets
                if "question" in example and "answer" in example:
                    # Direct Q&A format
                    formatted = {
                        "messages": [
                            {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
                            {"role": "user", "content": example["question"]},
                            {"role": "assistant", "content": example["answer"]}
                        ]
                    }
                # Handle error correction datasets
                elif "correct" in example and "incorrect" in example:
                    formatted = {
                        "messages": [
                            {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem leiðréttir íslensku."},
                            {"role": "user", "content": f"Leiðréttu þessa setningu: {example['incorrect']}"},
                            {"role": "assistant", "content": example["correct"]}
                        ]
                    }
                # Handle standard text formats
                elif "text" in example:
                    text = example["text"]
                elif "content" in example:
                    text = example["content"]
                elif "sentence" in example:
                    text = example["sentence"]
                elif "title" in example and "text" in example:
                    text = f"{example['title']}\n\n{example['text']}"
                
                # Format regular text if not already formatted
                if not formatted and text and len(text.strip()) > 100:
                    formatted = self.format_for_training(text)
                
                if formatted:
                    combined_examples.append(formatted)
        
        # Create dataset from combined examples
        print(f"Combined {len(combined_examples)} examples")
        return Dataset.from_list(combined_examples)
    
    def prepare_full_dataset(self, max_examples: Optional[int] = None, use_all_data: bool = True):
        """Prepare the full Icelandic training dataset
        
        Args:
            max_examples: Maximum number of examples to use (None for all data)
            use_all_data: If True, use all available data without sampling
        """
        print("Starting Icelandic dataset preparation...")
        print(f"Mode: {'Using ALL available data' if use_all_data else f'Sampling up to {max_examples} examples'}")
        
        # Load all available datasets
        datasets = []
        dataset_names = []
        
        # Primary high-quality datasets
        igc = self.load_igc_corpus()
        if igc:
            datasets.append(igc)
            dataset_names.append("IGC")
        
        wiki = self.load_wikipedia_icelandic()
        if wiki:
            datasets.append(wiki)
            dataset_names.append("Wikipedia")
        
        # Web-crawled datasets
        oscar = self.load_oscar_icelandic()
        if oscar:
            datasets.append(oscar)
            dataset_names.append("OSCAR")
        
        cc100 = self.load_cc100_icelandic()
        if cc100:
            datasets.append(cc100)
            dataset_names.append("CC-100")
        
        # Curated Icelandic datasets
        greynir = self.load_greynir_corpus()
        if greynir:
            datasets.append(greynir)
            dataset_names.append("GreynirCorpus")
        
        ic3 = self.load_icelandic_cc()
        if ic3:
            datasets.append(ic3)
            dataset_names.append("IC3")
        
        # Specialized datasets
        wiki_qa = self.load_icelandic_wiki_qa()
        if wiki_qa:
            datasets.append(wiki_qa)
            dataset_names.append("Wiki-QA")
        
        error_corpus = self.load_icelandic_error_corpus()
        if error_corpus:
            datasets.append(error_corpus)
            dataset_names.append("IceEC")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Print dataset statistics
        print("\n=== Dataset Statistics ===")
        total_examples = 0
        for name, dataset in zip(dataset_names, datasets):
            if dataset:
                print(f"{name}: {len(dataset):,} examples")
                total_examples += len(dataset)
        print(f"Total available: {total_examples:,} examples")
        print("=" * 26)
        
        # Determine sampling strategy
        sample_sizes = None
        if not use_all_data and max_examples:
            # Sample proportionally based on dataset quality and size
            total_proportion = 100
            proportions = {
                "IGC": 30,        # Highest quality
                "Wikipedia": 20,   # High quality
                "OSCAR": 15,      # Web data
                "CC-100": 10,     # Web data
                "GreynirCorpus": 10,  # News data
                "IC3": 8,         # Common Crawl
                "Wiki-QA": 5,     # Q&A format
                "IceEC": 2,       # Error corpus
            }
            
            sample_sizes = {}
            for i, name in enumerate(dataset_names):
                if name in proportions:
                    proportion = proportions[name] / 100
                    target_samples = int(max_examples * proportion)
                    actual_samples = min(len(datasets[i]), target_samples)
                    sample_sizes[i] = actual_samples
            
            print(f"\nSampling strategy:")
            for i, name in enumerate(dataset_names):
                if i in sample_sizes:
                    print(f"  {name}: {sample_sizes[i]:,} examples")
        else:
            print("\nUsing ALL available data without sampling...")
        
        # Combine datasets
        combined_dataset = self.combine_datasets(datasets, sample_sizes, dataset_names)
        
        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        # Split into train/validation
        split_dataset = combined_dataset.train_test_split(test_size=0.05, seed=42)
        
        # Save to disk
        output_path = self.output_dir / "icelandic_training_data"
        split_dataset.save_to_disk(str(output_path))
        
        # Also save as JSONL for compatibility
        train_jsonl = self.output_dir / "train.jsonl"
        val_jsonl = self.output_dir / "validation.jsonl"
        
        split_dataset["train"].to_json(str(train_jsonl), orient="records", lines=True)
        split_dataset["test"].to_json(str(val_jsonl), orient="records", lines=True)
        
        print(f"Dataset saved to {output_path}")
        print(f"Training examples: {len(split_dataset['train'])}")
        print(f"Validation examples: {len(split_dataset['test'])}")
        
        return split_dataset


def main():
    """Main dataset preparation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Icelandic datasets for Qwen training")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum number of examples (None for all data)")
    parser.add_argument("--use-sample", action="store_true",
                        help="Use sampling instead of all data")
    parser.add_argument("--output-dir", type=str, default="./data/icelandic",
                        help="Output directory for processed datasets")
    
    args = parser.parse_args()
    
    preparer = IcelandicDatasetPreparer(output_dir=args.output_dir)
    
    # Determine whether to use all data or sample
    use_all_data = not args.use_sample
    max_examples = args.max_examples if args.use_sample else None
    
    if use_all_data:
        print("=" * 60)
        print("PREPARING FULL ICELANDIC DATASET")
        print("This will download and process ALL available Icelandic data")
        print("This may take significant time and disk space!")
        print("=" * 60)
    
    # Prepare dataset
    dataset = preparer.prepare_full_dataset(
        max_examples=max_examples,
        use_all_data=use_all_data
    )
    
    print("\nDataset preparation complete!")
    print("Next steps:")
    print("1. Review the generated dataset in ./data/icelandic/")
    print("2. Run the training script: python train_qwen_icelandic.py")


if __name__ == "__main__":
    main()