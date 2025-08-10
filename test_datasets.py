#!/usr/bin/env python3
"""Test script to check which Icelandic datasets are accessible"""

import os
os.environ['HF_DATASETS_OFFLINE'] = '0'  # Force online mode

from datasets import load_dataset
import sys

def test_dataset(name, config=None, split="train"):
    """Test loading a single dataset"""
    print(f"\nTesting: {name}")
    print("-" * 40)
    try:
        if config:
            print(f"  Loading with config '{config}'...")
            dataset = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
        else:
            print(f"  Loading...")
            dataset = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
        
        # Try to get first example
        first_example = next(iter(dataset))
        print(f"  ✓ Success! First example keys: {list(first_example.keys())}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:200]}")
        return False

def main():
    print("=" * 60)
    print("TESTING ICELANDIC DATASETS AVAILABILITY")
    print("=" * 60)
    
    datasets_to_test = [
        # Primary datasets
        ("wikipedia", "20231101.is"),  # Wikipedia with config
        ("oscar-corpus/OSCAR-2301", "is"),  # OSCAR with language config
        
        # Icelandic specific datasets (without IGC for now)
        ("mideind/icelandic-common-crawl-corpus-IC3", None),
        ("mideind/icelandic_wiki_qa", None),
        
        # Try simpler datasets first
        ("squad", None),  # Test with known working dataset
    ]
    
    successful = []
    failed = []
    
    for dataset_info in datasets_to_test:
        if len(dataset_info) == 2:
            name, config = dataset_info
            success = test_dataset(name, config)
        else:
            name = dataset_info[0]
            success = test_dataset(name)
        
        if success:
            successful.append(name)
        else:
            failed.append(name)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}")
    for name in successful:
        print(f"  ✓ {name}")
    
    print(f"\nFailed: {len(failed)}")
    for name in failed:
        print(f"  ✗ {name}")
    
    return len(successful) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)