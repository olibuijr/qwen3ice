#!/usr/bin/env python3
"""
Simple script to download and save Icelandic datasets
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import random

def main():
    output_dir = Path("./data/icelandic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    
    print("Downloading Icelandic datasets...")
    print("=" * 60)
    
    # 1. Wikipedia
    print("\n[1] Downloading Wikipedia...")
    try:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.is", split="train")
        print(f"  ✓ Downloaded {len(wiki)} Wikipedia articles")
        
        # Sample and format
        for i in range(min(5000, len(wiki))):
            article = wiki[i]
            text = f"{article['title']}\n\n{article['text']}"
            if len(text) > 200:
                all_examples.append({
                    "messages": [
                        {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður."},
                        {"role": "user", "content": "Haltu áfram: " + text[:300]},
                        {"role": "assistant", "content": text[300:800] if len(text) > 800 else text[300:]}
                    ]
                })
        print(f"  Processed {len(all_examples)} examples so far")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 2. IC3
    print("\n[2] Downloading IC3...")
    try:
        ic3 = load_dataset("mideind/icelandic-common-crawl-corpus-IC3", split="train")
        print(f"  ✓ Downloaded {len(ic3)} IC3 documents")
        
        # Sample and format
        initial_len = len(all_examples)
        for i in range(min(5000, len(ic3))):
            doc = ic3[i]
            text = doc.get('text', '')
            if len(text) > 200:
                all_examples.append({
                    "messages": [
                        {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður."},
                        {"role": "user", "content": "Haltu áfram: " + text[:300]},
                        {"role": "assistant", "content": text[300:800] if len(text) > 800 else text[300:]}
                    ]
                })
        print(f"  Added {len(all_examples) - initial_len} examples")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 3. Wiki QA
    print("\n[3] Downloading Wiki QA...")
    try:
        wikiqa = load_dataset("mideind/icelandic_wiki_qa", split="train")
        print(f"  ✓ Downloaded {len(wikiqa)} Q&A pairs")
        
        initial_len = len(all_examples)
        for qa in wikiqa:
            if qa.get('query') and qa.get('answer'):
                all_examples.append({
                    "messages": [
                        {"role": "system", "content": "Þú svarar spurningum."},
                        {"role": "user", "content": qa['query']},
                        {"role": "assistant", "content": qa['answer']}
                    ]
                })
        print(f"  Added {len(all_examples) - initial_len} examples")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Total examples collected: {len(all_examples)}")
    
    if all_examples:
        # Shuffle
        random.seed(42)
        random.shuffle(all_examples)
        
        # Split
        split_point = int(len(all_examples) * 0.95)
        train_data = all_examples[:split_point]
        val_data = all_examples[split_point:]
        
        # Save as JSONL
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "validation.jsonl"
        
        print(f"\nSaving {len(train_data)} training examples to {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            for ex in train_data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"Saving {len(val_data)} validation examples to {val_file}")
        with open(val_file, 'w', encoding='utf-8') as f:
            for ex in val_data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print("\n✓ Dataset preparation complete!")
        print(f"  Training: {len(train_data)} examples")
        print(f"  Validation: {len(val_data)} examples")
        print(f"  Location: {output_dir}")
    else:
        print("\n✗ No data was collected!")

if __name__ == "__main__":
    main()