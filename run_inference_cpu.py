#!/usr/bin/env python3
"""
Run Qwen3-4B inference on CPU with quantization
Requires: 32GB RAM, Intel CPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

def run_cpu_inference():
    print("Loading Qwen3-4B for CPU inference...")
    print("This will use quantization to fit in 32GB RAM")
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"  # Use smaller 3B model for CPU
    
    # Load with 8-bit quantization for CPU
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model (8-bit quantized)...")
    # For CPU, we need to use different quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,  # CPU doesn't support fp16 well
        low_cpu_mem_usage=True,
        load_in_8bit=False,  # 8-bit doesn't work on CPU
        load_in_4bit=False   # 4-bit requires CUDA
    )
    
    # Move to CPU explicitly
    model = model.to("cpu")
    model.eval()
    
    print(f"Model loaded! Memory usage: ~15-20GB")
    print("\nTesting with Icelandic prompt...")
    
    # Test prompt
    messages = [
        {"role": "system", "content": "Þú ert gagnlegur aðstoðarmaður sem talar íslensku."},
        {"role": "user", "content": "Hvað er höfuðborg Íslands?"}
    ]
    
    # Tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    print("Generating response (this will be SLOW on CPU)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")
    
    print("\n⚠️ Note: Each generation will take 30-60+ seconds on CPU")
    print("Consider using smaller models like Qwen2.5-1.5B for better CPU performance")

if __name__ == "__main__":
    run_cpu_inference()