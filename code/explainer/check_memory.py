#!/usr/bin/env python
"""
Check memory usage of GPT-J-6B with LoRA adapters and verify text generation.
"""
import argparse
import torch
import os
import sys
from pathlib import Path

# Add the project root to the path if needed
proj_dir = Path(os.getenv("PROJ", "."))
sys.path.append(str(proj_dir))

from code.explainer.load_llm import load_base, add_lora

def main(int8):
    print(f"Loading GPT-J-6B with LoRA adapters (int8={int8})...")
    tok, base = load_base(int8=int8)
    model = add_lora(base)
    
    # Check VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"VRAM usage: {vram_used:.2f} GB")
    else:
        print("CUDA not available, running on CPU")
    
    # Generate sample text
    print("\nGenerating sample text...")
    prompt = "Hello world, I am an AI assistant that"
    inputs = tok(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    print(f"Prompt: {prompt}")
    
    # Generate with timing
    import time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    end_time = time.time()
    
    # Decode and print the generated text
    generated_text = tok.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", action="store_true", help="Use 8-bit quantization")
    args = parser.parse_args()
    
    sys.exit(main(args.int8))
