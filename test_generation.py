import torch
import sys
from pathlib import Path

# Add the code directory to the path
proj = Path("C:/Codeee/recrl")
sys.path.append(str(proj/'code'))

from explainer.load_llm import load_base

def test_generation():
    print("Testing model generation capability...")
    
    # Load the model
    tokenizer, model = load_base()
    model.eval()
    
    # Test prompt
    test_prompt = "User likes fantasy novels like LOTR and Harry Potter. Explain why we recommend Mistborn."
    
    print(f"Test prompt: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer([test_prompt], return_tensors="pt").to(model.device)
    print(f"Input tokens shape: {inputs['input_ids'].shape}")
    
    # Generate with different parameters
    print("\nTesting different generation parameters:")
    
    # Test 1: Basic generation
    print("\n1. Basic generation (max_new_tokens=20):")
    with torch.no_grad():
        gen1 = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens1 = gen1[0][inputs['input_ids'].shape[1]:]
        generated1 = tokenizer.decode(new_tokens1, skip_special_tokens=True)
        print(f"Generated: '{generated1}' (length: {len(new_tokens1)})")
    
    # Test 2: Sampling
    print("\n2. Sampling generation (max_new_tokens=20):")
    with torch.no_grad():
        gen2 = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens2 = gen2[0][inputs['input_ids'].shape[1]:]
        generated2 = tokenizer.decode(new_tokens2, skip_special_tokens=True)
        print(f"Generated: '{generated2}' (length: {len(new_tokens2)})")
    
    # Test 3: Different prompt
    print("\n3. Different prompt:")
    test_prompt2 = "The book Mistborn is"
    inputs2 = tokenizer([test_prompt2], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen3 = model.generate(
            **inputs2,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens3 = gen3[0][inputs2['input_ids'].shape[1]:]
        generated3 = tokenizer.decode(new_tokens3, skip_special_tokens=True)
        print(f"Prompt: '{test_prompt2}'")
        print(f"Generated: '{generated3}' (length: {len(new_tokens3)})")

if __name__ == "__main__":
    test_generation()
