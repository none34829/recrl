#!/usr/bin/env python
"""
Simple test to verify gradient flow in the model.
"""
import torch
import os
import sys
from pathlib import Path

# Add the code directory to the path
proj = Path(os.getenv("PROJ", "."))
sys.path.append(str(proj/'code'))

from explainer.load_llm import load_base, add_lora

def test_gradient_flow():
    print("Testing gradient flow...")
    
    # Load model with 4-bit quantization
    tok, base = load_base()
    model = add_lora(base)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Enable training mode
    model.train()
    
    # Create a simple input
    input_text = "Hello world"
    inputs = tok([input_text], return_tensors="pt").to(device)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input requires grad: {inputs['input_ids'].requires_grad}")
    
    # Forward pass
    with torch.set_grad_enabled(True):
        outputs = model(**inputs)
        logits = outputs.logits
        
        print(f"Logits shape: {logits.shape}")
        print(f"Logits requires grad: {logits.requires_grad}")
        print(f"Logits grad_fn: {logits.grad_fn}")
        
        # Create a simple loss
        loss = logits.sum()
        print(f"Loss: {loss}")
        print(f"Loss requires grad: {loss.requires_grad}")
        print(f"Loss grad_fn: {loss.grad_fn}")
        
        if loss.requires_grad:
            # Backward pass
            loss.backward()
            print("✅ Backward pass successful!")
            
            # Check gradients
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
                    print(f"Parameter {name} has gradient: {param.grad.shape}")
            
            print(f"Total parameters with gradients: {grad_count}")
        else:
            print("❌ Loss doesn't require gradients!")

if __name__ == "__main__":
    test_gradient_flow()

