#!/usr/bin/env python
# GPU sanity check script

import torch
from transformers import AutoModelForCausalLM

def test_torch_gpu():
    print("\n=== PyTorch GPU Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Matrix multiplication test
        x = torch.randn(4096, 4096, device='cuda')
        result = (x @ x).sum().item() != 0
        print(f"Matrix multiplication test: {'PASSED' if result else 'FAILED'}")
    else:
        print("CUDA not available. Please check your GPU setup.")

def test_transformers_gpu():
    print("\n=== Transformers GPU Test ===")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "bert-base-uncased",
            torch_dtype="auto",
            device_map="auto"
        )
        device = next(model.parameters()).device
        print(f"Model loaded successfully on: {device}")
        print(f"Transformers GPU test: {'PASSED' if 'cuda' in str(device) else 'FAILED'}")
    except Exception as e:
        print(f"Transformers GPU test FAILED: {str(e)}")

def test_wandb():
    print("\n=== W&B Connection Test ===")
    try:
        import wandb
        run = wandb.init(project="shielded-recrl", name="env_ping")
        wandb.log({"alive": 1})
        wandb.finish()
        print("W&B connection test: PASSED")
    except Exception as e:
        print(f"W&B connection test FAILED: {str(e)}")

if __name__ == "__main__":
    print("Running GPU and environment tests...")
    test_torch_gpu()
    test_transformers_gpu()
    test_wandb()
    print("\nAll tests completed.")
