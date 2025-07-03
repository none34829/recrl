#!/usr/bin/env python
"""
Check that the projection shield is working correctly by verifying that
the gradient components in the direction of the basis vectors are near zero.

Usage:
  python check_projection.py [--dataset books|ml25m|steam]
"""

import torch, os, pathlib, argparse
from pathlib import Path
import numpy as np

def main(args):
    # Initialize project path
    proj = Path(os.getenv("PROJ", "."))
    
    # Load the projection basis
    basis_path = proj / "checkpoints" / f"basis_{args.dataset}.pt"
    if not basis_path.exists():
        print(f"Error: Basis file not found at {basis_path}")
        return 1
    
    Q = torch.load(basis_path)
    if torch.cuda.is_available():
        Q = Q.cuda()
        device = "cuda"
    else:
        device = "cpu"
    
    QT = Q.t()  # Transpose for efficient projection
    
    # Load the trained LoRA weights
    lora_path = proj / "checkpoints" / f"recrl_{args.dataset}_toy.pt"
    if not lora_path.exists():
        print(f"Error: LoRA checkpoint not found at {lora_path}")
        return 1
    
    lora = torch.load(lora_path, map_location=device)
    if isinstance(lora, dict) and 'model' in lora:
        lora = lora['model']  # Extract model state dict if it's a full checkpoint
    
    # Check projection for each LoRA layer
    max_projections = []
    layer_names = []
    
    print(f"Checking projection for {args.dataset} dataset...")
    print(f"Basis shape: {Q.shape}")
    
    for name, param in lora.items():
        if 'lora_A' in name or 'lora_B' in name:  # Check LoRA layers
            param = param.to(device)
            # Compute projection component: Q^T * param
            proj_component = QT @ param.flatten().float()
            max_proj = proj_component.abs().max().item()
            max_projections.append(max_proj)
            layer_names.append(name)
            
            print(f"{name}: max projection component = {max_proj:.8f}")
    
    # Summary statistics
    max_overall = max(max_projections)
    avg_overall = sum(max_projections) / len(max_projections)
    
    print("\nSummary:")
    print(f"Maximum projection component: {max_overall:.8f}")
    print(f"Average projection component: {avg_overall:.8f}")
    
    # Check if projection is working (should be close to numerical precision)
    if max_overall < 1e-5:
        print("\n✅ Projection shield is working correctly!")
        print("   Maximum projection component is < 1e-5 (numerical noise)")
        return 0
    else:
        print("\n❌ Projection shield may not be working correctly!")
        print(f"   Maximum projection component is {max_overall:.8f} > 1e-5")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam"], default="books")
    
    args = parser.parse_args()
    exit(main(args))
