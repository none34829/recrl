#!/usr/bin/env python
# Script to compute and save the projection basis

import torch
import argparse
import os
from pathlib import Path
from basis import compute_basis


def main():
    parser = argparse.ArgumentParser(description="Compute and save projection basis")
    parser.add_argument('--dataset', required=True, choices=['books', 'ml', 'steam'],
                        help='Dataset name')
    parser.add_argument('--proj_dir', default=os.environ.get('PROJ', '.'),
                        help='Project directory')
    args = parser.parse_args()
    
    # Set paths
    proj_dir = Path(args.proj_dir)
    ckpt_dir = proj_dir / 'checkpoints'
    ranker_ckpt = ckpt_dir / f"sasrec_{args.dataset}.pt"
    basis_path = ckpt_dir / f"basis_{args.dataset}.pt"
    
    # Check if files exist
    if not ranker_ckpt.exists():
        raise FileNotFoundError(f"Ranker checkpoint not found: {ranker_ckpt}")
    
    # Load the embedding weights
    print(f"Loading ranker checkpoint from {ranker_ckpt}")
    ckpt = torch.load(ranker_ckpt, map_location='cpu')
    
    # Try to find the embedding key
    embedding_key = 'item_emb.weight'
    if embedding_key not in ckpt:
        possible_keys = [k for k in ckpt.keys() if 'embed' in k.lower() or 'emb' in k.lower()]
        if possible_keys:
            print(f"Warning: {embedding_key} not found. Using {possible_keys[0]} instead.")
            embedding_key = possible_keys[0]
        else:
            raise KeyError(f"Embedding key not found in checkpoint. "
                           f"Available keys: {list(ckpt.keys())}")
    
    emb = ckpt[embedding_key]
    print(f"Loaded embedding weights with shape {emb.shape}")
    
    # Compute the basis
    Q = compute_basis(emb)
    
    # Save the basis
    print(f"Saving basis matrix with shape {Q.shape} to {basis_path}")
    torch.save(Q, basis_path)
    print("Done!")


if __name__ == "__main__":
    main()
