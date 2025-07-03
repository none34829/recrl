#!/usr/bin/env python
# Projection basis computation

import torch
import numpy as np
import argparse
import os
from pathlib import Path


def compute_basis(embed_weights, tol=1e-6):
    """Compute the orthogonal basis for the embedding weights using SVD.
    
    Args:
        embed_weights: Tensor of embedding weights [vocab_size, embed_dim]
        tol: Tolerance for singular values
        
    Returns:
        Q: Orthogonal basis matrix [embed_dim, rank]
    """
    # Perform SVD decomposition
    u, s, _ = torch.linalg.svd(embed_weights, full_matrices=False)
    
    # Determine effective rank based on singular values above tolerance
    r = (s > tol).sum().item()
    
    print(f"Embedding matrix shape: {embed_weights.shape}")
    print(f"Singular values: {s[:10]} ... {s[-5:]}")
    print(f"Effective rank with tol={tol}: {r} (out of {len(s)})")
    
    # Return the orthogonal basis (Q matrix)
    return u[:, :r]


def main():
    parser = argparse.ArgumentParser(description="Compute projection basis for gradient shielding")
    parser.add_argument('--ranker_ckpt', required=True, help='Path to the ranker checkpoint')
    parser.add_argument('--output', required=True, help='Path to save the basis matrix')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for singular values')
    parser.add_argument('--embedding_key', default='item_emb.weight', 
                        help='Key for embedding weights in the checkpoint')
    args = parser.parse_args()
    
    # Load the embedding weights
    print(f"Loading ranker checkpoint from {args.ranker_ckpt}")
    ckpt = torch.load(args.ranker_ckpt, map_location='cpu')
    
    if args.embedding_key not in ckpt:
        # Try to find the embedding key
        possible_keys = [k for k in ckpt.keys() if 'embed' in k.lower() or 'emb' in k.lower()]
        if possible_keys:
            print(f"Warning: {args.embedding_key} not found. Using {possible_keys[0]} instead.")
            args.embedding_key = possible_keys[0]
        else:
            raise KeyError(f"Embedding key {args.embedding_key} not found in checkpoint. "
                           f"Available keys: {list(ckpt.keys())}")
    
    emb = ckpt[args.embedding_key]
    print(f"Loaded embedding weights with shape {emb.shape}")
    
    # Compute the basis
    Q = compute_basis(emb, tol=args.tol)
    
    # Save the basis
    print(f"Saving basis matrix with shape {Q.shape} to {args.output}")
    torch.save(Q, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
