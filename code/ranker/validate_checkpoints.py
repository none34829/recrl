#!/usr/bin/env python
"""
validate_checkpoints.py: Validate SASRec model checkpoints

Usage:
  python validate_checkpoints.py [--dataset books|ml25m|steam|all]
"""

import argparse
import json
import os
import hashlib
from pathlib import Path
import torch

def validate_checkpoint(dataset, proj_dir):
    """Validate a SASRec checkpoint for a specific dataset."""
    print(f"\n=== Validating {dataset} checkpoint ===")
    
    # Check paths
    ckpt_path = proj_dir / "checkpoints" / f"sasrec_{dataset}.pt"
    stats_path = proj_dir / "checkpoints" / f"sasrec_{dataset}_stats.json"
    
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False
    
    if not stats_path.exists():
        print(f"❌ Stats file not found: {stats_path}")
        return False
    
    # Load stats
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    print(f"Stats: HR@10={stats.get('hr10', 'N/A'):.4f}, NDCG@10={stats.get('ndcg10', 'N/A'):.4f}, Best epoch: {stats.get('epoch', 'N/A')}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check item embedding
        if 'item_emb.weight' in checkpoint:
            n_items = checkpoint['item_emb.weight'].shape[0] - 1  # -1 for padding idx
            emb_dim = checkpoint['item_emb.weight'].shape[1]
            print(f"Item embeddings: {n_items:,} items, {emb_dim} dimensions")
        else:
            print("❌ No item embeddings found in checkpoint")
            return False
        
        # Compute SHA-256
        with open(ckpt_path, "rb") as f:
            ckpt_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"Checkpoint SHA-256: {ckpt_hash}")
        
        # Check if hash is in manifest
        manifest_path = proj_dir / "data" / "_checksums" / "ranker_manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest_entries = [json.loads(line) for line in f if line.strip()]
            
            dataset_entries = [entry for entry in manifest_entries if entry.get("dataset") == dataset]
            if dataset_entries:
                manifest_hash = dataset_entries[-1].get("sha256")
                if manifest_hash == ckpt_hash:
                    print(f"✓ Checkpoint hash matches manifest entry")
                else:
                    print(f"❌ Checkpoint hash mismatch with manifest!")
                    print(f"  Manifest: {manifest_hash}")
                    print(f"  Computed: {ckpt_hash}")
            else:
                print("⚠️ No manifest entry found for this dataset")
        else:
            print("⚠️ No manifest file found")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate SASRec checkpoints")
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam", "all"],
                        default="all", help="Dataset to validate")
    args = parser.parse_args()
    
    proj_dir = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[2]))
    
    datasets = ["books", "ml25m", "steam"] if args.dataset == "all" else [args.dataset]
    all_valid = True
    
    for dataset in datasets:
        if not validate_checkpoint(dataset, proj_dir):
            all_valid = False
    
    if all_valid:
        print("\n✓ All checkpoints validated successfully")
    else:
        print("\n❌ Some checkpoints failed validation")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
