#!/usr/bin/env python
"""
Create manifest entries for LoRA adapters.
"""
import json
import os
import argparse
from pathlib import Path
from datetime import datetime

def create_manifest_entries(datasets, proj_dir):
    """Create manifest entries for LoRA adapters for the specified datasets."""
    manifest_path = proj_dir / "data" / "_checksums" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets:
        meta_path = proj_dir / "checkpoints" / f"lora_init_{dataset}_meta.json"
        
        if not meta_path.exists():
            print(f"Warning: Metadata file not found for {dataset}")
            continue
        
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Create manifest entry
        manifest_entry = {
            "type": "lora_init",
            "dataset": dataset,
            "sha256": meta.get("sha256", ""),
            "date": datetime.now().isoformat()
        }
        
        # Append to manifest file
        with open(manifest_path, "a") as f:
            f.write(json.dumps(manifest_entry) + "\n")
        
        print(f"Added manifest entry for {dataset}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["books", "ml25m", "steam"],
                        help="Datasets to create manifest entries for")
    args = parser.parse_args()
    
    proj_dir = Path(os.getenv("PROJ", "."))
    create_manifest_entries(args.datasets, proj_dir)

if __name__ == "__main__":
    main()
