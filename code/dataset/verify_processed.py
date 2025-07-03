#!/usr/bin/env python
"""
verify_processed.py: Verify the integrity of processed dataset files
by checking their SHA-256 checksums against recorded values.

Usage:
  python verify_processed.py [--dataset books|ml25m|steam|all]
"""

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path


def verify_checksum(file_path, recorded_hash):
    """Verify the SHA-256 checksum of a file against a recorded value."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    calculated = h.hexdigest()
    return calculated == recorded_hash, calculated


def verify_dataset(dataset, root_dir):
    """Verify all processed files for a specific dataset."""
    print(f"\nVerifying {dataset} processed files...")
    checksum_dir = root_dir / "data" / "_checksums"
    proc_dir = root_dir / "data" / "proc" / dataset
    
    # Get all checksum files for this dataset
    checksum_files = list(checksum_dir.glob(f"{dataset}_*.sha256"))
    if not checksum_files:
        print(f"No checksum files found for {dataset}")
        return False
    
    all_valid = True
    for checksum_file in checksum_files:
        with open(checksum_file, "r") as f:
            content = f.read().strip()
            recorded_hash, filename = content.split()
        
        file_path = proc_dir / filename
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            all_valid = False
            continue
        
        valid, calculated = verify_checksum(file_path, recorded_hash)
        if valid:
            print(f"✓ {filename} verified")
        else:
            print(f"❌ {filename} MISMATCH!")
            print(f"  Expected: {recorded_hash}")
            print(f"  Got:      {calculated}")
            all_valid = False
    
    # Check if stats.json exists and print some info
    stats_file = proc_dir / "stats.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
        print(f"\nDataset stats for {dataset}:")
        print(f"  Raw rows:     {stats.get('rows_raw', 'N/A'):,}")
        print(f"  Processed:    {stats.get('rows_proc', 'N/A'):,}")
        print(f"  Users:        {stats.get('n_users', 'N/A'):,}")
        print(f"  Items:        {stats.get('n_items', 'N/A'):,}")
        print(f"  Train split:  {stats.get('train', 'N/A'):,}")
        print(f"  Valid split:  {stats.get('valid', 'N/A'):,}")
        print(f"  Test split:   {stats.get('test', 'N/A'):,}")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Verify processed dataset files")
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam", "all"],
                        default="all", help="Dataset to verify")
    args = parser.parse_args()
    
    root_dir = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[2]))
    
    datasets = ["books", "ml25m", "steam"] if args.dataset == "all" else [args.dataset]
    all_valid = True
    
    for dataset in datasets:
        if not verify_dataset(dataset, root_dir):
            all_valid = False
    
    if all_valid:
        print("\n✓ All processed files match recorded SHA-256 checksums")
    else:
        print("\n❌ Some files failed verification!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
