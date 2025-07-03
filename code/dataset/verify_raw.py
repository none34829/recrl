#!/usr/bin/env python
# Dataset verification script

import json
import hashlib
import pathlib
import sys

def verify_datasets():
    """Verify that all raw datasets match their recorded checksums."""
    proj = pathlib.Path(__file__).resolve().parents[2]
    manifest = proj / "data" / "_checksums" / "manifest.jsonl"
    
    if not manifest.exists():
        print(f"Error: Manifest file not found at {manifest}")
        return False
    
    success = True
    for line in manifest.open():
        meta = json.loads(line)
        dataset_name = meta["dataset"]
        
        # Determine the directory name based on the dataset name
        if "amazon" in dataset_name or "books" in dataset_name:
            dir_name = "books"
        elif "ml" in dataset_name:
            dir_name = "ml25m"
        elif "steam" in dataset_name:
            dir_name = "steam"
        else:
            print(f"Unknown dataset type: {dataset_name}")
            success = False
            continue
        
        # Find the file
        raw_dir = proj / "data" / "raw" / dir_name
        if not raw_dir.exists():
            print(f"Error: Directory not found: {raw_dir}")
            success = False
            continue
        
        # Look for files matching the dataset pattern
        pattern = "*" + dataset_name.split('_')[-1] + "*"
        matching_files = list(raw_dir.glob(pattern))
        
        if not matching_files:
            print(f"Error: No files matching '{pattern}' found in {raw_dir}")
            success = False
            continue
        
        # Use the first matching file
        path = matching_files[0]
        print(f"Verifying {path}...")
        
        try:
            # Calculate SHA-256 hash
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            
            # Compare with recorded hash
            if sha == meta["sha256"]:
                print(f"✓ {dataset_name}: Hash verified")
            else:
                print(f"✗ {dataset_name}: Hash mismatch!")
                print(f"  Expected: {meta['sha256']}")
                print(f"  Actual:   {sha}")
                success = False
        except Exception as e:
            print(f"Error verifying {path}: {str(e)}")
            success = False
    
    return success

if __name__ == "__main__":
    if verify_datasets():
        print("\nAll raw datasets verified OK")
        sys.exit(0)
    else:
        print("\nVerification failed for one or more datasets")
        sys.exit(1)
