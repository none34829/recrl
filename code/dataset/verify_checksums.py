#!/usr/bin/env python
"""
verify_checksums.py: Quick script to verify all processed dataset checksums

Usage:
  python verify_checksums.py
"""

import glob
import hashlib
import pathlib
import json
import os
import sys

def main():
    root = pathlib.Path(os.getenv('PROJ', pathlib.Path(__file__).resolve().parents[2]))
    
    # Find all checksum files
    checksum_files = glob.glob(str(root/"data/_checksums/*.sha256"))
    if not checksum_files:
        print("No checksum files found!")
        return 1
    
    print(f"Found {len(checksum_files)} checksum files to verify")
    
    # Verify each checksum
    all_valid = True
    for sha_file in checksum_files:
        with open(sha_file) as f:
            content = f.read().strip()
            if not content:
                print(f"Empty checksum file: {sha_file}")
                all_valid = False
                continue
                
            try:
                recorded, fname = content.split()
            except ValueError:
                print(f"Invalid format in {sha_file}: {content}")
                all_valid = False
                continue
        
        # Determine the dataset from the filename
        sha_basename = os.path.basename(sha_file)
        if '_' in sha_basename:
            dataset = sha_basename.split('_')[0]
            file_path = (root/"data/proc") / dataset / fname
        else:
            # Raw dataset checksum
            dataset = sha_basename.split('.')[0]
            file_path = (root/"data/raw") / dataset / fname
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            all_valid = False
            continue
        
        # Calculate checksum
        calc = hashlib.sha256(open(file_path,'rb').read()).hexdigest()
        
        if calc == recorded:
            print(f"✓ {file_path.name}")
        else:
            print(f"❌ Mismatch for {file_path}")
            print(f"  Expected: {recorded}")
            print(f"  Got:      {calc}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All files match recorded SHA-256 checksums")
        return 0
    else:
        print("\n❌ Some files failed verification!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
