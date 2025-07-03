#!/usr/bin/env python
# Create raw data directories

import os
import argparse
from pathlib import Path

def create_directories(proj_dir):
    """Create the necessary directories for raw data."""
    dirs = [
        Path(proj_dir) / "data" / "raw" / "books",
        Path(proj_dir) / "data" / "raw" / "ml25m",
        Path(proj_dir) / "data" / "raw" / "steam",
        Path(proj_dir) / "data" / "_checksums"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {d}")
        
        # Create .keep file to maintain directory structure in git
        keep_file = d / ".keep"
        if not keep_file.exists():
            keep_file.touch()
            print(f"Created .keep file in {d}")

def main():
    parser = argparse.ArgumentParser(description="Create raw data directories")
    parser.add_argument('--proj_dir', default=os.environ.get('PROJ', '.'),
                        help='Project directory')
    args = parser.parse_args()
    
    create_directories(args.proj_dir)
    print("\nDirectory structure created successfully!")

if __name__ == "__main__":
    main()
