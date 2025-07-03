#!/usr/bin/env python
# Dataset download script

import os
import sys
import subprocess
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd, cwd=None):
    """Run a shell command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = ['aria2c', 'jq']
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run(['which', dep], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install them with:")
        print("sudo apt-get update -qq && sudo apt-get install -y aria2 jq")
        sys.exit(1)


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


def download_amazon_books(proj_dir):
    """Download Amazon Books 5-core dataset."""
    output_dir = Path(proj_dir) / "data" / "raw" / "books"
    output_file = output_dir / "books.json.gz"
    checksum_file = Path(proj_dir) / "data" / "_checksums" / "books.sha256"
    manifest_file = Path(proj_dir) / "data" / "_checksums" / "manifest.jsonl"
    
    # Download the file
    print("\n=== Downloading Amazon Books 5-core dataset ===")
    run_command([
        "aria2c", "-x", "8", "-s", "8", "-c",
        "https://deeprec-repl.s3.us-west-2.amazonaws.com/amazon_books_5.json.gz",
        "-o", "books.json.gz"
    ], cwd=output_dir)
    
    # Calculate checksum
    sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    with open(checksum_file, 'w') as f:
        f.write(f"{sha256}  books.json.gz\n")
    
    # Get file size
    size_bytes = output_file.stat().st_size
    
    # Create manifest entry
    manifest_entry = {
        "dataset": "amazon_books_5",
        "date": datetime.now().isoformat(),
        "sha256": sha256,
        "size_bytes": str(size_bytes)
    }
    
    with open(manifest_file, 'a') as f:
        f.write(json.dumps(manifest_entry) + "\n")
    
    # Sanity check
    print("\nSanity check:")
    run_command(["gunzip", "-c", "books.json.gz", "|", "head", "-n", "2"], cwd=output_dir)
    
    print(f"\nAmazon Books dataset downloaded to {output_file}")
    print(f"SHA-256: {sha256}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")


def download_movielens(proj_dir):
    """Download MovieLens-25M dataset."""
    output_dir = Path(proj_dir) / "data" / "raw" / "ml25m"
    output_file = output_dir / "ml-25m.zip"
    checksum_file = Path(proj_dir) / "data" / "_checksums" / "ml25m.sha256"
    manifest_file = Path(proj_dir) / "data" / "_checksums" / "manifest.jsonl"
    
    # Download the file
    print("\n=== Downloading MovieLens-25M dataset ===")
    run_command([
        "aria2c", "-x", "8", "-s", "8", "-c",
        "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    ], cwd=output_dir)
    
    # Calculate checksum
    sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    with open(checksum_file, 'w') as f:
        f.write(f"{sha256}  ml-25m.zip\n")
    
    # Get file size
    size_bytes = output_file.stat().st_size
    
    # Create manifest entry
    manifest_entry = {
        "dataset": "ml25m",
        "date": datetime.now().isoformat(),
        "sha256": sha256,
        "size_bytes": str(size_bytes)
    }
    
    with open(manifest_file, 'a') as f:
        f.write(json.dumps(manifest_entry) + "\n")
    
    # Extract to temp folder
    tmp_dir = Path(proj_dir) / "data" / "raw" / "tmp_ml25m"
    tmp_dir.mkdir(exist_ok=True)
    
    print("\nExtracting MovieLens dataset...")
    run_command(["unzip", "-q", "ml-25m.zip"], cwd=output_dir)
    run_command(["mv", "ml-25m", str(tmp_dir)], cwd=output_dir)
    
    print(f"\nMovieLens dataset downloaded to {output_file}")
    print(f"SHA-256: {sha256}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")
    print(f"Extracted to: {tmp_dir}")


def download_steam(proj_dir):
    """Download Steam-200K dataset."""
    output_dir = Path(proj_dir) / "data" / "raw" / "steam"
    output_file = output_dir / "steam-200k.csv"
    checksum_file = Path(proj_dir) / "data" / "_checksums" / "steam.sha256"
    manifest_file = Path(proj_dir) / "data" / "_checksums" / "manifest.jsonl"
    
    # Download the file
    print("\n=== Downloading Steam-200K dataset ===")
    run_command([
        "aria2c", "-x", "8", "-s", "8", "-c",
        "https://storage.googleapis.com/rec-sys-public-data/steam-200k.csv"
    ], cwd=output_dir)
    
    # Calculate checksum
    sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    with open(checksum_file, 'w') as f:
        f.write(f"{sha256}  steam-200k.csv\n")
    
    # Get file size
    size_bytes = output_file.stat().st_size
    
    # Create manifest entry
    manifest_entry = {
        "dataset": "steam200k",
        "date": datetime.now().isoformat(),
        "sha256": sha256,
        "size_bytes": str(size_bytes)
    }
    
    with open(manifest_file, 'a') as f:
        f.write(json.dumps(manifest_entry) + "\n")
    
    # Sanity check
    print("\nSanity check:")
    run_command(["wc", "-l", "steam-200k.csv"], cwd=output_dir)
    run_command(["head", "-n", "3", "steam-200k.csv"], cwd=output_dir)
    
    print(f"\nSteam dataset downloaded to {output_file}")
    print(f"SHA-256: {sha256}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Shielded RecRL")
    parser.add_argument('--proj_dir', default=os.environ.get('PROJ', '.'),
                        help='Project directory')
    parser.add_argument('--dataset', choices=['all', 'books', 'ml25m', 'steam'],
                        default='all', help='Dataset to download')
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Create directories
    create_directories(args.proj_dir)
    
    # Download datasets
    if args.dataset in ['all', 'books']:
        download_amazon_books(args.proj_dir)
    
    if args.dataset in ['all', 'ml25m']:
        download_movielens(args.proj_dir)
    
    if args.dataset in ['all', 'steam']:
        download_steam(args.proj_dir)
    
    # Verify datasets
    print("\n=== Verifying datasets ===")
    verify_script = Path(args.proj_dir) / "code" / "dataset" / "verify_raw.py"
    run_command(["python", str(verify_script)])
    
    # Print summary
    print("\n=== Dataset Download Summary ===")
    run_command(["du", "-sh", "data/raw/*"], cwd=args.proj_dir)
    run_command(["df", "-h", "/workspace"], cwd=args.proj_dir)
    
    print("\nDataset acquisition complete!")


if __name__ == "__main__":
    main()
