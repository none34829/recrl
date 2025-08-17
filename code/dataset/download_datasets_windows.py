#!/usr/bin/env python
# Windows-compatible dataset download script

import os
import sys
import subprocess
import json
import hashlib
import argparse
import urllib.request
import gzip
import shutil
from pathlib import Path
from datetime import datetime


def download_file(url, output_path):
    """Download a file from URL to output path."""
    print(f"Downloading {url} to {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Download completed: {output_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def check_file_exists(file_path):
    """Check if a file exists."""
    return Path(file_path).exists()


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_amazon_books(proj_dir):
    """Download Amazon Books 5-core dataset."""
    output_dir = Path(proj_dir) / "data" / "raw" / "books"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "books.json.gz"
    checksum_file = Path(proj_dir) / "data" / "_checksums" / "books.sha256"
    manifest_file = Path(proj_dir) / "data" / "_checksums" / "manifest.jsonl"
    
    # Create checksums directory
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    print("\n=== Downloading Amazon Books 5-core dataset ===")
    url = "https://deeprec-repl.s3.us-west-2.amazonaws.com/amazon_books_5.json.gz"
    download_file(url, output_file)
    
    # Calculate checksum
    sha256 = calculate_sha256(output_file)
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
    
    print(f"\nAmazon Books dataset downloaded to {output_file}")
    print(f"SHA-256: {sha256}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")


def download_movielens(proj_dir):
    """Download MovieLens-25M dataset."""
    output_dir = Path(proj_dir) / "data" / "raw" / "ml25m"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ml-25m.zip"
    checksum_file = Path(proj_dir) / "data" / "_checksums" / "ml25m.sha256"
    manifest_file = Path(proj_dir) / "data" / "_checksums" / "manifest.jsonl"
    
    # Create checksums directory
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    print("\n=== Downloading MovieLens-25M dataset ===")
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    download_file(url, output_file)
    
    # Calculate checksum
    sha256 = calculate_sha256(output_file)
    with open(checksum_file, 'w') as f:
        f.write(f"{sha256}  ml-25m.zip\n")
    
    # Get file size
    size_bytes = output_file.stat().st_size
    
    # Create manifest entry
    manifest_entry = {
        "dataset": "movielens_25m",
        "date": datetime.now().isoformat(),
        "sha256": sha256,
        "size_bytes": str(size_bytes)
    }
    
    with open(manifest_file, 'a') as f:
        f.write(json.dumps(manifest_entry) + "\n")
    
    print(f"\nMovieLens-25M dataset downloaded to {output_file}")
    print(f"SHA-256: {sha256}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")


def download_steam(proj_dir):
    """Download Steam-200K dataset."""
    output_dir = Path(proj_dir) / "data" / "raw" / "steam"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "steam-200k.json.gz"
    checksum_file = Path(proj_dir) / "data" / "_checksums" / "steam.sha256"
    manifest_file = Path(proj_dir) / "data" / "_checksums" / "manifest.jsonl"
    
    # Create checksums directory
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    print("\n=== Downloading Steam-200K dataset ===")
    url = "https://deeprec-repl.s3.us-west-2.amazonaws.com/steam_200k.json.gz"
    download_file(url, output_file)
    
    # Calculate checksum
    sha256 = calculate_sha256(output_file)
    with open(checksum_file, 'w') as f:
        f.write(f"{sha256}  steam-200k.json.gz\n")
    
    # Get file size
    size_bytes = output_file.stat().st_size
    
    # Create manifest entry
    manifest_entry = {
        "dataset": "steam_200k",
        "date": datetime.now().isoformat(),
        "sha256": sha256,
        "size_bytes": str(size_bytes)
    }
    
    with open(manifest_file, 'a') as f:
        f.write(json.dumps(manifest_entry) + "\n")
    
    print(f"\nSteam-200K dataset downloaded to {output_file}")
    print(f"SHA-256: {sha256}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Shielded RecRL")
    parser.add_argument("--dataset", required=True, choices=["books", "ml25m", "steam", "all"],
                        help="Dataset to download")
    parser.add_argument("--proj_dir", default=".",
                        help="Project directory (default: current directory)")
    args = parser.parse_args()
    
    proj_dir = Path(args.proj_dir).resolve()
    print(f"Project directory: {proj_dir}")
    
    if args.dataset == "books" or args.dataset == "all":
        download_amazon_books(proj_dir)
    
    if args.dataset == "ml25m" or args.dataset == "all":
        download_movielens(proj_dir)
    
    if args.dataset == "steam" or args.dataset == "all":
        download_steam(proj_dir)
    
    print("\n=== Download Summary ===")
    print("All requested datasets have been downloaded.")
    print("Next step: Run preprocessing with python preprocess.py --dataset <dataset>")


if __name__ == "__main__":
    main()

