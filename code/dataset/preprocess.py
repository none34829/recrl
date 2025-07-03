#!/usr/bin/env python
# Dataset preprocessing script

import json
import argparse
import pandas as pd
import numpy as np
import os
import random
import hashlib
import tqdm
import csv
from pathlib import Path


def books(src: Path, dest: Path):
    """Process Amazon Books dataset.
    
    Args:
        src: Path to the source JSON file
        dest: Path to the destination CSV file
    """
    print(f"Processing Amazon Books dataset from {src}")
    df = pd.read_json(src, lines=True)
    
    # Filter users and items with at least 5 interactions
    print("Filtering users and items with at least 5 interactions...")
    df = df.groupby('reviewerID').filter(lambda x: len(x) >= 5)
    df = df.groupby('asin').filter(lambda x: len(x) >= 5)
    
    # Sort by timestamp
    df = df.sort_values('unixReviewTime')
    
    # Create user and item mappings
    users = {u: i for i, u in enumerate(df.reviewerID.unique())}
    items = {a: i for i, a in enumerate(df.asin.unique())}
    
    print(f"Found {len(users)} users and {len(items)} items")
    
    # Write to CSV
    print(f"Writing to {dest}...")
    with open(dest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['user', 'item', 'ts'])
        for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
            w.writerow([users[r.reviewerID], items[r.asin], r.unixReviewTime])
    
    print(f"Done. Wrote {len(df)} interactions to {dest}")


def movielens(src_dir: Path, dest: Path):
    """Process MovieLens-25M dataset.
    
    Args:
        src_dir: Path to the source directory containing ratings.csv
        dest: Path to the destination CSV file
    """
    ratings_file = src_dir / "ratings.csv"
    print(f"Processing MovieLens dataset from {ratings_file}")
    
    # Read ratings
    df = pd.read_csv(ratings_file)
    
    # Filter users and items with at least 5 interactions
    print("Filtering users and items with at least 5 interactions...")
    df = df.groupby('userId').filter(lambda x: len(x) >= 5)
    df = df.groupby('movieId').filter(lambda x: len(x) >= 5)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create user and item mappings
    users = {u: i for i, u in enumerate(df.userId.unique())}
    items = {a: i for i, a in enumerate(df.movieId.unique())}
    
    print(f"Found {len(users)} users and {len(items)} items")
    
    # Write to CSV
    print(f"Writing to {dest}...")
    with open(dest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['user', 'item', 'ts'])
        for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
            w.writerow([users[r.userId], items[r.movieId], r.timestamp])
    
    print(f"Done. Wrote {len(df)} interactions to {dest}")


def steam(src: Path, dest: Path):
    """Process Steam-200K dataset.
    
    Args:
        src: Path to the source JSON file
        dest: Path to the destination CSV file
    """
    print(f"Processing Steam dataset from {src}")
    df = pd.read_json(src, lines=True)
    
    # Filter users and items with at least 5 interactions
    print("Filtering users and items with at least 5 interactions...")
    df = df.groupby('user_id').filter(lambda x: len(x) >= 5)
    df = df.groupby('game_id').filter(lambda x: len(x) >= 5)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create user and item mappings
    users = {u: i for i, u in enumerate(df.user_id.unique())}
    items = {a: i for i, a in enumerate(df.game_id.unique())}
    
    print(f"Found {len(users)} users and {len(items)} items")
    
    # Write to CSV
    print(f"Writing to {dest}...")
    with open(dest, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['user', 'item', 'ts'])
        for _, r in tqdm.tqdm(df.iterrows(), total=len(df)):
            w.writerow([users[r.user_id], items[r.game_id], r.timestamp])
    
    print(f"Done. Wrote {len(df)} interactions to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess recommendation datasets")
    parser.add_argument('--name', required=True, choices=['books', 'ml', 'steam'],
                        help='Dataset name to process')
    parser.add_argument('--data_dir', default=os.path.join(os.environ.get('PROJ', '.'), 'data'),
                        help='Directory containing the raw data files')
    parser.add_argument('--output_dir', default=os.path.join(os.environ.get('PROJ', '.'), 'data'),
                        help='Directory to save the processed data')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    out_file = output_dir / f"{args.name}.csv"
    
    if args.name == 'books':
        books(data_dir / "amazon-books-5core.json", out_file)
    elif args.name == 'ml':
        movielens(data_dir / "ml-25m", out_file)
    elif args.name == 'steam':
        steam(data_dir / "steam-200k.json", out_file)
    else:
        raise ValueError(f"Unknown dataset: {args.name}")


if __name__ == "__main__":
    main()
