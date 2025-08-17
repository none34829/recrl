#!/usr/bin/env python
# Convert CSV to parquet and create train/validation splits

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def create_splits(csv_file, output_dir, train_ratio=0.8, seed=42):
    """Create train/validation splits from CSV file."""
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Sort by timestamp
    df = df.sort_values('ts')
    
    # Group by user
    user_groups = df.groupby('user')
    
    train_data = []
    valid_data = []
    
    for user, group in user_groups:
        n_interactions = len(group)
        split_idx = int(n_interactions * train_ratio)
        
        train_data.append(group.iloc[:split_idx])
        valid_data.append(group.iloc[split_idx:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    valid_df = pd.concat(valid_data, ignore_index=True)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    train_file = output_dir / "train.parquet"
    valid_file = output_dir / "valid.parquet"
    
    train_df.to_parquet(train_file, index=False)
    valid_df.to_parquet(valid_file, index=False)
    
    print(f"Train set: {len(train_df)} interactions")
    print(f"Validation set: {len(valid_df)} interactions")
    print(f"Saved to: {train_file} and {valid_file}")

def main():
    parser = argparse.ArgumentParser(description="Create train/validation splits")
    parser.add_argument("--csv_file", required=True, help="Input CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    create_splits(args.csv_file, args.output_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()

