#!/usr/bin/env python
"""
check_processed_stats.py: Display statistics for processed datasets

Usage:
  python check_processed_stats.py [--dataset books|ml25m|steam|all]
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd

def format_number(n):
    """Format a number with commas as thousands separators."""
    return f"{n:,}" if n is not None else "N/A"

def check_dataset(dataset, root_dir):
    """Check statistics for a specific dataset."""
    proc_dir = root_dir / "data" / "proc" / dataset
    
    if not proc_dir.exists():
        print(f"❌ Processed directory not found for {dataset}")
        return False
    
    # Check stats.json
    stats_file = proc_dir / "stats.json"
    if not stats_file.exists():
        print(f"❌ stats.json not found for {dataset}")
        return False
    
    with open(stats_file, "r") as f:
        stats = json.load(f)
    
    print(f"\n=== {dataset.upper()} Dataset Statistics ===")
    print(f"Raw rows:       {format_number(stats.get('rows_raw'))}")
    print(f"Processed rows: {format_number(stats.get('rows_proc'))}")
    print(f"Users:          {format_number(stats.get('n_users'))}")
    print(f"Items:          {format_number(stats.get('n_items'))}")
    
    # Calculate density
    n_users = stats.get('n_users')
    n_items = stats.get('n_items')
    n_interactions = stats.get('rows_proc')
    
    if n_users and n_items and n_interactions:
        density = (n_interactions / (n_users * n_items)) * 100
        print(f"Density:        {density:.6f}%")
    
    print("\nSplit Statistics:")
    print(f"  Train:        {format_number(stats.get('train'))}")
    print(f"  Validation:   {format_number(stats.get('valid'))}")
    print(f"  Test:         {format_number(stats.get('test'))}")
    
    # Check if timestamp range is available
    ts_min = stats.get('ts_min')
    ts_max = stats.get('ts_max')
    if ts_min and ts_max:
        from datetime import datetime
        print("\nTimestamp Range:")
        print(f"  Start: {datetime.fromtimestamp(ts_min).strftime('%Y-%m-%d')}")
        print(f"  End:   {datetime.fromtimestamp(ts_max).strftime('%Y-%m-%d')}")
    
    # Check Parquet files
    for split in ["train", "valid", "test"]:
        parquet_file = proc_dir / f"{split}.parquet"
        if not parquet_file.exists():
            print(f"❌ {split}.parquet not found")
            return False
        
        # Get row count from Parquet metadata (fast)
        try:
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(parquet_file)
            row_count = metadata.num_rows
            print(f"✓ {split}.parquet: {format_number(row_count)} rows")
        except Exception as e:
            print(f"❌ Error reading {split}.parquet: {e}")
            return False
    
    # Check id_maps.json
    id_maps_file = proc_dir / "id_maps.json"
    if not id_maps_file.exists():
        print(f"❌ id_maps.json not found")
        return False
    
    print("✓ id_maps.json found")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Check processed dataset statistics")
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam", "all"],
                        default="all", help="Dataset to check")
    args = parser.parse_args()
    
    root_dir = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[2]))
    
    datasets = ["books", "ml25m", "steam"] if args.dataset == "all" else [args.dataset]
    all_valid = True
    
    for dataset in datasets:
        if not check_dataset(dataset, root_dir):
            all_valid = False
    
    # Print disk usage summary
    print("\n=== Disk Usage Summary ===")
    proc_dir = root_dir / "data" / "proc"
    if proc_dir.exists():
        import subprocess
        try:
            subprocess.run(["du", "-sh", str(proc_dir / "*")], check=False)
        except Exception:
            # Fall back to Python's os.path.getsize if du is not available
            for dataset in datasets:
                dataset_dir = proc_dir / dataset
                if dataset_dir.exists():
                    total_size = sum(f.stat().st_size for f in dataset_dir.glob('**/*') if f.is_file())
                    print(f"{dataset}: {total_size / (1024*1024):.2f} MB")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    exit(main())
