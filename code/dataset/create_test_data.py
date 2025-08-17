#!/usr/bin/env python
# Create test data for Shielded RecRL experiments

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_books_test_data(proj_dir):
    """Create a small test dataset for Amazon Books."""
    output_dir = Path(proj_dir) / "data" / "raw" / "books"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "books.json.gz"
    
    # Create sample data
    data = []
    for i in range(1000):  # 1000 interactions
        data.append({
            "reviewerID": f"user_{i % 100}",  # 100 users
            "asin": f"book_{i % 200}",  # 200 books
            "overall": np.random.randint(1, 6),
            "unixReviewTime": 1609459200 + i * 86400,  # Sequential timestamps
            "reviewText": f"This is a review for book {i % 200}",
            "summary": f"Summary for book {i % 200}"
        })
    
    # Save as JSON lines
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created test books dataset: {output_file}")
    print(f"Records: {len(data)}")
    print(f"Users: {len(set(item['reviewerID'] for item in data))}")
    print(f"Books: {len(set(item['asin'] for item in data))}")


def create_movielens_test_data(proj_dir):
    """Create a small test dataset for MovieLens."""
    output_dir = Path(proj_dir) / "data" / "raw" / "ml25m"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ratings.csv"
    
    # Create sample data
    data = []
    for i in range(1000):  # 1000 ratings
        data.append({
            "userId": i % 100,  # 100 users
            "movieId": i % 200,  # 200 movies
            "rating": np.random.randint(1, 6),
            "timestamp": 1609459200 + i * 86400  # Sequential timestamps
        })
    
    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"Created test MovieLens dataset: {output_file}")
    print(f"Records: {len(data)}")
    print(f"Users: {len(set(item['userId'] for item in data))}")
    print(f"Movies: {len(set(item['movieId'] for item in data))}")


def create_steam_test_data(proj_dir):
    """Create a small test dataset for Steam."""
    output_dir = Path(proj_dir) / "data" / "raw" / "steam"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "steam-200k.csv"
    
    # Create sample data
    data = []
    for i in range(1000):  # 1000 interactions
        data.append({
            "user_id": i % 100,  # 100 users
            "game_id": i % 200,  # 200 games
            "playtime_forever": np.random.randint(1, 1000),
            "playtime_2weeks": np.random.randint(0, 100),
            "timestamp": 1609459200 + i * 86400  # Sequential timestamps
        })
    
    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"Created test Steam dataset: {output_file}")
    print(f"Records: {len(data)}")
    print(f"Users: {len(set(item['user_id'] for item in data))}")
    print(f"Games: {len(set(item['game_id'] for item in data))}")


def main():
    parser = argparse.ArgumentParser(description="Create test datasets for Shielded RecRL")
    parser.add_argument("--dataset", required=True, choices=["books", "ml25m", "steam", "all"],
                        help="Dataset to create")
    parser.add_argument("--proj_dir", default="..",
                        help="Project directory (default: parent directory)")
    args = parser.parse_args()
    
    proj_dir = Path(args.proj_dir).resolve()
    print(f"Project directory: {proj_dir}")
    
    if args.dataset == "books" or args.dataset == "all":
        create_books_test_data(proj_dir)
    
    if args.dataset == "ml25m" or args.dataset == "all":
        create_movielens_test_data(proj_dir)
    
    if args.dataset == "steam" or args.dataset == "all":
        create_steam_test_data(proj_dir)
    
    print("\n=== Test Data Creation Summary ===")
    print("All requested test datasets have been created.")
    print("Next step: Run preprocessing with python preprocess.py --dataset <dataset>")


if __name__ == "__main__":
    main()

