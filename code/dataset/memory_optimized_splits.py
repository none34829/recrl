#!/usr/bin/env python
"""
memory_optimized_splits.py : Memory-optimized version of make_splits.py
for handling large datasets that may cause OOM errors.

Usage:
  python memory_optimized_splits.py --dataset books --seed 42
"""
import argparse
import json
import os
import hashlib
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]   # project root

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "proc"
OUT.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ----------------------------------------------------------------------
def load_books() -> pd.DataFrame:
    """Memory-optimized loader for Amazon Books dataset."""
    import gzip
    import json
    fp = RAW / "books" / "books.json.gz"
    
    # Process in chunks to avoid OOM
    chunks = []
    with gzip.open(fp, "rt") as f:
        chunk = []
        for i, line in enumerate(f):
            if i > 0 and i % 100_000 == 0:
                print(f"Processed {i:,} lines")
                record = json.loads(line)
                chunk.append({
                    "user": record["reviewerID"],
                    "item": record["asin"],
                    "ts": record["unixReviewTime"]
                })
                chunks.append(pd.DataFrame(chunk))
                chunk = []
            else:
                record = json.loads(line)
                chunk.append({
                    "user": record["reviewerID"],
                    "item": record["asin"],
                    "ts": record["unixReviewTime"]
                })
        
        # Don't forget the last chunk
        if chunk:
            chunks.append(pd.DataFrame(chunk))
    
    return pd.concat(chunks, ignore_index=True)

def load_ml25m() -> pd.DataFrame:
    """Memory-optimized loader for MovieLens-25M dataset."""
    ratings = RAW / "ml25m" / "../tmp_ml25m" / "ratings.csv"
    
    # Use pyarrow engine for faster CSV parsing
    df = pd.read_csv(ratings, dtype={"userId": str, "movieId": str}, engine='pyarrow')
    df = df[df["rating"] >= 3.0]
    df = df.rename(columns={"userId":"user","movieId":"item","timestamp":"ts"})
    return df[["user","item","ts"]]

def load_steam() -> pd.DataFrame:
    """Memory-optimized loader for Steam dataset."""
    fp = RAW / "steam" / "steam-200k.csv"
    df = pd.read_csv(fp, header=None,
                     names=["user","item","action","value","ts"],
                     engine='pyarrow')
    df = df[df["action"] == "purchase"]
    df = df[["user","item","ts"]]
    return df

LOADERS = {"books": load_books, "ml25m": load_ml25m, "steam": load_steam}

# ----------------------------------------------------------------------
def filter_k(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Keep users & items with at least k interactions, iteratively."""
    while True:
        before = len(df)
        user_cnt = df["user"].value_counts()
        item_cnt = df["item"].value_counts()
        df = df[df["user"].map(user_cnt) >= k]
        df = df[df["item"].map(item_cnt) >= k]
        if len(df) == before:
            return df

def chrono_split(df, seed=42):
    """80/10/10 per user, chronologically."""
    train, valid, test = [], [], []
    rng = np.random.default_rng(seed)
    for _, grp in tqdm(df.groupby("user", sort=False)):
        grp = grp.sort_values("ts")
        n = len(grp)
        if n < 3:       # should not happen after filter_k
            continue
        t_train = int(0.8 * n)
        t_valid = int(0.9 * n)
        train.append(grp.iloc[:t_train])
        valid.append(grp.iloc[t_train:t_valid])
        test.append(grp.iloc[t_valid:])
    return pd.concat(train), pd.concat(valid), pd.concat(test)

def encode_ids(train, valid, test):
    u2i = {u:i for i,u in enumerate(train["user"].unique())}
    it2i = {it:i for i,it in enumerate(train["item"].unique())}
    def map_df(d):
        return pd.DataFrame({
            "user": d["user"].map(u2i),
            "item": d["item"].map(it2i),
            "ts"  : d["ts"].astype(np.int64)
        })
    return map_df(train), map_df(valid), map_df(test), u2i, it2i

# ----------------------------------------------------------------------
def write_parquet_chunked(df, path, chunk_size=250_000):
    """Write a DataFrame to Parquet in chunks to avoid OOM."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, chunk_size=chunk_size)

def main(args):
    loader = LOADERS[args.dataset]
    print(f"Loading {args.dataset} dataset...")
    raw_df = loader()
    print(f"Raw rows: {len(raw_df):,}")

    print("Applying 5-core filter...")
    df = filter_k(raw_df, 5)
    print(f"After 5-core filter: {len(df):,}")

    print("Splitting data chronologically...")
    train, valid, test = chrono_split(df, args.seed)
    print("Encoding IDs...")
    train, valid, test, u2i, it2i = encode_ids(train, valid, test)

    ddir = OUT / args.dataset
    ddir.mkdir(parents=True, exist_ok=True)
    
    print("Writing train split...")
    write_parquet_chunked(train, ddir/"train.parquet")
    print("Writing validation split...")
    write_parquet_chunked(valid, ddir/"valid.parquet")
    print("Writing test split...")
    write_parquet_chunked(test, ddir/"test.parquet")

    print("Writing ID maps...")
    with open(ddir/"id_maps.json", "w") as f:
        json.dump({"user2id": u2i, "item2id": it2i}, f)

    stats = {
      "rows_raw": len(raw_df),
      "rows_proc": len(df),
      "n_users": len(u2i),
      "n_items": len(it2i),
      "train": len(train), "valid": len(valid), "test": len(test),
      "ts_min": int(df["ts"].min()), "ts_max": int(df["ts"].max())
    }
    with open(ddir/"stats.json","w") as f: json.dump(stats,f,indent=2)

    # checksums
    for fn in ["train.parquet","valid.parquet","test.parquet",
               "id_maps.json","stats.json"]:
        p = ddir/fn
        hash = sha256(p)
        with open(ROOT/"data/_checksums"/f"{args.dataset}_{fn}.sha256","w") as f:
            f.write(f"{hash}  {p.name}\n")
    print("Done!  hashes written.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=LOADERS.keys())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--memory-optimized", action="store_true", 
                   help="Use memory-optimized processing")
    args = p.parse_args()
    main(args)
