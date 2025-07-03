#!/bin/bash
# Dataset download script for Shielded RecRL

# Set project directory
if [ -z "$PROJ" ]; then
    export PROJ=$(pwd)
    echo "PROJ not set, using current directory: $PROJ"
fi

# Create directories
mkdir -p $PROJ/data/raw/{books,ml25m,steam} $PROJ/data/_checksums

# Install dependencies
echo "=== Installing dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y aria2 jq

# Function to download Amazon Books dataset
download_books() {
    echo "\n=== Downloading Amazon Books 5-core dataset ==="
    cd $PROJ/data/raw/books
    aria2c -x 8 -s 8 -c https://deeprec-repl.s3.us-west-2.amazonaws.com/amazon_books_5.json.gz \
           -o books.json.gz
    
    # Checksum and manifest
    sha256sum books.json.gz | tee $PROJ/data/_checksums/books.sha256
    jq -n --arg name "amazon_books_5" \
          --arg date "$(date -Iseconds)" \
          --arg sha "$(cut -d' ' -f1 $PROJ/data/_checksums/books.sha256)" \
          --arg bytes "$(stat -c%s books.json.gz)" \
          '{dataset:$name,date:$date,sha256:$sha,size_bytes:$bytes}' \
          >> $PROJ/data/_checksums/manifest.jsonl
    
    # Sanity check
    echo "\nSanity check:"
    gunzip -c books.json.gz | head -n 2 | jq .
}

# Function to download MovieLens dataset
download_movielens() {
    echo "\n=== Downloading MovieLens-25M dataset ==="
    cd $PROJ/data/raw/ml25m
    aria2c -x 8 -s 8 -c https://files.grouplens.org/datasets/movielens/ml-25m.zip
    
    # Checksum and manifest
    sha256sum ml-25m.zip | tee $PROJ/data/_checksums/ml25m.sha256
    jq -n --arg name "ml25m" --arg date "$(date -Iseconds)" \
          --arg sha "$(cut -d' ' -f1 $PROJ/data/_checksums/ml25m.sha256)" \
          --arg bytes "$(stat -c%s ml-25m.zip)" \
          '{dataset:$name,date:$date,sha256:$sha,size_bytes:$bytes}' \
          >> $PROJ/data/_checksums/manifest.jsonl
    
    # Extract to temp folder
    echo "\nExtracting MovieLens dataset..."
    unzip -q ml-25m.zip
    mkdir -p $PROJ/data/raw/tmp_ml25m
    mv ml-25m $PROJ/data/raw/tmp_ml25m
}

# Function to download Steam dataset
download_steam() {
    echo "\n=== Downloading Steam-200K dataset ==="
    cd $PROJ/data/raw/steam
    aria2c -x 8 -s 8 -c https://storage.googleapis.com/rec-sys-public-data/steam-200k.csv
    
    # Checksum and manifest
    sha256sum steam-200k.csv | tee $PROJ/data/_checksums/steam.sha256
    jq -n --arg name "steam200k" --arg date "$(date -Iseconds)" \
          --arg sha "$(cut -d' ' -f1 $PROJ/data/_checksums/steam.sha256)" \
          --arg bytes "$(stat -c%s steam-200k.csv)" \
          '{dataset:$name,date:$date,sha256:$sha,size_bytes:$bytes}' \
          >> $PROJ/data/_checksums/manifest.jsonl
    
    # Row count sanity check
    echo "\nSanity check:"
    wc -l steam-200k.csv
    head -n 3 steam-200k.csv
}

# Download all datasets
if [ "$1" == "books" ] || [ -z "$1" ]; then
    download_books
fi

if [ "$1" == "ml25m" ] || [ -z "$1" ]; then
    download_movielens
fi

if [ "$1" == "steam" ] || [ -z "$1" ]; then
    download_steam
fi

# Verify datasets
echo "\n=== Verifying datasets ==="
python $PROJ/code/dataset/verify_raw.py

# Print summary
echo "\n=== Dataset Download Summary ==="
du -sh $PROJ/data/raw/*
df -h /workspace

echo "\nDataset acquisition complete!"
