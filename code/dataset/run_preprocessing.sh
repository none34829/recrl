#!/bin/bash
# Run preprocessing for all datasets

# Set project directory
if [ -z "$PROJ" ]; then
    export PROJ=$(pwd)
    echo "PROJ not set, using current directory: $PROJ"
fi

# Ensure conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "rec" ]; then
    echo "Please activate the 'rec' conda environment first:"
    echo "conda activate rec"
    exit 1
fi

# Create proc directory if it doesn't exist
mkdir -p $PROJ/data/proc

# Process each dataset
echo "=== Starting dataset preprocessing ==="
for dataset in books ml25m steam; do
    echo "\n=== Processing $dataset dataset ==="
    python $PROJ/code/dataset/scripts/make_splits.py --dataset $dataset --seed 42
    
    # Check if processing was successful
    if [ $? -ne 0 ]; then
        echo "❌ Error processing $dataset dataset"
        exit 1
    fi
done

# Verify all processed files
echo "\n=== Verifying all processed files ==="
python $PROJ/code/dataset/verify_processed.py --dataset all

# Print disk usage summary
echo "\n=== Disk usage summary ==="
du -sh $PROJ/data/proc/*
df -h /workspace

echo "\n✓ Preprocessing complete! All datasets are ready for model training."
