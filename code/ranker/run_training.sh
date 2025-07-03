#!/bin/bash
# Run SASRec training for all datasets

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

# Create checkpoints directory if it doesn't exist
mkdir -p $PROJ/checkpoints

# Optional: Setup W&B logging
if [ "$1" == "--wandb" ]; then
    export WANDB_DISABLED=false
    echo "Enabling W&B logging"
    wandb login
else
    export WANDB_DISABLED=true
    echo "W&B logging disabled (use --wandb to enable)"
fi

# Train Books model
echo "\n=== Training SASRec on Amazon Books dataset ==="
python $PROJ/code/ranker/train_sasrec.py --dataset books --epochs 10 --seed 42
if [ $? -ne 0 ]; then
    echo "❌ Error training Books model"
    exit 1
fi

# Train MovieLens model
echo "\n=== Training SASRec on MovieLens-25M dataset ==="
python $PROJ/code/ranker/train_sasrec.py --dataset ml25m --epochs 7 --seed 42
if [ $? -ne 0 ]; then
    echo "❌ Error training MovieLens model"
    exit 1
fi

# Train Steam model
echo "\n=== Training SASRec on Steam-200K dataset ==="
python $PROJ/code/ranker/train_sasrec.py --dataset steam --epochs 8 --seed 42
if [ $? -ne 0 ]; then
    echo "❌ Error training Steam model"
    exit 1
fi

# Generate manifest
echo "\n=== Generating checkpoint manifest ==="
for d in books ml25m steam; do
  sha=$(sha256sum $PROJ/checkpoints/sasrec_${d}.pt | cut -d' ' -f1)
  echo "{\"dataset\":\"$d\",\"sha256\":\"$sha\",\"timestamp\":\"$(date -Iseconds)\"}" \
       >> $PROJ/data/_checksums/ranker_manifest.jsonl
done

# Validate checkpoints
echo "\n=== Validating checkpoints ==="
python $PROJ/code/ranker/validate_checkpoints.py --dataset all

# Print summary
echo "\n=== Training Summary ==="
python - <<'PY'
import torch, json, os, pathlib
proj = pathlib.Path(os.getenv('PROJ', '.'))
for d in ["books","ml25m","steam"]:
    ck = proj/"checkpoints"/f"sasrec_{d}.pt"
    stat_path = proj/"checkpoints"/f"sasrec_{d}_stats.json"
    if not ck.exists() or not stat_path.exists():
        print(f"{d}: ❌ Missing files")
        continue
    stat = json.load(open(stat_path))
    m = torch.load(ck, map_location='cpu')
    n_items = m['item_emb.weight'].shape[0]-1
    print(f"{d}: items={n_items:,}, NDCG={stat['ndcg10']:.3f}, HR={stat['hr10']:.3f}")
PY

echo "\n✓ SASRec training complete!"
