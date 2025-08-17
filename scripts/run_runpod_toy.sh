#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# RunPod environment (A100 80GB)
# -----------------------------

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_OFFLINE=0
export WANDB_MODE=offline

# choose dataset for toy run
DATASET="books"

# big model for full runs
export RECRL_MODEL_NAME="EleutherAI/gpt-j-6B"
export RECRL_4BIT=1 # 4-bit QLoRA keeps memory comfy
export RECRL_MAX_GPU_MEM="72GiB"
export RECRL_CPU_MEM="128GiB"

# project root (assumes you've git cloned your repo already)
ROOT_DIR="$HOME/recrl"
mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

echo "[1/6] Setup base image deps"
bash setup_runpod.sh
python gpu_test.py || true
bash environment_check.sh || true

echo "[2/6] Restore your local artifacts"
RESTORE_DIR="$ROOT_DIR/restore"

if [ -d "$RESTORE_DIR/$DATASET/proc" ]; then
    echo "Using restored processed data for $DATASET"
    mkdir -p data/proc
    rsync -a "$RESTORE_DIR/$DATASET/proc" "data/proc/$DATASET"
else
    echo "ERROR: Processed data not found in restore directory"
    echo "Expected: $RESTORE_DIR/$DATASET/proc"
    exit 1
fi

if [ -f "$RESTORE_DIR/$DATASET/ranker_ckpt.pt" ]; then
    echo "Using restored ranker ckpt for $DATASET"
    mkdir -p code/ranker/checkpoints/$DATASET
    cp "$RESTORE_DIR/$DATASET/ranker_ckpt.pt" "code/ranker/checkpoints/$DATASET/best.ckpt"
else
    echo "ERROR: Ranker checkpoint not found in restore directory"
    echo "Expected: $RESTORE_DIR/$DATASET/ranker_ckpt.pt"
    exit 1
fi

echo "[3/6] Resolve projection basis"
if [ -f "$RESTORE_DIR/$DATASET/${DATASET}_Q.pt" ]; then
    PROJ_BASIS="$RESTORE_DIR/$DATASET/${DATASET}_Q.pt"
    echo "Using restored projection basis: $PROJ_BASIS"
else
    echo "Computing projection basis for $DATASET"
    cd code/projection
    python run_basis.py --ranker_ckpt "$ROOT_DIR/code/ranker/checkpoints/$DATASET/best.ckpt" --out "$ROOT_DIR/checkpoints/${DATASET}_Q.pt"
    PROJ_BASIS="$ROOT_DIR/checkpoints/${DATASET}_Q.pt"
    cd "$ROOT_DIR"
fi

echo "[4/6] Initialize LoRA adapters"
cd code/explainer
bash run_lora_init.sh --int8 || true
cd "$ROOT_DIR"

echo "[5/6] Run improved toy PPO training"
cd code/trainer

# Run the improved toy training with different step counts
echo "Running 5-step toy training..."
python run_toy.py \
    --dataset "$DATASET" \
    --steps 5 \
    --lr 3e-5 \
    --kl_beta 0.05

echo "Running 20-step toy training..."
python run_toy.py \
    --dataset "$DATASET" \
    --steps 20 \
    --lr 3e-5 \
    --kl_beta 0.05

echo "Running 50-step toy training..."
python run_toy.py \
    --dataset "$DATASET" \
    --steps 50 \
    --lr 3e-5 \
    --kl_beta 0.05

cd "$ROOT_DIR"

echo "[6/6] Results summary"
echo "Checkpoints saved to: checkpoints/recrl_books_toy.pt"
echo "W&B logs saved to: wandb/offline-run-*"
echo "Training completed successfully!"

echo "DONE. Your improved PPO training has been completed on RunPod!"
