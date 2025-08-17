#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Local environment (6 GB GPU)
# -----------------------------
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_OFFLINE=0

# pick one dataset locally to save time
DATASET="books"    # options: books | ml25m | steam

# project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/8] Setup env & quick checks"
bash setup_local.sh
python gpu_test.py || true
bash environment_check.sh || true

echo "[2/8] Download + preprocess datasets (will only use $DATASET later)"
bash code/dataset/run_preprocessing.sh

echo "[3/8] Train frozen SASRec ranker"
cd code/ranker
# Note: run_training.sh trains all datasets, we'll use the one we need
bash run_training.sh

# resolve ranker checkpoint path (adjust if your script writes elsewhere)
RANKER_CKPT="checkpoints/sasrec_${DATASET}.pt"
echo "RANKER_CKPT=$RANKER_CKPT"
cd "$ROOT_DIR"

echo "[4/8] Compute projection basis from the trained ranker"
cd code/projection
python run_basis.py --dataset "$DATASET" --proj_dir "$ROOT_DIR"
PROJ_BASIS="checkpoints/basis_${DATASET}.pt"
echo "PROJ_BASIS=$PROJ_BASIS"
cd "$ROOT_DIR"

echo "[5/8] Init LoRA (tiny settings) + load small model in 4-bit"
# Small model + 4-bit for 6 GB GPU
export RECRL_MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export RECRL_4BIT=1
export RECRL_MAX_GPU_MEM="5GiB"   # keep headroom
export RECRL_CPU_MEM="48GiB"

cd code/explainer
# Note: run_lora_init.sh initializes all datasets, we'll use the one we need
bash run_lora_init.sh --int8 || true
cd "$ROOT_DIR"

echo "[6/8] Tiny end-to-end smoke test (Shielded RecRL)"
cd code/trainer
python run_toy.py \
  --dataset "$DATASET" \
  --ranker_ckpt "$RANKER_CKPT" \
  --projection_basis "$ROOT_DIR/$PROJ_BASIS" \
  --lora_rank 8 \
  --kl_beta 0.05 \
  --max_steps 500 \
  --micro_batch_size 1 \
  --grad_accum 8 \
  --max_seq_len_explainer 192 \
  --explanation_max_len 96

cd "$ROOT_DIR"

echo "[7/8] Sanity: verify projection & CTR script can run"
cd code/trainer
python check_projection.py --projection_basis "$ROOT_DIR/$PROJ_BASIS" || true
python verify_ctr_lift.py --dataset "$DATASET" --dry_run || true
cd "$ROOT_DIR"

echo "[8/8] Package artifacts to transfer later"
mkdir -p transfer_out/${DATASET}
cp -r data/proc/${DATASET} transfer_out/${DATASET}/proc
cp -r code/ranker/checkpoints/${DATASET} transfer_out/${DATASET}/ranker_ckpts
cp "$PROJ_BASIS" transfer_out/${DATASET}/

echo "DONE. Artifacts in transfer_out/${DATASET}"
