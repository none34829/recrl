#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# RunPod environment (A100 80GB)
# -----------------------------

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_OFFLINE=0

# choose dataset(s) for full runs
DATASETS=("books") # add "ml25m" "steam" later if you want

# big model for full runs
export RECRL_MODEL_NAME="EleutherAI/gpt-j-6B"
export RECRL_4BIT=1 # 4-bit QLoRA keeps memory comfy; you can set 8-bit too
export RECRL_MAX_GPU_MEM="72GiB"
export RECRL_CPU_MEM="128GiB"

# project root (assumes you've git cloned your repo already)
ROOT_DIR="$HOME/recrl"
mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

echo "[1/10] Setup base image deps"
bash setup_runpod.sh
python gpu_test.py || true
bash environment_check.sh || true

echo "[2/10] Restore your local artifacts"
RESTORE_DIR="$ROOT_DIR/restore"

echo "[3/10] Prepare data and ranker for each dataset"
for DATASET in "${DATASETS[@]}"; do
    if [ -d "$RESTORE_DIR/$DATASET/proc" ]; then
        echo "Using restored processed data for $DATASET"
        mkdir -p data/proc
        rsync -a "$RESTORE_DIR/$DATASET/proc" "data/proc/$DATASET"
    else
        echo "Re-preprocessing $DATASET"
        bash code/dataset/run_preprocessing.sh
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
done

echo "[4/10] Resolve ranker ckpt & projection basis"
for DATASET in "${DATASETS[@]}"; do
    RANKER_CKPT="code/ranker/checkpoints/$DATASET/best.ckpt"
    
    if [ -f "$RESTORE_DIR/$DATASET/${DATASET}_Q.pt" ]; then
        PROJ_BASIS="$RESTORE_DIR/$DATASET/${DATASET}_Q.pt"
    else
        echo "Computing projection basis for $DATASET"
        cd code/projection
        python run_basis.py --ranker_ckpt "$ROOT_DIR/$RANKER_CKPT" --out "$ROOT_DIR/checkpoints/${DATASET}_Q.pt"
        PROJ_BASIS="$ROOT_DIR/checkpoints/${DATASET}_Q.pt"
        cd "$ROOT_DIR"
    fi
    
    echo "$DATASET:RANKER_CKPT=$RANKER_CKPT"
    echo "$DATASET:PROJ_BASIS=$PROJ_BASIS"
    
    echo "[5/10] MAIN RUN (Shielded RecRL)"
    cd code/trainer
    python run_recrl_cli.py \
        --dataset "$DATASET" \
        --ranker_ckpt "$ROOT_DIR/$RANKER_CKPT" \
        --projection_basis "$PROJ_BASIS" \
        --lora_rank 16 \
        --kl_beta 0.05 \
        --micro_batch_size 4 \
        --grad_accum 2 \
        --max_seq_len_explainer 384 \
        --explanation_max_len 160 \
        --max_steps 20000 \
        --tag shielded
    
    echo "[6/10] ABLATIONS"
    # (a) No projection
    python run_recrl_cli.py \
        --dataset "$DATASET" \
        --ranker_ckpt "$ROOT_DIR/$RANKER_CKPT" \
        --no_projection \
        --lora_rank 16 \
        --kl_beta 0.05 \
        --micro_batch_size 4 \
        --grad_accum 2 \
        --max_seq_len_explainer 384 \
        --explanation_max_len 160 \
        --max_steps 10000 \
        --tag no_proj
    
    # (b) KL=0
    python run_recrl_cli.py \
        --dataset "$DATASET" \
        --ranker_ckpt "$ROOT_DIR/$RANKER_CKPT" \
        --projection_basis "$PROJ_BASIS" \
        --lora_rank 16 \
        --kl_beta 0.0 \
        --micro_batch_size 4 \
        --grad_accum 2 \
        --max_seq_len_explainer 384 \
        --explanation_max_len 160 \
        --max_steps 10000 \
        --tag kl0
    
    # (c) LoRA rank 4
    python run_recrl_cli.py \
        --dataset "$DATASET" \
        --ranker_ckpt "$ROOT_DIR/$RANKER_CKPT" \
        --projection_basis "$PROJ_BASIS" \
        --lora_rank 4 \
        --kl_beta 0.05 \
        --micro_batch_size 6 \
        --grad_accum 2 \
        --max_seq_len_explainer 384 \
        --explanation_max_len 160 \
        --max_steps 10000 \
        --tag lora4
    
    # (d) Explanation length x2
    python run_recrl_cli.py \
        --dataset "$DATASET" \
        --ranker_ckpt "$ROOT_DIR/$RANKER_CKPT" \
        --projection_basis "$PROJ_BASIS" \
        --lora_rank 16 \
        --kl_beta 0.05 \
        --micro_batch_size 4 \
        --grad_accum 2 \
        --max_seq_len_explainer 512 \
        --explanation_max_len 320 \
        --max_steps 10000 \
        --tag long_expl
    
    cd "$ROOT_DIR"
done

echo "[7/10] Audits (toxicity, bias, privacy)"
cd code/audit
# point to the *latest* LoRA diff or model dir saved by the trainer
LATEST_MODEL="$(ls -dt "$ROOT_DIR"/checkpoints/recrl/${DATASET}/* | head -n1)"
python run_audit.py --model_ckpt "$LATEST_MODEL"
cd "$ROOT_DIR"

echo "[8/10] Aggregate results into one CSV + figs"
cd code/eval
python aggregate_main.py \
    --runs_dir "$ROOT_DIR/logs" \
    --out_csv "$ROOT_DIR/experiments/aggregate_results.csv" \
    --out_dir "$ROOT_DIR/experiments/figs"
cd "$ROOT_DIR"

echo "[9/10] Snapshot configs for reproducibility"
mkdir -p experiments/snapshots
cp experiments/recrl_default.yaml experiments/snapshots/recrl_default_$(date +%Y%m%d_%H%M%S).yaml || true

echo "[10/10] Where to find things"
echo " Metrics CSV: experiments/aggregate_results.csv"
echo " Plots: experiments/figs/"
echo " Audits (PDFs): code/audit/outputs/ (or wherever run_audit.py writes)"
echo " Checkpoints: checkpoints/recrl/<dataset>/..."
echo "DONE."

