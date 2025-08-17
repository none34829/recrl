# RunPod Full Experiments - Execution Guide

## 🚀 Quick Start (Copy-Paste Runbook)

### Step 1: Upload Transfer Bundle

**On your Windows machine (PowerShell):**
```powershell
# Upload the transfer package we created
scp .\transfer_out_books.zip runpod@<YOUR_POD_HOST>:~/recrl/
```

### Step 2: Bootstrap RunPod

**On your RunPod instance:**
```bash
cd ~/recrl
unzip -o transfer_out_books.zip -d restore

# Base setup
bash setup_runpod.sh

# Install dependencies (GPU builds)
pip install -U pip
pip install -r requirements.txt
pip install -U bitsandbytes accelerate datasets wandb
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U recsim-ng

# Set environment variables
export WANDB_MODE=offline
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Test setup
python gpu_test.py
bash environment_check.sh
```

### Step 3: Set Paths

```bash
# Use our restored artifacts
RANKER="restore/books/ranker_ckpt.pt"
BASIS="restore/books/books_Q.pt"

# Verify files exist
ls -la "$RANKER" "$BASIS"
```

### Step 4: Start tmux (Recommended)

```bash
tmux new -s recrl
# (Detach later with Ctrl-b then d)
```

### Step 5: Run Main Experiment

```bash
# Set model environment
export RECRL_MODEL_NAME="EleutherAI/gpt-j-6B"
export RECRL_4BIT=1
export RECRL_MAX_GPU_MEM="72GiB"
export RECRL_CPU_MEM="128GiB"

# Run main Shielded RecRL experiment
python code/trainer/run_recrl_cli.py \
  --dataset books \
  --ranker_ckpt "$RANKER" \
  --projection_basis "$BASIS" \
  --lora_rank 16 \
  --kl_beta 0.05 \
  --micro_batch_size 4 \
  --grad_accum 2 \
  --max_seq_len_explainer 384 \
  --explanation_max_len 160 \
  --max_steps 20000 \
  --tag shielded
```

### Step 6: Monitor Progress

**In another tmux pane:**
```bash
# Watch logs for CTR/NDCG
tail -f logs/*.log 2>/dev/null | sed -n 's/.*\(CTR[^,]*\|NDCG[^,]*\).*/\0/p'

# Monitor GPU usage
watch -n 10 nvidia-smi

# Check disk space
df -h .
```

### Step 7: Run Ablations

**After main experiment completes:**

```bash
# A) No projection
python code/trainer/run_recrl_cli.py \
  --dataset books --ranker_ckpt "$RANKER" --no_projection \
  --lora_rank 16 --kl_beta 0.05 \
  --micro_batch_size 4 --grad_accum 2 \
  --max_seq_len_explainer 384 --explanation_max_len 160 \
  --max_steps 10000 --tag no_proj

# B) KL = 0
python code/trainer/run_recrl_cli.py \
  --dataset books --ranker_ckpt "$RANKER" --projection_basis "$BASIS" \
  --lora_rank 16 --kl_beta 0.0 \
  --micro_batch_size 4 --grad_accum 2 \
  --max_seq_len_explainer 384 --explanation_max_len 160 \
  --max_steps 10000 --tag kl0

# C) LoRA rank 4
python code/trainer/run_recrl_cli.py \
  --dataset books --ranker_ckpt "$RANKER" --projection_basis "$BASIS" \
  --lora_rank 4 --kl_beta 0.05 \
  --micro_batch_size 6 --grad_accum 2 \
  --max_seq_len_explainer 384 --explanation_max_len 160 \
  --max_steps 10000 --tag lora4

# D) Explanation length ×2
python code/trainer/run_recrl_cli.py \
  --dataset books --ranker_ckpt "$RANKER" --projection_basis "$BASIS" \
  --lora_rank 16 --kl_beta 0.05 \
  --micro_batch_size 4 --grad_accum 2 \
  --max_seq_len_explainer 512 --explanation_max_len 320 \
  --max_steps 10000 --tag long_expl
```

### Step 8: Safety Audits + Aggregation

```bash
# Run safety audits
LATEST="$(ls -dt checkpoints/recrl/books/* | head -n1 || true)"
[ -n "$LATEST" ] && python code/audit/run_audit.py --model_ckpt "$LATEST" || true

# Aggregate results
python code/eval/aggregate_main.py \
  --runs_dir logs \
  --out_csv experiments/aggregate_results.csv \
  --out_dir experiments/figs
```

### Step 9: Download Results

**From RunPod to your laptop:**
```bash
# Download experiment results
scp -r runpod@<YOUR_POD_HOST>:~/recrl/experiments .

# Download audit reports
scp -r runpod@<YOUR_POD_HOST>:~/recrl/code/audit/outputs ./audit_outputs
```

## 📊 Expected Outputs

After completion, you'll have:

1. **`experiments/aggregate_results.csv`** - Main results table
2. **`experiments/figs/`** - CTR curves, NDCG stability, ablation bars
3. **`audit_outputs/`** - Toxicity %, Δ-Gini, privacy flags
4. **`checkpoints/recrl/books/`** - Trained LoRA models

## 🎯 Success Indicators

**What to look for during training:**
- ✅ Logs mention "projection basis shape" and "RecSim active"
- ✅ CTR trending upward
- ✅ NDCG@10 staying relatively flat (shielding works)
- ✅ Checkpoints appearing under `checkpoints/recrl/books/`

## ⚠️ Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce memory usage
--max_seq_len_explainer 256  # instead of 384
--micro_batch_size 2         # instead of 4
--lora_rank 8                # instead of 16
```

### RecSim/JAX Errors
```bash
# Reinstall JAX with CUDA support
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
```

### Weird Metrics
```bash
# Recompute projection basis
python code/projection/run_basis.py \
  --ranker_ckpt "$RANKER" \
  --out checkpoints/books_Q.pt
```

## 📝 Paper Checklist

**For your paper, you'll have:**

- **Table 1:** Shielded vs. Baseline (NDCG@10, CTR Δ, coverage)
- **Table 2:** Ablation studies (No-proj, KL=0, LoRA-4, Long-expl)
- **Figure 1:** CTR vs training steps
- **Figure 2:** NDCG stability over time
- **Figure 3:** Ablation comparison bars
- **Methods Section:** LoRA rank, KL β, sequence lengths, model specs
- **Reproducibility:** Commit SHA, seeds, GPU type, checkpoint paths

## 🚀 Ready to Launch!

**Your local smoke test proved everything works. The full experiments will give you:**

- **Main Result:** CTR↑ with flat NDCG (proving shielding works)
- **Ablation Studies:** 4 different configurations for analysis
- **Safety Audits:** Comprehensive evaluation reports
- **Paper-Ready Outputs:** Tables, plots, and analysis ready for publication

**Good luck with your experiments!** 🎯
