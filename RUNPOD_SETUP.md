# RunPod Full Experiments Setup

## 🚀 Quick Start

### 1. Upload Local Artifacts to RunPod

**From your local Windows machine:**

```powershell
# The transfer package is ready: transfer_out_books.zip (1.7MB)
# Upload this to your RunPod instance using one of these methods:

# Method A: WinSCP/FileZilla
# - Upload transfer_out_books.zip to ~/recrl/ on your RunPod

# Method B: Web Console
# - Go to RunPod web console → Files → Upload transfer_out_books.zip

# Method C: SCP (if you have SSH access)
# scp transfer_out_books.zip runpoduser@YOUR_POD_IP:~/recrl/
```

### 2. RunPod Setup Commands

**On your RunPod instance:**

```bash
# 1. Clone your repo (if not already done)
cd ~
git clone https://github.com/YOUR_USERNAME/recrl.git
cd recrl

# 2. Extract the transfer package
unzip -o transfer_out_books.zip -d restore

# 3. Run the full experiment script
chmod +x scripts/run_runpod_full.sh
./scripts/run_runpod_full.sh
```

## 📦 What's in the Transfer Package

The `transfer_out_books.zip` contains:
- `proc/` - Preprocessed dataset (train/valid splits)
- `ranker_ckpt.pt` - Trained SASRec model (1.7MB)
- `books_Q.pt` - Projection basis matrix (104KB)

## 🔧 Manual Setup (if needed)

If you prefer to run steps manually:

```bash
# 1. Base setup
bash setup_runpod.sh

# 2. Install dependencies
pip install -U pip
pip install -r requirements.txt
pip install -U bitsandbytes accelerate datasets wandb
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U recsim-ng

# 3. Set environment variables
export WANDB_MODE=offline
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export RECRL_MODEL_NAME="EleutherAI/gpt-j-6B"
export RECRL_4BIT=1
export RECRL_MAX_GPU_MEM="72GiB"
export RECRL_CPU_MEM="128GiB"

# 4. Restore artifacts
mkdir -p data/proc
cp -r restore/books/proc data/proc/books
mkdir -p code/ranker/checkpoints/books
cp restore/books/ranker_ckpt.pt code/ranker/checkpoints/books/best.ckpt

# 5. Run main experiment
cd code/trainer
python run_recrl.py \
  --dataset books \
  --ranker_ckpt ../../code/ranker/checkpoints/books/best.ckpt \
  --projection_basis ../../restore/books/books_Q.pt \
  --lora_rank 16 \
  --kl_beta 0.05 \
  --micro_batch_size 4 \
  --grad_accum 2 \
  --max_seq_len_explainer 384 \
  --explanation_max_len 160 \
  --max_steps 20000 \
  --tag shielded
```

## 📊 Expected Results

After running the full script, you'll have:

1. **Main Results:**
   - `experiments/aggregate_results.csv` - All metrics in one table
   - `experiments/figs/` - Plots showing CTR↑ and NDCG flat
   - `checkpoints/recrl/books/` - Trained LoRA models

2. **Ablation Studies:**
   - No projection → NDCG/CTR degradation
   - KL=0 → verbosity/drift effects  
   - LoRA rank 4 vs 16 → parameter efficiency
   - Longer explanations → quality trade-offs

3. **Safety Audits:**
   - `code/audit/outputs/` - Toxicity, bias, privacy reports

## ⚠️ Troubleshooting

**Common Issues:**

1. **JAX/CUDA errors:**
   ```bash
   pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
   ```

2. **Out of Memory:**
   - Reduce `--micro_batch_size` from 4 to 2
   - Increase `--grad_accum` from 2 to 4

3. **Projection basis mismatch:**
   ```bash
   cd code/projection
   python run_basis.py --ranker_ckpt ../ranker/checkpoints/books/best.ckpt --out ../../checkpoints/books_Q.pt
   ```

4. **W&B blocking:**
   ```bash
   export WANDB_MODE=offline
   ```

## 🎯 Success Criteria

The experiment is successful if:
- ✅ CTR trends upward during training
- ✅ NDCG@10 stays relatively flat (shielding works)
- ✅ No-projection ablation shows degradation
- ✅ All ablations complete without errors
- ✅ Audit reports are generated

## 📝 For Your Paper

The results will give you:
- **Table 1:** Main results (CTR↑, NDCG flat)
- **Figure 1:** CTR vs training steps
- **Figure 2:** Ablation study bars
- **Section 9:** Safety audit findings

Ready to run! 🚀
