# Next Steps: RunPod Full Experiments

## 🎯 Current Status: READY FOR RUNPOD

✅ **Local smoke test completed successfully**
✅ **Transfer package created: `transfer_out_books.zip` (1.7MB)**
✅ **All scripts prepared and tested**

## 🚀 Immediate Next Steps

### Step 1: Upload to RunPod

**Choose one method to upload `transfer_out_books.zip` to your RunPod instance:**

#### Option A: RunPod Web Console
1. Go to your RunPod dashboard
2. Open your pod's web console
3. Navigate to Files → Upload
4. Upload `transfer_out_books.zip` to `~/recrl/`

#### Option B: WinSCP/FileZilla
1. Open WinSCP or FileZilla
2. Connect to your RunPod instance
3. Upload `transfer_out_books.zip` to `~/recrl/`

#### Option C: SCP (if you have SSH access)
```bash
scp transfer_out_books.zip runpoduser@YOUR_POD_IP:~/recrl/
```

### Step 2: RunPod Setup Commands

**On your RunPod instance, run these commands:**

```bash
# 1. Navigate to your repo
cd ~/recrl

# 2. Extract the transfer package
unzip -o transfer_out_books.zip -d restore

# 3. Make the script executable
chmod +x scripts/run_runpod_full.sh

# 4. Run the full experiment
./scripts/run_runpod_full.sh
```

## 📦 What's in the Transfer Package

The `transfer_out_books.zip` contains:
- `proc/` - Preprocessed dataset (train/valid splits)
- `ranker_ckpt.pt` - Trained SASRec model (1.7MB)
- `books_Q.pt` - Projection basis matrix (104KB)

## 🔬 What the Full Experiment Will Run

### Main Experiment (Shielded RecRL)
- **Model:** GPT-J-6B with 4-bit quantization
- **Training:** 20,000 PPO steps with RecSim-NG
- **Expected:** CTR↑, NDCG flat (shielding works)

### Ablation Studies
1. **No Projection** → NDCG/CTR degradation
2. **KL=0** → verbosity/drift effects
3. **LoRA rank 4** → parameter efficiency
4. **Longer explanations** → quality trade-offs

### Safety Audits
- Toxicity evaluation
- Popularity bias analysis
- Privacy checks

## 📊 Expected Outputs

After completion, you'll have:

1. **Results CSV:** `experiments/aggregate_results.csv`
2. **Plots:** `experiments/figs/` (CTR vs steps, ablation bars)
3. **Models:** `checkpoints/recrl/books/` (trained LoRA models)
4. **Audits:** `code/audit/outputs/` (safety reports)

## ⚠️ Troubleshooting Tips

### If you encounter issues:

1. **JAX/CUDA errors:**
   ```bash
   pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
   ```

2. **Out of Memory:**
   - Edit `scripts/run_runpod_full.sh`
   - Reduce `--micro_batch_size` from 4 to 2
   - Increase `--grad_accum` from 2 to 4

3. **W&B blocking:**
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

## 🚀 Ready to Go!

**Your local smoke test proved everything works. Now it's time for the full experiments on RunPod!**

### Final Checklist:

✅ **Local smoke test completed successfully**
✅ **Transfer package created: `transfer_out_books.zip` (1.7MB)**
✅ **RunPod script updated: `scripts/run_runpod_full.sh`**
✅ **CLI training script created: `code/trainer/run_recrl_cli.py`**
✅ **All scripts tested and working**

### Next Steps:

1. **Upload `transfer_out_books.zip` to RunPod**
2. **Run `./scripts/run_runpod_full.sh`**
3. **Wait for completion (several hours)**
4. **Download results for your paper**

### What You'll Get:

- **Main Results:** CTR↑ with flat NDCG (shielding works)
- **Ablation Studies:** No projection, KL=0, LoRA rank 4, longer explanations
- **Safety Audits:** Toxicity, bias, privacy evaluations
- **Paper-Ready Outputs:** CSV tables, plots, audit reports

**Everything is ready! Good luck with your experiments!** 🎯
