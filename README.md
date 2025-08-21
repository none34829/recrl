# Shielded RecRL: Gradient-Shielded Recommender Systems with Explanations

This repository contains the implementation of **Shielded RecRL**, a novel method for adding chat-style explanations to recommender systems without affecting the underlying ranking model's performance. The key innovation is a gradient projection technique that prevents the explanation model from interfering with the ranking model.

## 🎯 Project Overview

Shielded RecRL uses a two-tower architecture:
- **Frozen Ranking Model**: SASRec (collaborative filtering) that remains unchanged
- **Trainable Language Model**: TinyLlama-1.1B with LoRA adapters that generates explanations

The gradient projection technique ensures that the explanation model can learn to generate helpful explanations without degrading the ranking model's recommendation quality.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (8GB+ VRAM for local testing, 24GB+ for full experiments)
- **Git**
- **Conda** (recommended for environment management)

### Local Setup (Linux/macOS)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/shielded-recrl.git
   cd shielded-recrl
   ```

2. **Set up the environment:**
   ```bash
   # Create conda environment
   conda create -n rec python=3.10
   conda activate rec
   
   # Install PyTorch with CUDA support
   conda install pytorch=2.3 torchvision=0.18 torchaudio=2.3 pytorch-cuda=12.2 -c pytorch -c nvidia
   
   # Install other dependencies
   pip install -r requirements.txt
   
   # Fix bitsandbytes for CUDA 12.2
   pip install bitsandbytes-cuda122==0.43.0
   ```

3. **Verify installation:**
   ```bash
   python gpu_test.py
   bash environment_check.sh
   ```

### Windows Setup

For Windows users, see the detailed guide in [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md) or run:

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows.ps1
```

## 📊 Datasets

The project supports three datasets:
- **Amazon Books** (recommended for testing)
- **MovieLens-25M** 
- **Steam-200K**

## 🔧 Running Experiments

### Option 1: Quick Smoke Test (Local)

For a quick test on a 6GB GPU:

```bash
# Run the complete pipeline with minimal settings
bash scripts/run_local_smoke.sh
```

This script will:
1. Download and preprocess the Books dataset
2. Train a SASRec ranking model
3. Compute gradient projection basis
4. Initialize LoRA adapters with TinyLlama
5. Run a small Shielded RecRL training session

### Option 2: Full Experiments (RunPod/High-end GPU)

For full experiments, we recommend using RunPod with an A100 80GB GPU:

1. **Launch RunPod instance:**
   - Runtime: PyTorch 2.3 | Python 3.10 | CUDA 12.2
   - GPU: NVIDIA A100 80GB or 2× RTX 4090 24GB
   - Volume: ≥ 400GB

2. **SSH into your instance:**
   ```bash
   ssh -p YOUR_PORT runpod@YOUR_POD_ID.connect.runpod.io
   ```

3. **Set up the environment:**
   ```bash
   # Clone your repository
   cd /workspace
   git clone https://github.com/your-username/shielded-recrl.git
   cd shielded-recrl
   
   # Run setup script
   bash setup_runpod.sh
   ```

4. **Run full experiments:**
   ```bash
   bash scripts/run_runpod_full.sh
   ```

### Option 3: Manual Step-by-Step

If you prefer to run each step manually:

#### Step 1: Data Preprocessing
```bash
# Download and preprocess all datasets
bash code/dataset/run_preprocessing.sh
```

#### Step 2: Train Ranking Models
```bash
# Train SASRec on all datasets
cd code/ranker
bash run_training.sh
cd ../..
```

#### Step 3: Compute Projection Basis
```bash
# Compute gradient projection basis for each dataset
cd code/projection
python run_basis.py --dataset books --proj_dir ../..
python run_basis.py --dataset ml25m --proj_dir ../..
python run_basis.py --dataset steam --proj_dir ../..
cd ../..
```

#### Step 4: Initialize Language Models
```bash
# Initialize LoRA adapters for all datasets
cd code/explainer
bash run_lora_init.sh --int8
cd ../..
```

#### Step 5: Run Shielded RecRL Training
```bash
# Main experiment with gradient projection
cd code/trainer
python run_recrl_cli.py \
  --dataset books \
  --ranker_ckpt ../../checkpoints/sasrec_books.pt \
  --projection_basis ../../checkpoints/basis_books.pt \
  --lora_rank 16 \
  --kl_beta 0.05 \
  --micro_batch_size 4 \
  --grad_accum 2 \
  --max_seq_len_explainer 384 \
  --explanation_max_len 160 \
  --max_steps 20000 \
  --tag shielded

# Ablation: No projection
python run_recrl_cli.py \
  --dataset books \
  --ranker_ckpt ../../checkpoints/sasrec_books.pt \
  --no_projection \
  --lora_rank 16 \
  --kl_beta 0.05 \
  --max_steps 10000 \
  --tag no_proj

# Ablation: KL=0
python run_recrl_cli.py \
  --dataset books \
  --ranker_ckpt ../../checkpoints/sasrec_books.pt \
  --projection_basis ../../checkpoints/basis_books.pt \
  --lora_rank 16 \
  --kl_beta 0.0 \
  --max_steps 10000 \
  --tag kl0
cd ../..
```

#### Step 6: Run Audits
```bash
# Run toxicity, bias, and privacy audits
cd code/audit
python run_audit.py --model_ckpt ../../checkpoints/recrl/books/latest
cd ../..
```

#### Step 7: Aggregate Results
```bash
# Generate final results and plots
cd code/eval
python aggregate_main.py \
  --runs_dir ../../logs \
  --out_csv ../../experiments/aggregate_results.csv \
  --out_dir ../../experiments/figs
cd ../..
```

## 📁 Project Structure

```
shielded-recrl/
├── code/
│   ├── dataset/          # Dataset preprocessing and download
│   ├── ranker/           # SASRec ranking model implementation
│   ├── explainer/        # LLM with LoRA adapters
│   ├── projection/       # Gradient projection implementation
│   ├── trainer/          # Shielded PPO training
│   ├── eval/             # Evaluation and aggregation
│   └── audit/            # Toxicity, bias, privacy audits
├── data/
│   ├── raw/              # Raw datasets
│   ├── proc/             # Processed datasets
│   └── _checksums/       # Dataset integrity checks
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── experiments/          # Configuration files
├── docs/                 # Documentation
├── scripts/              # Automation scripts
└── docker/               # Docker configuration
```

## 🔬 Key Components

### 1. Gradient Projection (`code/projection/`)
- Computes orthogonal basis for the ranking model's parameter space
- Projects gradients to prevent interference with ranking performance

### 2. Shielded PPO (`code/trainer/`)
- Implements PPO with gradient projection
- Maintains ranking model performance while training explanations

### 3. Multi-dataset Support
- **Amazon Books**: E-commerce recommendations
- **MovieLens-25M**: Movie recommendations  
- **Steam-200K**: Game recommendations

## 📈 Results

The experiments evaluate:
- **Ranking Performance**: NDCG@10, HR@10 (should remain stable)
- **Explanation Quality**: BLEU, ROUGE, human evaluation
- **Safety**: Toxicity, bias, privacy audits

## 🛠️ Configuration

Modify experiment settings in `experiments/recrl_default.yaml`:

```yaml
books:
  epochs: 8
  ppo_batch: 256
  sim_batch: 32
  kl_beta: 0.05
  int8: true
  lr: 3e-5
  max_new_tokens: 40
  temperature: 0.7
  top_p: 0.9
  seed: 42
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--int8` flag or reduce batch sizes
2. **Import errors**: Ensure conda environment is activated (`conda activate rec`)
3. **Dataset download fails**: Check internet connection and disk space
4. **Git LFS issues**: Install Git LFS: `sudo apt-get install git-lfs && git lfs install`

### Memory Requirements

- **Local testing**: 6GB GPU (use TinyLlama + 4-bit quantization)
- **Full experiments**: 24GB+ GPU (A100 80GB recommended)
- **CPU fallback**: Available but very slow

### Performance Tips

- Use `--int8` for 8-bit quantization to reduce memory usage
- Adjust `micro_batch_size` and `grad_accum` based on your GPU
- For local testing, use the Books dataset only

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{shielded-recrl,
  title={Shielded RecRL: Gradient-Shielded Recommender Systems with Explanations},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- SASRec implementation based on [SASRec-PyTorch](https://github.com/kang205/SASRec)
- LoRA implementation using [PEFT](https://github.com/huggingface/peft)
- Evaluation metrics from [RecSim-NG](https://github.com/google-research/recsim_ng)

## 📞 Support

For questions and issues:
1. Check the troubleshooting section above
2. Review the documentation in `docs/`
3. Open an issue on GitHub
4. Check the project's discussion page

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.
