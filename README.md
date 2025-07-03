# Shielded RecRL

This repository contains the implementation of Shielded RecRL, a method for adding chat-style explanations to recommender systems without affecting the underlying ranking model.

## Project Overview

Shielded RecRL uses a two-tower architecture:
- A frozen ranking model (collaborative filtering)
- A trainable language model that generates explanations

The key innovation is the gradient projection technique that prevents the explanation model from affecting the ranking model's performance.

## Setup Instructions

### Local Setup (Any OS)

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/shielded-recrl.git
   cd shielded-recrl
   ```

2. Edit `setup_local.sh` to update your GitHub username, then run:
   ```bash
   bash setup_local.sh
   ```

### RunPod Setup (Remote GPU)

1. Launch a RunPod instance with:
   - Runtime: PyTorch 2.3 | Python 3.10 | CUDA 12.2
   - GPU: NVIDIA A100 80GB or 2× RTX 4090 24GB
   - Volume: ≥ 400GB

2. SSH into your RunPod instance:
   ```bash
   ssh -p YOUR_PORT runpod@YOUR_POD_ID.connect.runpod.io
   ```

3. Edit `setup_runpod.sh` to update your GitHub username, then run:
   ```bash
   bash setup_runpod.sh
   ```

4. Verify the setup:
   ```bash
   python gpu_test.py
   ```

## Project Structure

```
├── code
│   ├── dataset/    # Dataset preprocessing
│   ├── ranker/     # SASRec implementation
│   ├── explainer/  # LLM with LoRA
│   ├── projection/ # Gradient projection
│   ├── trainer/    # Shielded PPO
│   └── eval/       # Evaluation metrics
├── data           # Datasets
├── checkpoints    # Model checkpoints
├── logs           # Training logs
├── experiments    # Experiment configurations
├── docs           # Documentation
└── docker         # Docker configuration
```

## Workflow

1. Edit code on your local machine
2. Commit and push changes to GitHub
3. Pull changes on RunPod and execute experiments
4. Results are logged to W&B and saved to the persistent volume

## License

[Add your license information here]
