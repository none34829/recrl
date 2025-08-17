#!/bin/bash

# RunPod Setup and Experiment Script
# Copy and paste this entire script into the RunPod web terminal

set -e  # Exit on any error

echo "=== Starting RunPod Setup and Experiments ==="

# Navigate to the workspace directory
cd /workspace/recrl
echo "Working directory: $(pwd)"

# Install unzip if not available
if ! command -v unzip &> /dev/null; then
    echo "Installing unzip..."
    apt update && apt install -y unzip
fi

# Install Python dependencies manually (avoiding requirements.txt issues)
echo "Installing Python dependencies..."
pip install transformers==4.41.0
pip install peft==0.11.1
pip install accelerate==0.29.2
pip install bitsandbytes==0.43.0
pip install datasets==2.19.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install tqdm==4.66.1
pip install faiss-cpu==1.7.4
pip install recsim-ng==0.1.2
pip install torch==2.1.2
pip install wandb==0.16.3
pip install click==8.1.7
pip install pyyaml==6.0.1

# Set up environment variables
export RECRL_MODEL_NAME="EleutherAI/gpt-j-6B"
export RECRL_4BIT=0
export RECRL_MAX_GPU_MEM="72GiB"
export RECRL_CPU_MEM="128GiB"
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"

# Create necessary directories
mkdir -p data/proc/books
mkdir -p code/ranker/checkpoints/books
mkdir -p checkpoints

# Extract the transfer package if it exists
if [ -f "transfer_out_books.zip" ]; then
    echo "Extracting transfer package..."
    unzip -o transfer_out_books.zip
fi

# Copy files to correct locations
if [ -d "transfer_out/books" ]; then
    echo "Copying files to correct locations..."
    cp -r transfer_out/books/proc/* data/proc/books/ 2>/dev/null || echo "No proc files to copy"
    cp transfer_out/books/ranker_ckpt.pt code/ranker/checkpoints/books/best.ckpt 2>/dev/null || echo "No ranker checkpoint to copy"
    cp transfer_out/books/books_Q.pt checkpoints/books_Q.pt 2>/dev/null || echo "No Q matrix to copy"
fi

# Verify setup
echo "=== Verifying Setup ==="
echo "GPU Status:"
nvidia-smi

echo "Directory Structure:"
ls -la data/proc/books/ 2>/dev/null || echo "No proc data found"
ls -la code/ranker/checkpoints/books/ 2>/dev/null || echo "No ranker checkpoint found"
ls -la checkpoints/ 2>/dev/null || echo "No checkpoints found"

# Initialize LoRA
echo "=== Initializing LoRA ==="
cd code/explainer
python init_lora.py --dataset books
cd ../..

# Run the main experiment
echo "=== Starting Main Experiment ==="
cd code/trainer
python run_toy.py --dataset books --epochs 10 --ppo_batch 32
cd ../..

echo "=== Setup Complete! ==="
echo "Check the output above for any errors."
echo "If successful, the experiment should be running!"
