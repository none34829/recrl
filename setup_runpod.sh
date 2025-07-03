#!/bin/bash
# RunPod setup script for Shielded RecRL project

# Replace with your GitHub username
GITHUB_USERNAME="your_username"

# Clone repository
cd /workspace
git clone https://github.com/${GITHUB_USERNAME}/shielded-recrl.git
cd shielded-recrl
export PROJ=$PWD
echo 'export PROJ=/workspace/shielded-recrl' >> ~/.bashrc

# Create folder structure
mkdir -p $PROJ/{data,code,checkpoints,logs,experiments,docs}
mkdir -p $PROJ/code/{dataset,ranker,projection,explainer,trainer,eval}

# Create placeholder files to maintain directory structure
find code -type d -empty -exec touch {}/.keep \;
git add code/*/.keep
git commit -m "Add folder scaffolding"
git push

# Pull latest changes
git pull

# Install dependencies
conda install -y pytorch=2.3 torchvision=0.18 torchaudio=2.3 pytorch-cuda=12.2 -c pytorch -c nvidia
pip install -r requirements.txt

# Fix bitsandbytes for CUDA 12.2
pip install bitsandbytes-cuda122==0.43.0

# Setup Git LFS for large checkpoints
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
echo "checkpoints/** filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
git add .gitattributes
git commit -m "Enable Git LFS"
git push

# Freeze environment
conda list --export > docs/conda_freeze.txt
pip freeze > docs/pip_freeze.txt
git add docs/*.txt
git commit -m "Freeze env @RunPod"
git push

echo "RunPod setup complete. Please update GITHUB_USERNAME in the script before running."
