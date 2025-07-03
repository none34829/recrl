#!/bin/bash
# Initialize LoRA adapters for all datasets on RunPod

# Set project directory
if [ -z "$PROJ" ]; then
    export PROJ=$(pwd)
    echo "PROJ not set, using current directory: $PROJ"
fi

# Configure HuggingFace cache path
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME

# Ensure conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "rec" ]; then
    echo "Please activate the 'rec' conda environment first:"
    echo "conda activate rec"
    exit 1
fi

# Check for CUDA
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "❌ CUDA not available. This script requires GPU access."
    exit 1
fi

# Check for int8 flag
USE_INT8=false
if [ "$1" == "--int8" ]; then
    USE_INT8=true
    INT8_FLAG="--int8"
    echo "Using 8-bit quantization to reduce VRAM usage"
else
    INT8_FLAG=""
    echo "Using full FP16 precision (use --int8 to reduce VRAM usage)"
fi

# Create checkpoints directory if it doesn't exist
mkdir -p $PROJ/checkpoints

# Initialize LoRA adapters for each dataset
for dataset in books ml25m steam; do
    echo "\n=== Initializing LoRA adapter for $dataset dataset ==="
    python $PROJ/code/explainer/init_lora.py --dataset $dataset $INT8_FLAG
    if [ $? -ne 0 ]; then
        echo "❌ Error initializing LoRA adapter for $dataset"
        exit 1
    fi
done

# Create manifest entries
echo "\n=== Creating manifest entries ==="
python $PROJ/code/explainer/create_manifest.py

# Check memory usage and generation
echo "\n=== Checking memory usage and generation ==="
python $PROJ/code/explainer/check_memory.py $INT8_FLAG

# Optional: Track LoRA weights in Git LFS
echo "\n=== Setting up Git LFS tracking (optional) ==="
echo "To track LoRA weights in Git LFS, run:"
echo "git lfs track \"checkpoints/lora_init_*.pt\""
echo "git add .gitattributes checkpoints/*_meta.json"
echo "git commit -m \"Add untrained LoRA adapters + meta\""
echo "git push"

# Print summary
echo "\n=== LoRA Initialization Summary ==="
for dataset in books ml25m steam; do
    meta_file="$PROJ/checkpoints/lora_init_${dataset}_meta.json"
    if [ -f "$meta_file" ]; then
        lora_r=$(jq -r '.lora_r' "$meta_file")
        sha=$(jq -r '.sha256' "$meta_file")
        echo "$dataset: LoRA rank=$lora_r, SHA256=${sha:0:8}..."
    else
        echo "$dataset: ❌ Missing metadata file"
    fi
done

echo "\n✓ LoRA initialization complete!"
