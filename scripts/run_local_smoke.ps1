# PowerShell version of local smoke test script for Windows
# This runs everything that fits on a 6GB GPU

param(
    [string]$Dataset = "books"  # options: books | ml25m | steam
)

# -----------------------------
# Local environment (6 GB GPU)
# -----------------------------
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
$env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "0"

# project root
$ROOT_DIR = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT_DIR

Write-Host "[1/8] Setup env & quick checks" -ForegroundColor Green
# Note: setup_local.sh is bash, we'll skip it for now
python gpu_test.py
if (Test-Path "environment_check.sh") {
    bash environment_check.sh
}

Write-Host "[2/8] Download + preprocess datasets (will only use $Dataset later)" -ForegroundColor Green
# Note: This requires bash, we'll need to run it manually or create a PowerShell version
Write-Host "Please run: bash code/dataset/run_preprocessing.sh" -ForegroundColor Yellow

Write-Host "[3/8] Train frozen SASRec ranker" -ForegroundColor Green
Set-Location "code/ranker"
# Note: This requires bash and conda environment 'rec'
Write-Host "Please run: bash run_training.sh" -ForegroundColor Yellow
Set-Location $ROOT_DIR

# resolve ranker checkpoint path
$RANKER_CKPT = "checkpoints/sasrec_${Dataset}.pt"
Write-Host "RANKER_CKPT=$RANKER_CKPT" -ForegroundColor Cyan

Write-Host "[4/8] Compute projection basis from the trained ranker" -ForegroundColor Green
Set-Location "code/projection"
python run_basis.py --dataset $Dataset --proj_dir $ROOT_DIR
$PROJ_BASIS = "checkpoints/basis_${Dataset}.pt"
Write-Host "PROJ_BASIS=$PROJ_BASIS" -ForegroundColor Cyan
Set-Location $ROOT_DIR

Write-Host "[5/8] Init LoRA (tiny settings) + load small model in 4-bit" -ForegroundColor Green
# Small model + 4-bit for 6 GB GPU
$env:RECRL_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
$env:RECRL_4BIT = "1"
$env:RECRL_MAX_GPU_MEM = "5GiB"   # keep headroom
$env:RECRL_CPU_MEM = "48GiB"

Set-Location "code/explainer"
# Note: This requires bash and conda environment 'rec'
Write-Host "Please run: bash run_lora_init.sh --int8" -ForegroundColor Yellow
Set-Location $ROOT_DIR

Write-Host "[6/8] Tiny end-to-end smoke test (Shielded RecRL)" -ForegroundColor Green
Set-Location "code/trainer"
python run_toy.py `
  --dataset $Dataset `
  --ranker_ckpt $RANKER_CKPT `
  --projection_basis "$ROOT_DIR/$PROJ_BASIS" `
  --lora_rank 8 `
  --kl_beta 0.05 `
  --max_steps 500 `
  --micro_batch_size 1 `
  --grad_accum 8 `
  --max_seq_len_explainer 192 `
  --explanation_max_len 96

Set-Location $ROOT_DIR

Write-Host "[7/8] Sanity: verify projection & CTR script can run" -ForegroundColor Green
Set-Location "code/trainer"
python check_projection.py --projection_basis "$ROOT_DIR/$PROJ_BASIS"
python verify_ctr_lift.py --dataset $Dataset --dry_run
Set-Location $ROOT_DIR

Write-Host "[8/8] Package artifacts to transfer later" -ForegroundColor Green
New-Item -ItemType Directory -Path "transfer_out/$Dataset" -Force | Out-Null
Copy-Item -Recurse "data/proc/$Dataset" "transfer_out/$Dataset/proc"
Copy-Item -Recurse "code/ranker/checkpoints/$Dataset" "transfer_out/$Dataset/ranker_ckpts"
Copy-Item $PROJ_BASIS "transfer_out/$Dataset/"

Write-Host "DONE. Artifacts in transfer_out/$Dataset" -ForegroundColor Green

