# Windows Setup Guide for Shielded RecRL

This guide will help you set up the Shielded RecRL project on Windows without using bash scripts.

## Prerequisites

1. **Python 3.8+**: Download from [python.org](https://python.org/)
2. **Git**: Download from [git-scm.com](https://git-scm.com/)
3. **CUDA Toolkit** (optional, for GPU support): Download from [nvidia.com](https://developer.nvidia.com/cuda-downloads)

## Option 1: Automated Setup (Recommended)

### Using PowerShell
```powershell
# Run PowerShell as Administrator if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows.ps1
```

### Using Batch File
```cmd
# Double-click setup_windows.bat or run in Command Prompt
setup_windows.bat
```

## Option 2: Manual Setup

### Step 1: Create Project Structure
```cmd
mkdir data
mkdir data\raw
mkdir data\proc
mkdir data\_checksums
mkdir checkpoints
mkdir logs
mkdir experiments
mkdir docs
```

### Step 2: Create .gitignore
Create a file named `.gitignore` with the following content:
```
*.pyc
__pycache__/
checkpoints/
logs/
*.pt
*.pth
*.bin
.env
.venv/
venv/
.idea/
.vscode/
*.log
```

### Step 3: Initialize Git Repository
```cmd
git init -b main
git add .gitignore
git commit -m "Init repo"
```

### Step 4: Install Python Dependencies
```cmd
pip install -r requirements.txt
```

**Note**: Some packages might require Microsoft Visual C++ Build Tools. If you encounter errors:
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Or use pre-compiled wheels: `pip install --only-binary=all -r requirements.txt`

### Step 5: Verify Installation
```cmd
python gpu_test.py
```

## Running the Project

### 1. Download Datasets
```cmd
cd code\dataset
python download_datasets.py --dataset books
```

### 2. Preprocess Data
```cmd
python preprocess.py --dataset books
```

### 3. Train Ranking Model
```cmd
cd ..\ranker
python train_sasrec.py --dataset books --epochs 50
```

### 4. Initialize Language Model
```cmd
cd ..\explainer
python init_lora.py --dataset books --int8
```

### 5. Compute Projection Basis
```cmd
cd ..\projection
python basis.py --ranker_ckpt ..\checkpoints\sasrec_books.pt --output ..\checkpoints\basis_books.pt
```

### 6. Run Training
```cmd
cd ..\trainer
python run_recrl.py --dataset books --cfg ..\..\experiments\recrl_default.yaml
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: Install CUDA Toolkit and PyTorch with CUDA support
2. **Memory errors**: Use `--int8` flag for 8-bit quantization
3. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
4. **Git errors**: Make sure Git is in your PATH

### GPU Support
To enable GPU support on Windows:
1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Alternative: Use WSL2
If you prefer a Linux environment:
1. Install WSL2 from Microsoft Store
2. Install Ubuntu on WSL2
3. Follow the original bash setup instructions

## Next Steps

After setup, you can:
1. Run experiments with different datasets (books, ml25m, steam)
2. Modify training parameters in `experiments/recrl_default.yaml`
3. Analyze results in the `logs/` directory
4. Generate evaluation reports with `code/eval/aggregate_main.py`

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your Python and CUDA versions
3. Ensure all dependencies are correctly installed
4. Check the project's GitHub issues page

