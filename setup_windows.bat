@echo off
echo Setting up Shielded RecRL project on Windows...

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is not installed. Please install Git from https://git-scm.com/
    pause
    exit /b 1
)
echo Git is installed

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python from https://python.org/
    pause
    exit /b 1
)
echo Python is installed

REM Create .gitignore file
echo Creating .gitignore file...
(
echo *.pyc
echo __pycache__/
echo checkpoints/
echo logs/
echo *.pt
echo *.pth
echo *.bin
echo .env
echo .venv/
echo venv/
echo .idea/
echo .vscode/
echo *.log
) > .gitignore

REM Initialize Git repository
echo Initializing Git repository...
git init -b main
git add .gitignore
git commit -m "Init repo"

REM Add requirements.txt if it exists
if exist requirements.txt (
    git add requirements.txt
    git commit -m "Add requirements"
    echo Added requirements.txt to Git
)

REM Create necessary directories
echo Creating project directories...
if not exist data mkdir data
if not exist data\raw mkdir data\raw
if not exist data\proc mkdir data\proc
if not exist data\_checksums mkdir data\_checksums
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist experiments mkdir experiments
if not exist docs mkdir docs

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Install Python dependencies: pip install -r requirements.txt
echo 2. Download datasets: python code/dataset/download_datasets.py --dataset books
echo 3. Preprocess data: python code/dataset/preprocess.py --dataset books
echo.
pause

