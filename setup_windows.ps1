# Windows PowerShell setup script for Shielded RecRL project

Write-Host "Setting up Shielded RecRL project on Windows..." -ForegroundColor Green

# Check if Git is installed
try {
    git --version | Out-Null
    Write-Host "Git is installed" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git from https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# Check if Python is installed
try {
    python --version | Out-Null
    Write-Host "Python is installed" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed. Please install Python from https://python.org/" -ForegroundColor Red
    exit 1
}

# Create .gitignore file
Write-Host "Creating .gitignore file..." -ForegroundColor Yellow
@"
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
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8

# Initialize Git repository
Write-Host "Initializing Git repository..." -ForegroundColor Yellow
git init -b main
git add .gitignore
git commit -m "Init repo"

# Prompt for GitHub username
$githubUsername = Read-Host "Enter your GitHub username (or press Enter to skip remote setup)"
if ($githubUsername -and $githubUsername -ne "") {
    Write-Host "Setting up GitHub remote..." -ForegroundColor Yellow
    git remote add origin "git@github.com:$githubUsername/shielded-recrl.git"
    Write-Host "Note: You'll need to push manually with: git push -u origin main" -ForegroundColor Yellow
}

# Add requirements.txt
if (Test-Path "requirements.txt") {
    git add requirements.txt
    git commit -m "Add requirements"
    Write-Host "Added requirements.txt to Git" -ForegroundColor Green
}

# Create necessary directories
Write-Host "Creating project directories..." -ForegroundColor Yellow
$directories = @(
    "data",
    "data/raw",
    "data/proc", 
    "data/_checksums",
    "checkpoints",
    "logs",
    "experiments",
    "docs"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Install Python dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "2. Download datasets: python code/dataset/download_datasets.py --dataset books" -ForegroundColor White
Write-Host "3. Preprocess data: python code/dataset/preprocess.py --dataset books" -ForegroundColor White

