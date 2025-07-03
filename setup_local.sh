#!/bin/bash
# Local setup script for Shielded RecRL project

# Create GitHub repository
mkdir -p ~/code/shielded-recrl
cd ~/code/shielded-recrl
git init -b main
echo -e "*.pyc\n__pycache__/\ncheckpoints/\nlogs/" > .gitignore
git add .gitignore
git commit -m "Init repo"

# Replace with your GitHub username
GITHUB_USERNAME="your_username"

# Add remote and push
git remote add origin git@github.com:${GITHUB_USERNAME}/shielded-recrl.git
git push -u origin main

# Add requirements.txt
git add requirements.txt
git commit -m "Add requirements"
git push

echo "Local setup complete. Please update GITHUB_USERNAME in the script before running."
