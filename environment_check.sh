#!/bin/bash
# Environment validation script for Shielded RecRL

echo "=== System Information ==="
cat /etc/os-release | grep PRETTY_NAME

echo -e "\n=== GPU Information ==="
nvidia-smi | head -n 3

echo -e "\n=== Memory Information ==="
free -h | grep Mem:

echo -e "\n=== Disk Space ==="
df -h /workspace | tail -1

echo -e "\n=== Conda Information ==="
/opt/conda/bin/conda --version
conda env list | grep rec

echo -e "\n=== Python Environment ==="
python --version
pip --version

echo -e "\n=== Running GPU Tests ==="
python gpu_test.py

echo -e "\n=== Environment Check Complete ==="
