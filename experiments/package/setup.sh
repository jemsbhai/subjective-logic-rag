#!/bin/bash
# Setup script for A100 experiment runs
# Run this once on the A100 machine before running experiments.

set -e

# Ensure we're at the repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== SL-RAG A100 Experiment Setup ==="
echo "Working directory: $REPO_ROOT"
echo ""

# 1. Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# 3. Download models
echo "[3/4] Downloading models..."
mkdir -p models

# Llama 3.1 70B (primary — ~140GB)
if [ ! -d "models/Llama-3.1-70B-Instruct" ]; then
    echo "  Downloading Llama 3.1 70B Instruct..."
    hf download meta-llama/Llama-3.1-70B-Instruct \
        --local-dir models/Llama-3.1-70B-Instruct \
        --exclude "original/*"
    echo "  Done."
else
    echo "  Llama 3.1 70B already present."
fi

# Qwen 2.5 72B (secondary — if time permits, ~150GB)
if [ ! -d "models/Qwen2.5-72B-Instruct" ]; then
    echo "  Downloading Qwen 2.5 72B Instruct..."
    hf download Qwen/Qwen2.5-72B-Instruct \
        --local-dir models/Qwen2.5-72B-Instruct
    echo "  Done."
else
    echo "  Qwen 2.5 72B already present."
fi

# 4. Verify installation
echo "[4/4] Verifying installation..."
python -c "
from xrag.experiment.config import load_config
from xrag.experiment.runner import run_experiment
print('Import OK')
"

echo ""
echo "=== Setup complete ==="
echo "Run experiments with: bash experiments/package/run_a100.sh"
