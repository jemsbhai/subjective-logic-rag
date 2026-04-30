#!/bin/bash
# Run all A100-targeted experiments.
# Prerequisites: run setup.sh first.
#
# Estimated time:
#   - Main experiments (Llama 70B clean): ~4-6 hours
#   - Corruption sweep (Llama 70B): ~8-12 hours
#   - Qwen 72B clean (if time): ~4-6 hours
#
# Total: ~16-24 hours across 3 days
#
# Results are saved to experiments/results/

set -e

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1

echo "=== SL-RAG A100 Experiments ==="
echo "Start time: $(date)"
echo ""

# ────────────────────────────────────────────────────────────────
# Priority 1: Main experiments (Llama 70B, clean)
# ────────────────────────────────────────────────────────────────

echo "=== PRIORITY 1: Main experiments (Llama 70B, clean) ==="

python -m xrag.experiment.runner \
    --config experiments/configs/a100/hotpotqa_llama70b_clean.yaml

python -m xrag.experiment.runner \
    --config experiments/configs/a100/musique_llama70b_clean.yaml

echo "Priority 1 complete: $(date)"
echo ""

# ────────────────────────────────────────────────────────────────
# Priority 2: Corruption sweep (Llama 70B)
# ────────────────────────────────────────────────────────────────

echo "=== PRIORITY 2: Corruption sweep (Llama 70B) ==="

# Generate corruption configs if not already present
if [ ! -d "experiments/configs/a100/corruption" ]; then
    echo "Generating corruption configs..."
    python experiments/scripts/generate_configs.py
fi

python experiments/scripts/run_all.py \
    --config-dir experiments/configs/a100/corruption \
    --skip-existing

echo "Priority 2 complete: $(date)"
echo ""

# ────────────────────────────────────────────────────────────────
# Priority 3: Qwen 72B (if time permits)
# ────────────────────────────────────────────────────────────────

echo "=== PRIORITY 3: Qwen 72B clean (if time permits) ==="

if [ -d "models/Qwen2.5-72B-Instruct" ]; then
    python -m xrag.experiment.runner \
        --config experiments/configs/a100/hotpotqa_qwen72b_clean.yaml

    python -m xrag.experiment.runner \
        --config experiments/configs/a100/musique_qwen72b_clean.yaml
else
    echo "Qwen 72B not downloaded — skipping."
fi

echo ""
echo "=== All A100 experiments complete ==="
echo "End time: $(date)"
echo "Results in: experiments/results/"
echo ""
echo "Please tar and send results back:"
echo "  tar czf results_a100.tar.gz experiments/results/"
