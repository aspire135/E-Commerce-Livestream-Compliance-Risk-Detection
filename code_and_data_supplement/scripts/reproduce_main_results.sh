#!/bin/bash
# Reproduce all main results in the paper.
#
# Tables produced:
#   Table 1: Main results (LRE, CMR, CPO-Guard)
#   Table 2: Ablation results (modality + component)
#   Supplementary: threshold sensitivity
#
# Usage: bash scripts/reproduce_main_results.sh

set -e

echo "======================================"
echo " CPO-Guard — Reproducing Main Results"
echo "======================================"

# 1. Train LRE
echo ""
echo "[Step 1] Training LRE..."
bash scripts/train_lre.sh

# 2. Run full CPO-Guard (14B)
echo ""
echo "[Step 2] Running CPO-Guard with Qwen2.5-14B..."
bash scripts/run_cpo_guard.sh 14b

# 3. Run full CPO-Guard (9B)
echo ""
echo "[Step 3] Running CPO-Guard with Qwen3.5-9B..."
bash scripts/run_cpo_guard.sh 9b

echo ""
echo "======================================"
echo " All results reproduced."
echo " See outputs/ for metrics and figures."
echo "======================================"
