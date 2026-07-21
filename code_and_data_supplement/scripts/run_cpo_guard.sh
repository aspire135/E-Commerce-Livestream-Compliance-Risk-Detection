#!/bin/bash
# Full CPO-Guard pipeline: LRE → CMR → collaborative inference.
# Reproduces Table 1 main results.
#
# Prerequisites:
#   1. CMR case base built (see scripts/run_cmr.sh)
#   2. LRE checkpoint trained (see scripts/train_lre.sh)
#   3. SGLang server running on port 8000 (for 9B variant)

set -e

MODE=${1:-14b}
RESULT_DIR="outputs"
mkdir -p "$RESULT_DIR"

# Step 1: LRE inference
echo "[1/3] Running LRE inference..."
python lre/inference.py \
    --config configs/lre.yaml \
    --input data/split_metadata/test.jsonl \
    --output "$RESULT_DIR/lre_scores.jsonl"

# Step 2: CMR inference
echo "[2/3] Running CMR inference ($MODE)..."
bash scripts/run_cmr.sh "$MODE"

# Step 3: Collaborative inference + evaluation
echo "[3/3] Running collaborative inference..."
python routing/collaborative_inference.py \
    --lre_scores "$RESULT_DIR/lre_scores.jsonl" \
    --cmr_scores "$RESULT_DIR/cmr_${MODE}_results.jsonl" \
    --labels data/split_metadata/test.jsonl \
    --theta 0.30 --tau 0.75 \
    --output "$RESULT_DIR/collaborative_metrics.json" \
    --sweep

echo ""
echo "=== Main Results ==="
python evaluation/reproduce_tables.py --result_dir "$RESULT_DIR"
echo ""
echo "All results saved to $RESULT_DIR/"
