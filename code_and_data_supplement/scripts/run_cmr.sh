#!/bin/bash
# Run CMR inference with retrieval-augmented CPO reasoning.
# Uses Qwen2.5-14B (local HF) or Qwen3.5-9B (SGLang API).
#
# Usage: bash scripts/run_cmr.sh [14b|9b]

MODE=${1:-14b}

if [ "$MODE" == "14b" ]; then
    CONFIG="configs/cmr_qwen25_14b.yaml"
elif [ "$MODE" == "9b" ]; then
    CONFIG="configs/cmr_qwen35_9b.yaml"
else
    echo "Usage: bash scripts/run_cmr.sh [14b|9b]"
    exit 1
fi

# Step 1: Build case base from training data (one-time)
python retrieval/build_case_base.py \
    --train_samples data/split_metadata/train.jsonl \
    --output data/case_base.jsonl

# Step 2: Run CMR on test samples
python cmr/inference.py \
    --config "$CONFIG" \
    --input data/split_metadata/test.jsonl \
    --output outputs/cmr_${MODE}_results.jsonl \
    --case_base data/case_base.jsonl

echo "CMR ($MODE) inference complete."
