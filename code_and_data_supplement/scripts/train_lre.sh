#!/bin/bash
# Train the Lightweight Risk Estimator (LRE).
# Produces outputs/lre_best.pt

set -e

python lre/train.py --config configs/lre.yaml
echo "LRE training complete."
