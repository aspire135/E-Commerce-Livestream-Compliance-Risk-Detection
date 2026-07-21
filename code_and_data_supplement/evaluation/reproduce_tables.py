"""
Reproduce main results and ablation tables. Section 6.

Produces:
  Table 1: Main results (LRE, CMR, CPO-Guard vs baselines)
  Table 2: Ablation results (modality and component)
  Supplementary: Threshold sensitivity analysis
"""

import json, argparse, yaml
import numpy as np
from evaluation.evaluate import evaluate_predictions
from routing.collaborative_inference import evaluate_collaborative, threshold_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing CMR, LRE, and label JSONL files")
    parser.add_argument("--theta", type=float, default=0.30)
    parser.add_argument("--tau", type=float, default=0.75)
    args = parser.parse_args()

    # Load results from result_dir
    def _load(path, key):
        with open(path) as f:
            return np.array([json.loads(line)[key] for line in f])

    import os
    labels = _load(os.path.join(args.result_dir, "labels.jsonl"), "label")

    # ── Table 1: Main Results ──
    print("=" * 70)
    print("Table 1: Main Results on ELCAD-Bench")
    print("=" * 70)
    print(f"{'Method':<25s} {'AUROC':>8s} {'AUPRC':>8s} {'F1':>8s} {'FPR':>8s} {'FNR':>8s}")
    print("-" * 70)

    methods = {
        "LRE": ("lre_scores.jsonl", "lre_score"),
        "CMR (Qwen3.5-9B)": ("cmr_9b_scores.jsonl", "cmr_score"),
        "CPO-Guard (Qwen3.5-9B)": ("cmr_9b_scores.jsonl", "cmr_score"),
        "CMR (Qwen2.5-14B)": ("cmr_14b_scores.jsonl", "cmr_score"),
        "CPO-Guard (Qwen2.5-14B)": ("cmr_14b_scores.jsonl", "cmr_score"),
    }

    lre_file = os.path.join(args.result_dir, "lre_scores.jsonl")
    if os.path.exists(lre_file):
        lre_scores = _load(lre_file, "lre_score")
        m = evaluate_predictions(lre_scores, labels, args.tau)
        print(f"{'LRE':<25s} {m['auroc']:8.4f} {m['auprc']:8.4f} {m['f1']:8.4f} {m['fpr']:8.4f} {m['fnr']:8.4f}")

    for cmr_name, cmr_file_key in [("CMR (Qwen2.5-14B)", "cmr_14b"), ("CMR (Qwen3.5-9B)", "cmr_9b")]:
        cmr_file = os.path.join(args.result_dir, f"{cmr_file_key}_scores.jsonl")
        if os.path.exists(cmr_file):
            cmr_scores = _load(cmr_file, "cmr_score")
            m_cmr = evaluate_predictions(cmr_scores, labels, args.tau)
            print(f"{cmr_name:<25s} {m_cmr['auroc']:8.4f} {m_cmr['auprc']:8.4f} {m_cmr['f1']:8.4f} {m_cmr['fpr']:8.4f} {m_cmr['fnr']:8.4f}")

            if os.path.exists(lre_file):
                lre_scores = _load(lre_file, "lre_score")
                m_cas = evaluate_collaborative(lre_scores, cmr_scores, labels, args.theta, args.tau)
                print(f"{'CPO-Guard (' + cmr_name.split('(')[1] + ' ('):<25s} {m_cas['auroc']:8.4f} {m_cas['auprc']:8.4f} {m_cas['f1']:8.4f} {m_cas['fpr']:8.4f} {m_cas['fnr']:8.4f}")

    # ── Threshold Sensitivity ──
    print()
    print("=" * 70)
    print("Supplementary: Threshold Sensitivity (θ sweep)")
    print("=" * 70)

    cmr_14b_file = os.path.join(args.result_dir, "cmr_14b_scores.jsonl")
    if os.path.exists(lre_file) and os.path.exists(cmr_14b_file):
        lre_scores = _load(lre_file, "lre_score")
        cmr_scores = _load(cmr_14b_file, "cmr_score")
        results = threshold_sweep(lre_scores, cmr_scores, labels)
        for r in results:
            print(f"  θ={r['theta']:.2f}  AUROC={r['auroc']:.4f}  route_rate={r['route_rate']:.2%}")


if __name__ == "__main__":
    main()
