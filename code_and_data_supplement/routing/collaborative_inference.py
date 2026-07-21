"""
Risk-aware collaborative inference. Section 5.5, Eq. 16-17.

LRE screens each sample first. Samples with r_LRE < θ retain the LRE score.
Samples with r_LRE ≥ θ are routed to CMR for full multimodal reasoning.

Parameters selected on validation set: θ=0.30, τ=0.75.
"""

import json, time, argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
)


def collaborative_inference(
    lre_scores: np.ndarray,
    cmr_scores: np.ndarray,
    theta: float = 0.30,
) -> np.ndarray:
    """
    Eq. 16: Collaborative routing.

    Args:
        lre_scores: LRE risk predictions, shape (N,), values in [0,1]
        cmr_scores: CMR risk predictions, shape (N,), values in [0,1]
        theta: routing threshold (selected on validation set)

    Returns:
        final_scores: collaborative risk scores, shape (N,)
    """
    return np.where(lre_scores < theta, lre_scores, cmr_scores)


def evaluate_collaborative(
    lre_scores: np.ndarray,
    cmr_scores: np.ndarray,
    labels: np.ndarray,
    theta: float = 0.30,
    tau: float = 0.75,
) -> dict:
    """
    Full evaluation of collaborative inference.

    Args:
        lre_scores, cmr_scores: model predictions [0,1]
        labels: ground-truth binary labels {0,1}
        theta: routing threshold
        tau: decision threshold for binary prediction

    Returns:
        metrics dict with AUROC, AUPRC, F1, FPR, FNR
    """
    final_scores = collaborative_inference(lre_scores, cmr_scores, theta)

    # AUROC and AUPRC
    auroc = float(roc_auc_score(labels, final_scores))
    auprc = float(average_precision_score(labels, final_scores))

    # F1, FPR, FNR at decision threshold tau
    preds = (final_scores >= tau).astype(int)
    tn = ((labels == 0) & (preds == 0)).sum()
    fp = ((labels == 0) & (preds == 1)).sum()
    fn = ((labels == 1) & (preds == 0)).sum()
    tp = ((labels == 1) & (preds == 1)).sum()

    f1 = float(f1_score(labels, preds))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Best F1
    prec, rec, _ = precision_recall_curve(labels, final_scores)
    best_f1 = float(np.max(2 * rec * prec / (rec + prec + 1e-10)))

    # Routing statistics
    safe_mask = lre_scores < theta
    n_safe = safe_mask.sum()
    n_routed = (~safe_mask).sum()

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "best_f1": best_f1,
        "fpr": fpr,
        "fnr": fnr,
        "theta": theta,
        "tau": tau,
        "n_safe": int(n_safe),
        "n_routed": int(n_routed),
        "route_rate": float(n_routed / len(labels)),
    }


def threshold_sweep(lre_scores, cmr_scores, labels, step=0.05):
    """Sweep routing threshold θ and return AUROC for each value."""
    results = []
    for theta in np.arange(0.0, 1.0 + step, step):
        final = collaborative_inference(lre_scores, cmr_scores, theta)
        auroc = roc_auc_score(labels, final)
        route_rate = (lre_scores >= theta).mean()
        results.append({"theta": round(float(theta), 2), "auroc": float(auroc), "route_rate": float(route_rate)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lre_scores", type=str, required=True,
                        help="JSONL with LRE predictions")
    parser.add_argument("--cmr_scores", type=str, required=True,
                        help="JSONL with CMR predictions")
    parser.add_argument("--labels", type=str, required=True,
                        help="JSONL with ground-truth labels")
    parser.add_argument("--theta", type=float, default=0.30)
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="Run threshold sensitivity sweep")
    args = parser.parse_args()

    # Load data
    def load_jsonl(path, key):
        with open(path) as f:
            return np.array([json.loads(line)[key] for line in f])

    lre_scores = load_jsonl(args.lre_scores, "lre_score")
    cmr_scores = load_jsonl(args.cmr_scores, "cmr_score")
    labels = load_jsonl(args.labels, "label")

    if args.sweep:
        results = threshold_sweep(lre_scores, cmr_scores, labels)
        for r in results:
            print(f"θ={r['theta']:.2f}  AUROC={r['auroc']:.4f}  route_rate={r['route_rate']:.2%}")
        return

    metrics = evaluate_collaborative(lre_scores, cmr_scores, labels, args.theta, args.tau)
    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
