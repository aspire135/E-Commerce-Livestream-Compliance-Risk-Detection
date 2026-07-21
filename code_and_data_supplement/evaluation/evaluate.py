"""
Evaluation script. Computes AUROC, AUPRC, F1, Best F1, FPR, FNR, and latency.
Matches Section 6 experimental setup.

Metrics:
  - AUROC, AUPRC: threshold-independent ranking metrics
  - F1, FPR, FNR: computed at decision threshold τ selected on val set
  - Best F1: optimal F1 from precision-recall curve
  - Latency: per-sample wall-clock time (batch_size=1)
"""

import json, time, argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
)


def evaluate_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float = 0.50,
    latencies: np.ndarray = None,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        scores: model predictions [0,1], shape (N,)
        labels: ground-truth binary labels {0,1}, shape (N,)
        tau: decision threshold for F1/FPR/FNR
        latencies: per-sample latency in seconds, shape (N,) or None

    Returns:
        dict with all metrics
    """
    # Threshold-independent
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    # Decision-threshold metrics (τ)
    preds = (scores >= tau).astype(int)
    tn = ((labels == 0) & (preds == 0)).sum()
    fp = ((labels == 0) & (preds == 1)).sum()
    fn = ((labels == 1) & (preds == 0)).sum()
    tp = ((labels == 1) & (preds == 1)).sum()

    f1 = float(f1_score(labels, preds, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Best F1 (optimal threshold from PR curve)
    prec, rec, _ = precision_recall_curve(labels, scores)
    best_f1 = float(np.max(2 * rec * prec / (rec + prec + 1e-10)))

    result = {
        "n_samples": len(scores),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "best_f1": best_f1,
        "fpr": fpr,
        "fnr": fnr,
        "tau": tau,
    }

    if latencies is not None:
        result["latency_mean_s"] = float(np.mean(latencies))
        result["latency_std_s"] = float(np.std(latencies))
        result["latency_total_s"] = float(np.sum(latencies))

    return result


def load_scores_and_labels(scores_path: str, labels_path: str):
    """Load scores and labels from JSONL files."""
    with open(scores_path) as f:
        score_data = [json.loads(line) for line in f]
    with open(labels_path) as f:
        label_data = [json.loads(line) for line in f]

    # Align by sample_id
    label_map = {d["sample_id"]: d["label"] for d in label_data}
    scores = []
    labels = []
    latencies = []
    for d in score_data:
        sid = d["sample_id"]
        if sid in label_map:
            scores.append(d.get("score", d.get("cmr_score", 0.5)))
            labels.append(label_map[sid])
            latencies.append(d.get("latency_s", 0.0))

    return np.array(scores), np.array(labels), np.array(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    scores, labels, latencies = load_scores_and_labels(args.scores, args.labels)
    metrics = evaluate_predictions(scores, labels, args.tau, latencies)

    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
