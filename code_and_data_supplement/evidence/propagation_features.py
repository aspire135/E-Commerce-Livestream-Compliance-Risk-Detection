"""
Propagation evidence construction. Section 5.2, Eq. 1-4.

For each 60s sample, divide into W=12 non-overlapping 5s windows.
Each window w computes:
  n_{i,w}   : comment count
  d_{i,w}   : duplicate-comment ratio
  q_{i,w}   : template-matching ratio
  I_i^P      : coordinated interaction indicator (sample-level)
  ρ_{i,w}^P : interaction-amplification ratio
  δ_{i,w}^P : comment-surge indicator

Parameters: λ_I=0.8, α_P=20, b_P=20, ε=0.1
"""

import numpy as np


def build_propagation_windows(
    comments: list,          # list of {timestamp, text} within the 60s clip
    window_size: int = 5,    # seconds
    num_windows: int = 12,
    lambda_I: float = 0.8,   # coordinated interaction threshold
    alpha_P: float = 20,     # surge multiplier threshold
    b_P: float = 20,         # surge absolute threshold
    epsilon: float = 0.1,    # small constant for division
):
    """
    Construct propagation window features from timestamped comments.

    Returns:
        prop_seq: list of [n, d, q, δ_P, ρ_P, I_P] for each of 12 windows
    """
    # Partition comments into windows
    window_comments = [[] for _ in range(num_windows)]
    for c in comments:
        t = c.get("timestamp", 0)
        w = min(int(t // window_size), num_windows - 1)
        window_comments[w].append(c.get("text", ""))

    # Window-level features
    prop_seq = []
    for w in range(num_windows):
        texts = window_comments[w]
        n = len(texts)                                         # comment count

        # Duplicate ratio d: fraction of comments repeated verbatim
        unique = set(texts)
        d = 1.0 - (len(unique) / max(n, 1)) if n > 0 else 0.0

        # Template ratio q: placeholder (requires external template matching)
        q = 0.0  # computed by external Chinese-RoBERTa template classifier

        # Contextual baseline (Eq. 2)
        other_counts = [len(window_comments[u]) for u in range(num_windows) if u != w]
        n_bar = np.mean(other_counts) if other_counts else 0.0

        # Amplification ratio (Eq. 3)
        rho = n / (n_bar + epsilon)

        # Surge indicator (Eq. 4)
        delta = 1 if n > max(alpha_P * n_bar, b_P) else 0

        prop_seq.append([n, d, q, delta, rho])

    # Sample-level coordinated interaction indicator (Eq. 1)
    # Computed by external Chinese-RoBERTa classifier with 20+20 templates
    I_P = 0  # placeholder — requires the template classifier

    # Append I_P to each window (sample-level feature replicated)
    for w in range(num_windows):
        prop_seq[w].append(I_P)

    return prop_seq
