"""
Outcome evidence construction. Section 5.2, Eq. 10-12.

For each 60s sample, use W=12 non-overlapping 5s windows.
  s_{i,w}        : cumulative sales at window w end (carry-forward)
  Δs_{i,w}       : sales increment in window w
  ρ_{i,w}^O      : sales-amplification ratio
  δ_{i,w}^O      : sales-surge indicator

Parameters: α_O=8, b_O=10, ε=0.1
"""

import numpy as np


def build_outcome_windows(
    sales_records: list,     # list of {timestamp, cumulative_sales} within the 60s clip
    window_size: int = 5,     # seconds
    num_windows: int = 12,
    alpha_O: float = 8,       # surge multiplier threshold
    b_O: float = 10,          # surge absolute threshold
    epsilon: float = 0.1,     # small constant for division
):
    """
    Construct outcome window features from timestamped cumulative sales records.

    Each record is a {timestamp, cumulative_sales} dict. Values are carried forward
    from the latest available observation within each window.

    Returns:
        out_seq: list of [s, Δs, δ_O, ρ_O] for each of 12 windows
    """
    # Assign cumulative sales to windows (carry-forward)
    window_sales = np.full(num_windows, np.nan)
    for rec in sales_records:
        t = rec.get("timestamp", 0)
        w = min(int(t // window_size), num_windows - 1)
        if np.isnan(window_sales[w]) or rec["cumulative_sales"] > window_sales[w]:
            window_sales[w] = rec["cumulative_sales"]

    # Forward-fill NaN values
    last_val = 0.0
    for w in range(num_windows):
        if not np.isnan(window_sales[w]):
            last_val = window_sales[w]
        window_sales[w] = last_val

    # Compute increments (Eq. 10)
    increments = np.zeros(num_windows)
    for w in range(1, num_windows):
        increments[w] = max(0.0, window_sales[w] - window_sales[w - 1])

    # Contextual baseline (Eq. 11)
    out_seq = []
    for w in range(num_windows):
        s = window_sales[w]                                    # cumulative sales
        ds = increments[w]                                     # increment

        other_incs = [increments[u] for u in range(num_windows) if u != w]
        ds_bar = np.mean(other_incs) if other_incs else 0.0

        # Amplification ratio (Eq. 12)
        rho = ds / (ds_bar + epsilon)

        # Surge indicator (Eq. 12)
        delta = 1 if ds > max(alpha_O * ds_bar, b_O) else 0

        out_seq.append([float(s), float(ds), delta, rho])

    return out_seq
