"""
Construct 60s clips with W=12 five-second windows. Section 5.2.

Input: raw per-session data (ASR utterances, comments with timestamps,
       cumulative sales records).
Output: per-clip JSONL with:
  - clip_id, session_id
  - asr_text (concatenated utterances)
  - comment_text (temporal-order serialized)
  - propagation_seq: [[n,d,q,δ_P,ρ_P,I_P], ...]  (12 windows)
  - outcome_seq:      [[s,Δs,δ_O,ρ_O], ...]      (12 windows)
  - label, rationale
"""

import json, argparse
from evidence.propagation_features import build_propagation_windows
from evidence.outcome_features import build_outcome_windows


def construct_clip_windows(
    session_data: dict,
    clip_start: float,
    clip_end: float,
    window_size: int = 5,
    num_windows: int = 12,
) -> dict:
    """
    Construct windows for a single 60s clip from session data.

    Args:
        session_data: dict with keys: asr_utterances, comments, sales_records
        clip_start, clip_end: clip boundaries in seconds
        window_size: window duration (5s)
        num_windows: number of windows (12)

    Returns:
        clip dict with asr_text, comment_text, propagation_seq, outcome_seq
    """
    # Filter ASR utterances in clip
    asr_lines = []
    for u in session_data.get("asr_utterances", []):
        t = u.get("timestamp", 0)
        if clip_start <= t < clip_end:
            asr_lines.append(u.get("text", ""))
    asr_text = " ".join(asr_lines)

    # Filter comments in clip
    clip_comments = []
    for c in session_data.get("comments", []):
        t = c.get("timestamp", 0)
        if clip_start <= t < clip_end:
            clip_comments.append(c)

    # Comment text in temporal order
    clip_comments.sort(key=lambda x: x.get("timestamp", 0))
    comment_text = " ".join(c.get("text", "") for c in clip_comments)

    # Filter sales records in clip
    clip_sales = []
    for s in session_data.get("sales_records", []):
        t = s.get("timestamp", 0)
        if clip_start <= t < clip_end:
            clip_sales.append(s)
    clip_sales.sort(key=lambda x: x.get("timestamp", 0))

    # Build window features
    propagation_seq = build_propagation_windows(
        clip_comments, window_size, num_windows,
    )
    outcome_seq = build_outcome_windows(
        clip_sales, window_size, num_windows,
    )

    return {
        "asr_text": asr_text,
        "comment_text": comment_text,
        "propagation_seq": propagation_seq,
        "outcome_seq": outcome_seq,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Constructing windows: window_size={cfg['window_size']}s, num_windows={cfg['num_windows']}")
    # ... (load session data, split into clips, construct windows, save)


if __name__ == "__main__":
    main()
