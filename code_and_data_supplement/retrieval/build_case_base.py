"""
Build the CPO case base from training data.
Each case: serialized CPO evidence + binary label + annotation rationale.

Requires preprocessed samples with:
  - sample_id, session_id
  - asr_text, visual_caption, comment_text
  - propagation_seq, outcome_seq
  - label (0/1), rationale (optional)
"""

import json
import argparse
from evidence.construct_cpo import serialize_cpo_evidence


def build_case_base(train_samples_path: str, output_path: str):
    """
    Load training samples, serialize CPO evidence for each, and save
    the case base (without embeddings — those are built at retrieval time).
    """
    with open(train_samples_path) as f:
        samples = [json.loads(line) for line in f]

    cases = []
    for s in samples:
        evidence_text = serialize_cpo_evidence(
            asr_text=s.get("asr_text", ""),
            visual_caption=s.get("visual_caption", ""),
            comment_text=s.get("comment_text", ""),
            propagation_seq=s.get("propagation_seq", []),
            outcome_seq=s.get("outcome_seq", []),
            I_P=s.get("I_P", 0),
        )
        cases.append({
            "sample_id": s["sample_id"],
            "session_id": s["session_id"],
            "evidence_text": evidence_text,
            "label": s["label"],
            "rationale": s.get("rationale", ""),
        })

    with open(output_path, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Case base: {len(cases)} cases → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    build_case_base(args.train_samples, args.output)
