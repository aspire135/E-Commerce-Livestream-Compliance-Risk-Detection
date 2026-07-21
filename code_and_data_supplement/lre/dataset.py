"""
LRE Dataset for CPO evidence sequences.
Loads precomputed propagation (12×6) and outcome (12×4) window features
together with ASR + comment text and CMR risk scores.

Input format: JSONL with fields:
  {
    "sample_id": str,
    "asr_text": str,
    "comment_text": str,
    "propagation_seq": [[n,d,q,δ_P,ρ_P,I_P], ...],   # 12 windows × 6 features
    "outcome_seq":      [[s,Δs,δ_O,ρ_O], ...],       # 12 windows × 4 features
    "cmr_score": float,                                # CMR risk score ∈ [0,1]
    "cmr_embedding": [float, ...],                     # 4096-dim all-MiniLM-L6-v2 encoding of CMR response
  }
"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ELCDataset(Dataset):
    """ELCAD-Bench dataset with CPO evidence for LRE training."""

    def __init__(
        self,
        samples: list,
        tokenizer: BertTokenizer,
        max_text_len: int = 256,
        num_windows: int = 12,
        prop_dim: int = 6,
        out_dim: int = 4,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.num_windows = num_windows
        self.prop_dim = prop_dim
        self.out_dim = out_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Text encoding (Eq. 5): [ASR; comments]
        asr = s.get("asr_text", "") or ""
        cmt = s.get("comment_text", "") or ""
        text = f"ASR: {asr} [SEP] {cmt}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # Propagation sequence (Eq. 1-4): 12 windows × 6 features [n,d,q,δ_P,ρ_P,I_P]
        prop_seq = self._pad_sequence(
            s.get("propagation_seq", []),
            self.num_windows,
            self.prop_dim,
        )

        # Outcome sequence (Eq. 10-12): 12 windows × 4 features [s,Δs,δ_O,ρ_O]
        out_seq = self._pad_sequence(
            s.get("outcome_seq", []),
            self.num_windows,
            self.out_dim,
        )

        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "propagation_seq": torch.tensor(prop_seq, dtype=torch.float),
            "outcome_seq": torch.tensor(out_seq, dtype=torch.float),
            "cmr_score": torch.tensor(s.get("cmr_score", 0.5), dtype=torch.float),
            "cmr_embedding": torch.tensor(
                s.get("cmr_embedding", [0.0] * 4096), dtype=torch.float
            ),
        }

    @staticmethod
    def _pad_sequence(seq, target_len, feat_dim):
        """Pad or truncate a variable-length sequence to target_len windows."""
        result = []
        for w in seq[:target_len]:
            feat = []
            for key in w if isinstance(w, dict) else range(len(w)):
                feat.append(float(w[key]) if isinstance(w, dict) else float(w[key]))
            while len(feat) < feat_dim:
                feat.append(0.0)
            result.append(feat[:feat_dim])
        while len(result) < target_len:
            result.append([0.0] * feat_dim)
        return result
