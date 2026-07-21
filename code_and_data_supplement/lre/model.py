"""
LRE (Lightweight Risk Estimator) model architecture.
Matches Section 5.4 of the paper.

Architecture:
  ASR + Comments → BERT-base-Chinese → h_T     [B, 768]
  Propagation seq  → BiGRU_P → h_P             [B, 64]
  Outcome seq      → BiGRU_O → h_O             [B, 64]
  Evidence-aware gate: g = σ(W_g[h_P;h_O] + b_g)
  h_T' = g ⊙ h_T
  h_fused = [h_T'; h_P; h_O]                    [B, 896]
  Bottleneck → h_LRE                             [B, 256]
  Score head: σ(FC(h_LRE))                       [B, 1]
  Projection head: FC(h_LRE)                     [B, D_proj]  (training only)
"""

import torch
import torch.nn as nn
from transformers import BertModel


class LightweightRiskEstimator(nn.Module):
    """LRE: BERT-BiGRU with evidence-aware gate and reasoning projection."""

    def __init__(
        self,
        pretrained_bert: str = "bert-base-chinese",
        prop_input_dim: int = 6,
        out_input_dim: int = 4,
        gru_hidden: int = 32,
        bottleneck_dim: int = 256,
        proj_dim: int = 4096,
        max_text_len: int = 256,
    ):
        super().__init__()

        # Text encoder (frozen bottom, last 2 layers + pooler trainable)
        self.text_encoder = BertModel.from_pretrained(pretrained_bert)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        # Propagation temporal encoder (Eq. 6)
        self.prop_gru = nn.GRU(
            input_size=prop_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Outcome temporal encoder (Eq. 6)
        self.out_gru = nn.GRU(
            input_size=out_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Evidence-aware gate (Eq. 7)
        bert_dim = self.text_encoder.config.hidden_size
        joint_dim = 2 * (2 * gru_hidden)  # prop + out, both bidirectional
        self.gate = nn.Sequential(
            nn.Linear(joint_dim, bert_dim),
            nn.Sigmoid(),
        )

        # Bottleneck fusion (Eq. 8)
        fused_dim = bert_dim + joint_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(fused_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Score prediction head (Eq. 8)
        self.score_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )

        # Reasoning-semantic alignment projection (Eq. 9, training only)
        self.projection = nn.Linear(bottleneck_dim, proj_dim)

    def forward(
        self,
        text_ids,
        attention_mask,
        prop_seq,   # [B, 12, 6]
        out_seq,    # [B, 12, 4]
        mode: str = "eval",
    ):
        # Text encoding (Eq. 5)
        bert_out = self.text_encoder(
            input_ids=text_ids,
            attention_mask=attention_mask,
        )
        h_T = bert_out.last_hidden_state[:, 0, :]   # [CLS] token, [B, 768]

        # Propagation encoding (Eq. 6)
        _, h_prop_last = self.prop_gru(prop_seq)
        h_P = h_prop_last.transpose(0, 1).reshape(prop_seq.size(0), -1)  # [B, 64]

        # Outcome encoding (Eq. 6)
        _, h_out_last = self.out_gru(out_seq)
        h_O = h_out_last.transpose(0, 1).reshape(out_seq.size(0), -1)    # [B, 64]

        # Evidence-aware gate (Eq. 7)
        h_joint = torch.cat([h_P, h_O], dim=-1)       # [B, 128]
        gate = self.gate(h_joint)                       # [B, 768]
        h_T_gated = gate * h_T                          # [B, 768]

        # Fused representation (Eq. 8)
        h_fused = torch.cat([h_T_gated, h_joint], dim=-1)  # [B, 896]
        h_LRE = self.bottleneck(h_fused)                     # [B, 256]

        # Risk score (Eq. 8)
        r_LRE = self.score_head(h_LRE).squeeze(-1)           # [B]

        if mode == "train":
            z_LRE = self.projection(h_LRE)                   # [B, D_proj]
            return r_LRE, z_LRE

        return r_LRE
