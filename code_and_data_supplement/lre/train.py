"""
LRE Training with risk-score distillation and reasoning-semantic alignment.
Matches Section 5.4, Eq. 13-15.

Loss: L = L_score + λ_align * L_align
  L_score  = MSE(r_LRE, r_CMR)                                    (Eq. 13)
  L_align  = 1 - cos(W_p·h_LRE, z_CMR)                            (Eq. 14-15)
"""

import os, json, yaml, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score

from model import LightweightRiskEstimator
from dataset import ELCDataset


def cosine_alignment_loss(z_lre, z_cmr):
    """Eq. 15: 1 - cosine_similarity."""
    z_lre_norm = nn.functional.normalize(z_lre, dim=1)
    z_cmr_norm = nn.functional.normalize(z_cmr, dim=1)
    cos_sim = (z_lre_norm * z_cmr_norm).sum(dim=1)
    return (1.0 - cos_sim).mean()


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda:0"))

    # Load data
    train_samples = load_jsonl(cfg["train_path"])
    val_samples = load_jsonl(cfg["val_path"])

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg["bert_model"])

    # Datasets & loaders
    train_ds = ELCDataset(train_samples, tokenizer)
    val_ds = ELCDataset(val_samples, tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"])

    # Model
    model = LightweightRiskEstimator(
        pretrained_bert=cfg["bert_model"],
        proj_dim=cfg.get("proj_dim", 4096),
    ).to(device)

    # Optimizer: BERT layers smaller LR
    bert_trainable = set()
    for p in model.text_encoder.encoder.layer[-2:].parameters():
        bert_trainable.add(id(p))
    for p in model.text_encoder.pooler.parameters():
        bert_trainable.add(id(p))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if id(p) in bert_trainable or not p.requires_grad:
            continue
        (no_decay if ("bias" in n or "LayerNorm" in n or "norm" in n) else decay).append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "lr": cfg["lr"], "weight_decay": 0},
        {"params": model.text_encoder.encoder.layer[-2:].parameters(),
         "lr": cfg["bert_lr"], "weight_decay": 0},
        {"params": model.text_encoder.pooler.parameters(),
         "lr": cfg["bert_lr"], "weight_decay": 0},
    ], lr=cfg["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["patience"], factor=0.5,
    )

    lambda_align = cfg.get("lambda_align", 0.3)
    loss_mse = nn.MSELoss()

    best_val_auc = 0.0
    best_ckpt = None

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            prop_seq = batch["propagation_seq"].to(device)
            out_seq = batch["outcome_seq"].to(device)
            cmr_score = batch["cmr_score"].to(device)
            cmr_emb = batch["cmr_embedding"].to(device)

            r_lre, z_lre = model(ids, mask, prop_seq, out_seq, mode="train")

            # Eq. 13: risk-score distillation
            loss_score = loss_mse(r_lre, cmr_score)

            # Eq. 14-15: reasoning-semantic alignment
            loss_align = cosine_alignment_loss(z_lre, cmr_emb)

            loss = loss_score + lambda_align * loss_align
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                prop_seq = batch["propagation_seq"].to(device)
                out_seq = batch["outcome_seq"].to(device)
                r_lre = model(ids, mask, prop_seq, out_seq, mode="eval")
                preds.extend(r_lre.cpu().numpy().tolist())
                # Labels from cmr_score for val monitoring
                labels.extend(batch["cmr_score"].cpu().numpy().tolist())

        val_auc = roc_auc_score(
            [1 if l > 0.5 else 0 for l in labels],
            preds,
        )
        scheduler.step(total_loss / len(train_dl))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_ckpt = model.state_dict()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | loss={total_loss/len(train_dl):.4f} | val_auc={val_auc:.4f}")

    # Save best checkpoint
    os.makedirs(os.path.dirname(cfg["output_ckpt"]), exist_ok=True)
    torch.save(best_ckpt, cfg["output_ckpt"])
    print(f"Best val AUROC: {best_val_auc:.4f} → saved to {cfg['output_ckpt']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
