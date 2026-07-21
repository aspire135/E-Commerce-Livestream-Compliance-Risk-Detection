"""
LRE inference for test samples. Section 5.4, Eq. 5-8.

Usage:
  python lre/inference.py --config configs/lre.yaml \
      --input data/split_metadata/test.jsonl \
      --output outputs/lre_scores.jsonl
"""

import json, time, argparse, yaml
import numpy as np
import torch
from transformers import BertTokenizer
from model import LightweightRiskEstimator
from dataset import ELCDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda:0"))

    # Load model
    tokenizer = BertTokenizer.from_pretrained(cfg["bert_model"])
    model = LightweightRiskEstimator(
        pretrained_bert=cfg["bert_model"],
        proj_dim=cfg.get("proj_dim", 4096),
    ).to(device)

    ckpt = torch.load(cfg["output_ckpt"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    # Load samples
    with open(args.input) as f:
        samples = [json.loads(line) for line in f]

    dataset = ELCDataset(samples, tokenizer)

    results = []
    for i in range(len(dataset)):
        item = dataset[i]
        ids = item["input_ids"].unsqueeze(0).to(device)
        mask = item["attention_mask"].unsqueeze(0).to(device)
        prop_seq = item["propagation_seq"].unsqueeze(0).to(device)
        out_seq = item["outcome_seq"].unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            r_lre = model(ids, mask, prop_seq, out_seq, mode="eval").item()
        latency = time.perf_counter() - t0

        results.append({
            "sample_id": samples[i]["sample_id"],
            "lre_score": float(r_lre),
            "latency_s": float(latency),
        })

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"LRE inference: {len(results)} samples → {args.output}")
    print(f"Avg latency: {np.mean([r['latency_s'] for r in results])*1000:.2f} ms")


if __name__ == "__main__":
    main()
