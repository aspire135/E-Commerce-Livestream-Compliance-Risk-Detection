# HCRP Inference for E-Commerce Livestream Compliance Detection

This repository contains the main inference script for the paper:

**Mechanism-Aligned Multimodal Evidence Reasoning for E-Commerce Livestream Compliance Detection**

The code implements the HCRP inference pipeline used to score one-minute e-commerce livestream clips using mechanism-aligned evidence streams:

- **ASR transcripts** for Cause-level evidence
- **Live comments / interaction signals** for Propagation-level evidence
- **Sales time series** for Outcome-level evidence

Raw videos are not used as primary evidence in the released inference script.

## Repository Contents

```text
.
├── run_hcrp_release.py
└── README.md
```

This anonymous repository intentionally contains only the main release script.  
The dataset is not included in this repository because it contains livestream-derived user comments, ASR transcripts, sales records, and other platform-specific data that require anonymization and release review.

## Requirements

Recommended environment:

```bash
python >= 3.9
torch
transformers
scikit-learn
numpy
tqdm
```

Install dependencies with:

```bash
pip install torch transformers scikit-learn numpy tqdm
```

A local or HuggingFace-compatible instruction-tuned language model is required. For example:

```text
Qwen/Qwen2.5-14B-Instruct
```

## Expected Inputs

The script expects three main input files and one optional file.

### 1. Annotation file

A whitespace-separated text file:

```text
<clip_id> <label>
```

Example:

```text
clip_000001 0
clip_000002 1
```

where `label=1` indicates abnormal / compliance-risk and `label=0` indicates normal.

### 2. ASR JSONL file

Each line should contain one clip-level ASR record:

```json
{
  "video_stem": "clip_000001",
  "audio_analysis": [
    {"start": 0.5, "text": "example host utterance", "speed": 4.2}
  ]
}
```

### 3. Comment / sales JSONL file

Each line should contain one clip-level record with 5-second aggregated comment and sales signals:

```json
{
  "video_stem": "clip_000001",
  "crowd_analysis": [
    {
      "时间窗口": "0-5s",
      "评论总数": 12,
      "销量变化数": 3,
      "销量平均增速": 0.4,
      "评论平均增速": 0.8,
      "顶部评论": ["example comment"]
    }
  ]
}
```

English key names are also partially supported by the script:

```json
{
  "time_window": "0-5s",
  "comment_count": 12,
  "sales_delta": 3,
  "sales_avg_speed": 0.4,
  "comment_avg_speed": 0.8,
  "top_comments": ["example comment"]
}
```

### 4. Optional spam-interaction report

A JSON file containing a list of spam / coordinated-interaction reports:

```json
[
  {
    "video_stem": "clip_000001",
    "abnormal_rate": 0.82,
    "total_comments": 35,
    "abnormal_count": 29,
    "samples": ["example repeated comment"]
  }
]
```

## Running Inference

Example command:

```bash
python run_hcrp_release.py \
  --annotations data/test.txt \
  --asr data/asr_results.jsonl \
  --comments data/crowd_captions.jsonl \
  --spam data/clips_spam_report.json \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --output outputs/test_predictions.jsonl \
  --prompt-log outputs/test_prompts.txt \
  --temperature 0.0
```

The script uses deterministic decoding by default when `--temperature 0.0` is specified.

## Output Format

The prediction file is written in JSONL format:

```json
{
  "clip_id": "clip_000001",
  "gt_label": 0,
  "pred_score": 0.12,
  "parsed": {
    "cause": "...",
    "propagation": "...",
    "outcome": "...",
    "reason": "...",
    "score": 12
  },
  "llm_response": "..."
}
```

`pred_score` is normalized to `[0, 1]` and is used for AUROC and F1 computation.

## Evaluation Only

If predictions already exist, compute metrics without rerunning the model:

```bash
python run_hcrp_release.py \
  --eval-only \
  --output outputs/test_predictions.jsonl
```

The script reports:

```json
{
  "auroc": ...,
  "best_f1": ...
}
```

## Optional Retrieval

The released script supports an optional retrieval hook:

```bash
--use-rag
```

This option requires a separate `rag_manager.py` implementation providing `StreamGazeRAG`.  
If `rag_manager.py` is not included, do not enable `--use-rag`.

In the paper, retrieval-augmented case grounding is treated as a lightweight calibration component rather than the core method.

## Notes on Data Release

This repository does not include raw videos, raw user identifiers, host identifiers, shop identifiers, product URLs, private messages, payment information, or user profiles.

The anonymized benchmark dataset will be released after paper acceptance and completion of privacy review. The release will include:

- timestamped ASR transcripts
- anonymized live comments
- product-level sales time series
- binary compliance labels
- abnormality rationales
- predefined train / validation / test splits

Raw videos will not be publicly released due to privacy, portrait-right, and platform-compliance considerations.

## License

This code is provided for research use.  
Please refer to the final dataset release for dataset-specific license and usage restrictions.
