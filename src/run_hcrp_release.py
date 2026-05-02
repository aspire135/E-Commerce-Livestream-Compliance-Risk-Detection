#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run HCRP inference for e-commerce livestream compliance detection.

This script is an anonymized, reproducible version intended for release with the paper.
It removes hard-coded local paths, API keys, personal identifiers, and debugging-only code.

Expected input files:
  --annotations: whitespace-separated file with lines: <clip_id> <label>
  --asr: JSONL file, each line contains {"video_stem": ..., "audio_analysis": [...]}
  --comments: JSONL file, each line contains {"video_stem": ..., "crowd_analysis": [...]}
  --spam: optional JSON file, a list of spam-interaction reports
  --output: JSONL predictions

Optional:
  --use-rag with --rag-module if you include a separate retrieval implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


Sample = Dict[str, Any]


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_annotations(path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            clip_id, label = parts[0], parts[1]
            try:
                labels[clip_id] = int(label)
            except ValueError:
                continue
    return labels


def load_by_stem_jsonl(path: Optional[Path], stem_key: str = "video_stem") -> Dict[str, List[Dict[str, Any]]]:
    data: Dict[str, List[Dict[str, Any]]] = {}
    if path is None or not path.exists():
        return data

    for item in read_jsonl(path):
        stem = item.get(stem_key)
        if stem is None:
            continue
        data.setdefault(stem, []).append(item)
    return data


def load_spam_reports(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.exists():
        return reports

    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        stem = item.get("video_stem")
        if stem is not None:
            reports[stem] = item
    return reports


def load_samples(
    annotations: Path,
    asr: Path,
    comments: Path,
    spam: Optional[Path] = None,
) -> List[Sample]:
    labels = load_annotations(annotations)
    asr_data = load_by_stem_jsonl(asr)
    comment_data = load_by_stem_jsonl(comments)
    spam_data = load_spam_reports(spam)

    samples: List[Sample] = []
    for stem, label in labels.items():
        samples.append(
            {
                "stem": stem,
                "label": label,
                "audio": asr_data.get(stem, []),
                "crowd": comment_data.get(stem, []),
                "spam": spam_data.get(stem),
            }
        )
    return samples


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------
def _get_first_list(sample: Sample, field: str, analysis_key: str) -> List[Dict[str, Any]]:
    items = sample.get(field, [])
    if not items:
        return []
    first = items[0]
    value = first.get(analysis_key, [])
    return value if isinstance(value, list) else []


def construct_prompt(sample: Sample, retrieved_cases: Optional[List[Dict[str, Any]]] = None) -> str:
    """Construct the HCRP prompt.

    The final predictor uses mechanism-aligned evidence streams:
    ASR -> Cause, comments/spam -> Propagation, sales -> Outcome.
    Raw video is intentionally not used in the released inference prompt.
    """
    lines: List[str] = []

    lines.append("你是一名电商直播合规分析师。")
    lines.append("请根据 ASR、观众评论、销量时间序列和派生信号，判断该 60 秒片段的异常转化风险。")
    lines.append("请按 Cause → Propagation → Outcome 的顺序分析，但最终只输出 JSON。\n")

    # Cause evidence: ASR
    lines.append("1. Cause Evidence｜主播话术（ASR）:")
    audio_logs = _get_first_list(sample, "audio", "audio_analysis")
    if audio_logs:
        for entry in audio_logs:
            start = entry.get("start", "")
            text = entry.get("text", "")
            speed = entry.get("speed", 0)
            if text:
                speed_mark = " [语速较快]" if isinstance(speed, (int, float)) and speed > 6.0 else ""
                lines.append(f"- [{start}s] {text}{speed_mark}")
    else:
        lines.append("(无 ASR 数据)")
    lines.append("")

    # Propagation and outcome evidence: comments + sales in 5s windows
    lines.append("2. Propagation/Outcome Evidence｜评论与销量流水（5 秒聚合）:")
    crowd_logs = _get_first_list(sample, "crowd", "crowd_analysis")
    has_event = False

    for entry in crowd_logs:
        time_window = entry.get("时间窗口", entry.get("time_window", ""))
        comment_count = entry.get("评论总数", entry.get("comment_count", 0))
        sales_delta = entry.get("销量变化数", entry.get("sales_delta", 0))
        top_comments = entry.get("顶部评论", entry.get("top_comments", []))

        sales_baseline = float(entry.get("销量平均增速", entry.get("sales_avg_speed", 0.0)) or 0.0)
        comment_baseline = float(entry.get("评论平均增速", entry.get("comment_avg_speed", 0.0)) or 0.0)

        expected_sales = sales_baseline * 7.0
        expected_comments = comment_baseline * 10.0

        is_sales_surge = sales_delta > max(expected_sales, 10.0)
        is_comment_surge = comment_count > max(expected_comments, 5.0)

        if sales_delta == 0 and not is_comment_surge and comment_count < 3:
            continue

        has_event = True
        tags: List[str] = []
        if is_sales_surge:
            tags.append("[销量激增]")
        if is_comment_surge:
            tags.append("[互动激增]")

        sales_ratio = sales_delta / max(expected_sales, 0.1)
        comment_ratio = comment_count / max(expected_comments, 0.1)

        sales_info = f"销量+{sales_delta}"
        if sales_delta > 3 and sales_ratio >= 1.5:
            sales_info += f" (超基准 {sales_ratio:.1f} 倍)"

        comment_info = ""
        valid_comments = [str(c) for c in top_comments if isinstance(c, str) and len(c) > 1]
        if valid_comments:
            comment_info += f" | 热评: {'; '.join(valid_comments[:5])}"
        if comment_count > 3:
            comment_info += f" | 评论数:{comment_count}"
            if comment_ratio >= 2.0:
                comment_info += f" (超基准 {comment_ratio:.1f} 倍)"

        lines.append(f"- [{time_window}] {sales_info} {' '.join(tags)}{comment_info}")

    if not has_event:
        lines.append("(该片段内评论与销量均较平稳)")
    lines.append("")

    # Spam / coordinated interaction evidence
    lines.append("3. Propagation Evidence｜疑似水军或模板化互动:")
    spam_info = sample.get("spam")
    if spam_info and spam_info.get("abnormal_rate", 0) > 0.7 and spam_info.get("total_comments", 0) > 10:
        rate = float(spam_info.get("abnormal_rate", 0.0))
        count = int(spam_info.get("abnormal_count", 0))
        examples = spam_info.get("samples", [])
        examples_str = " | ".join(map(str, examples[:5])) if examples else "无样例"
        lines.append(f"- 高危预警: 异常互动占比 {rate * 100:.1f}%，异常评论数 {count}")
        lines.append(f"- 典型样例: {examples_str}")
    else:
        lines.append("(未检测到明显的高比例水军或模板化刷屏信号)")
    lines.append("")

    # Optional historical references
    if retrieved_cases:
        lines.append("4. Historical References｜历史相似案例（仅作校准参考）:")
        for i, case in enumerate(retrieved_cases[:3], start=1):
            label = "异常" if int(case.get("label", 0)) == 1 else "正常"
            desc = str(case.get("desc", ""))[:160]
            lines.append(f"- Case {i} [{label}]: {desc}")
        lines.append("注意：历史案例仅用于参考，最终判断必须基于当前片段证据。")
        lines.append("")

    # HCRP instruction
    lines.append("5. HCRP 判断规则:")
    lines.append("- Cause: 只基于 ASR 判断主播是否存在夸大宣传、医疗/功效暗示、情绪操纵、稀缺性制造或诱导刷屏。")
    lines.append("- Propagation: 基于评论、互动激增和疑似水军信号判断风险是否被观众响应或放大。")
    lines.append("- Outcome: 基于销量变化和超基准倍数判断是否出现异常转化。")
    lines.append("- 不要仅凭销量上涨判定违规；也不要因为销量未爆发而忽略明显的高风险话术。")
    lines.append("")
    lines.append("风险分数映射:")
    lines.append("- 0-30: 正常或低风险")
    lines.append("- 31-50: 边缘风险")
    lines.append("- 51-70: 高风险诱因期")
    lines.append("- 71-85: 违规扩散期")
    lines.append("- 86-100: 异常转化期")
    lines.append("")
    lines.append("请仅输出一个 JSON 对象，不要输出 Markdown 或额外解释。格式如下:")
    lines.append('{"cause": "...", "propagation": "...", "outcome": "...", "reason": "...", "score": 0}')

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Inference and parsing
# ---------------------------------------------------------------------
def parse_score(response_text: str) -> Tuple[float, Dict[str, Any]]:
    """Parse a 0-100 score from the model response and return normalized score."""
    parsed: Dict[str, Any] = {}
    score = 0.0

    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            score = float(parsed.get("score", 0.0))
        else:
            nums = re.findall(r"\d+(?:\.\d+)?", response_text)
            if nums:
                score = float(nums[-1])
    except Exception:
        score = 0.0

    score = max(0.0, min(100.0, score))
    return score / 100.0, parsed


def build_retriever(args: argparse.Namespace, train_samples: Optional[List[Sample]] = None) -> Any:
    """Optional hook for retrieval. Keeps the main script runnable without RAG."""
    if not args.use_rag:
        return None
    try:
        from rag_manager import StreamGazeRAG  # Optional local module
    except ImportError as exc:
        raise ImportError("use_rag=True but rag_manager.py was not found.") from exc

    if train_samples is None:
        train_samples = []
    retriever = StreamGazeRAG(model_name=args.embedding_model)
    retriever.build_index(train_samples)
    return retriever


def run_local_model(args: argparse.Namespace, samples: List[Sample], retriever: Any = None) -> Tuple[List[int], List[float]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    y_true: List[int] = []
    y_scores: List[float] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prompt_log:
        args.prompt_log.parent.mkdir(parents=True, exist_ok=True)

    prompt_handle = args.prompt_log.open("w", encoding="utf-8") if args.prompt_log else None

    with args.output.open("w", encoding="utf-8") as f_out:
        for sample in tqdm(samples, desc="HCRP inference"):
            retrieved_cases = retriever.retrieve(sample, k=args.top_k) if retriever else None
            prompt = construct_prompt(sample, retrieved_cases=retrieved_cases)

            if prompt_handle is not None:
                prompt_handle.write(prompt + "\n" + "=" * 80 + "\n")

            messages = [
                {"role": "system", "content": "You are a careful compliance analysis assistant."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    num_return_sequences=1,
                )

            generated = output_ids[:, model_inputs.input_ids.shape[1]:]
            response_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

            prob, parsed = parse_score(response_text)
            y_true.append(int(sample["label"]))
            y_scores.append(prob)

            f_out.write(
                json.dumps(
                    {
                        "clip_id": sample["stem"],
                        "gt_label": int(sample["label"]),
                        "pred_score": prob,
                        "parsed": parsed,
                        "llm_response": response_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_out.flush()

    if prompt_handle is not None:
        prompt_handle.close()

    return y_true, y_scores


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics(y_true: List[int], y_scores: List[float]) -> Dict[str, float]:
    if len(set(y_true)) < 2:
        raise ValueError("Ground truth contains only one class; AUROC is undefined.")

    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "auroc": float(auroc),
        "best_f1": float(np.max(f1_scores)),
    }


def load_jsonl_predictions(path: Path) -> Tuple[List[int], List[float]]:
    y_true: List[int] = []
    y_scores: List[float] = []
    for item in read_jsonl(path):
        y_true.append(int(item["gt_label"]))
        y_scores.append(float(item["pred_score"]))
    return y_true, y_scores


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HCRP inference for livestream compliance detection.")

    parser.add_argument("--annotations", type=Path, required=True, help="Annotation split file: <clip_id> <label>.")
    parser.add_argument("--asr", type=Path, required=True, help="ASR JSONL file.")
    parser.add_argument("--comments", type=Path, required=True, help="Comment/sales JSONL file.")
    parser.add_argument("--spam", type=Path, default=None, help="Optional spam-interaction report JSON file.")

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="HF model path or local model path.")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.jsonl"))
    parser.add_argument("--prompt-log", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for deterministic decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--use-rag", action="store_true", help="Enable optional retrieval-augmented case grounding.")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--eval-only", action="store_true", help="Only compute metrics from --output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.eval_only:
        y_true, y_scores = load_jsonl_predictions(args.output)
    else:
        samples = load_samples(args.annotations, args.asr, args.comments, args.spam)
        if not samples:
            raise RuntimeError("No samples loaded. Please check paths and split files.")

        # For release simplicity, retrieval is optional. If enabled, build the index from the same
        # provided samples unless you pass a training-only split in a separate wrapper script.
        retriever = build_retriever(args, train_samples=samples) if args.use_rag else None
        y_true, y_scores = run_local_model(args, samples, retriever=retriever)

    metrics = compute_metrics(y_true, y_scores)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
