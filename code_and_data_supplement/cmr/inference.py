"""
CMR inference with retrieval-augmented CPO-ordered reasoning. Section 5.3.
Supports Qwen3.5-9B (SGLang API) and Qwen2.5-14B (local HF).

Input: serialized CPO evidence + retrieved cases + fixed prompt.
Output: {cause_score, propagation_score, outcome_score, final_score, score, reason}
"""

import json, re, time, argparse
import numpy as np
from cmr.parse_output import parse_cmr_response
from cmr.prompts.cpo_prompt import CPO_PROMPT_TEMPLATE
from retrieval.retrieve_cases import CaseRetriever


def build_cmr_prompt(evidence_text: str, retrieved_cases: list) -> str:
    """Build the full CMR prompt with evidence and retrieved cases."""
    prompt = CPO_PROMPT_TEMPLATE

    prompt += "\n## Current Sample Evidence\n\n"
    prompt += evidence_text
    prompt += "\n\n"

    if retrieved_cases:
        prompt += "## Retrieved Similar Cases\n\n"
        for i, case in enumerate(retrieved_cases, 1):
            label_str = "RISKY" if case["label"] == 1 else "NORMAL"
            prompt += (
                f"### Case {i} (similarity={case['similarity']:.3f}, label={label_str})\n"
                f"{case.get('rationale', '(no rationale)')[:300]}\n\n"
            )

    prompt += "\nPlease analyze the CPO evidence and output a structured JSON response."
    return prompt


def run_cmr_inference_qwen_api(
    evidence_text: str,
    retrieved_cases: list,
    api_base: str,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> dict:
    """
    Run CMR via SGLang/OpenAI-compatible API (Qwen3.5-9B).
    """
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=api_base)

    prompt = build_cmr_prompt(evidence_text, retrieved_cases)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an e-commerce livestream compliance review expert."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return parse_cmr_response(response.choices[0].message.content or "")


def run_cmr_inference_hf(
    evidence_text: str,
    retrieved_cases: list,
    model,
    tokenizer,
) -> dict:
    """
    Run CMR via local HuggingFace model (Qwen2.5-14B, FP16).
    """
    prompt = build_cmr_prompt(evidence_text, retrieved_cases)
    messages = [
        {"role": "system", "content": "You are an e-commerce livestream compliance review expert."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_cmr_response(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file for CMR")
    parser.add_argument("--input", type=str, required=True,
                        help="JSONL file of test samples with CPO evidence")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for CMR results")
    parser.add_argument("--case_base", type=str, required=True,
                        help="Training case base JSONL")
    parser.add_argument("--mode", type=str, default="api",
                        choices=["api", "hf"])
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load case base
    with open(args.case_base) as f:
        cases = [json.loads(line) for line in f]

    # Build retriever
    retriever = CaseRetriever(top_k=cfg.get("top_k", 3))
    retriever.build_index(cases)

    # Load test samples
    with open(args.input) as f:
        samples = [json.loads(line) for line in f]

    results = []
    for s in samples:
        t0 = time.perf_counter()

        # Retrieve cases (exclude self and same session)
        retrieved = retriever.retrieve(
            query_evidence_text=s["evidence_text"],
            query_sample_id=s["sample_id"],
            query_session_id=s["session_id"],
        )

        # Run CMR
        if args.mode == "api":
            cmr_out = run_cmr_inference_qwen_api(
                s["evidence_text"], retrieved,
                api_base=cfg["api_base"],
                model_name=cfg["model_name"],
            )
        else:
            cmr_out = run_cmr_inference_hf(
                s["evidence_text"], retrieved,
                model=cfg["model"],  # pre-loaded
                tokenizer=cfg["tokenizer"],
            )

        latency = time.perf_counter() - t0

        results.append({
            "sample_id": s["sample_id"],
            "cmr_score": cmr_out.get("final_score", 0.5),
            "cmr_raw_score": cmr_out.get("raw_score", 50),
            "cause_score": cmr_out.get("cause_score", 0.0),
            "propagation_score": cmr_out.get("propagation_score", 0.0),
            "outcome_score": cmr_out.get("outcome_score", 0.0),
            "reason": cmr_out.get("reason", ""),
            "is_violation": cmr_out.get("is_violation", False),
            "latency_s": latency,
        })

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"CMR inference complete: {len(results)} samples → {args.output}")


if __name__ == "__main__":
    main()
