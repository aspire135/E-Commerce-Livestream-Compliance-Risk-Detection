"""
CMR CPO-ordered reasoning prompt template. Section 5.3.

The prompt organizes evidence in Cause → Propagation → Outcome order.
A fixed scoring rubric maps evidence patterns to risk scores 0-100.
"""

CPO_PROMPT_TEMPLATE = """You are an e-commerce livestream compliance review expert. Analyze the following livestream sample using the Cause → Propagation → Outcome (CPO) framework.

## Scoring Protocol

Analyze the evidence in strict CPO order. Do not skip stages.

### Step 1: Cause Layer (highest priority)
Examine host speech and visual context for policy-sensitive behavior:
- Exaggerated or unverifiable product claims
- Unauthorized health/medical/efficacy claims
- Misleading prices, discounts, refunds, or purchase conditions
- Misleading scarcity, urgency, inventory, or guaranteed-benefit claims
- Coordinated interaction manipulation by the host

A finding of HIGH risk at this layer alone can place the score above 50,
even without downstream propagation or outcome signals.

### Step 2: Propagation Layer
Examine audience comments and interaction patterns:
- Repetitive, synchronized, or template-like comments
- Abnormal comment bursts relative to the sample-internal baseline
- Coordinated promotional interaction indicator (I_P)

Strong Propagation confirms and amplifies Cause risk.
Weak Propagation does NOT invalidate Cause risk.

### Step 3: Outcome Layer
Examine cumulative sales records and transaction dynamics:
- Sales surges relative to sample-internal baseline
- Amplification ratio of current window sales

Outcome evidence calibrates the assessment:
- Strong Cause + strong Outcome → score 86-100 (confirmed anomaly)
- Strong Cause + weak Outcome → score 51-70 (high-risk cause, not yet converted)
- Weak Cause → score 0-30 (normal selling)

## Scoring Rubric
| Score Range | Label | Condition |
|-------------|-------|-----------|
| 0-30 | Normal | Cause Layer = low risk |
| 31-50 | Borderline | Cause Layer = low risk, marginal promotional pressure |
| 51-70 | High-Risk Cause | Cause Layer = high risk, Propagation = weak |
| 71-85 | Risky Propagation | Cause Layer = high risk, Propagation = strong |
| 86-100 | Anomalous Conversion | Cause = high, Propagation = strong, Outcome = confirmed |

## Output Format

Output ONLY a JSON object (no markdown, no extra text):

{
  "cause_score": 0.0-1.0,
  "propagation_score": 0.0-1.0,
  "outcome_score": 0.0-1.0,
  "final_score": 0.0-1.0,
  "score": 0-100,
  "reason": "Brief rationale in 80 characters or fewer."
}

"""
