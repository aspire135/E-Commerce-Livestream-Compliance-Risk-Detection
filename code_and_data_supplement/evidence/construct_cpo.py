"""
CPO evidence serialization. Section 5.2.
Serializes heterogeneous multimodal observations into structured textual
representation for CMR input.

Input streams: ASR transcript T_A, visual caption T_V, comments C, sales S.
Output: structured CPO text with Cause → Propagation → Outcome ordering.
"""


def serialize_cpo_evidence(
    asr_text: str,
    visual_caption: str,
    comment_text: str,
    propagation_seq: list,    # [[n,d,q,δ_P,ρ_P,I_P], ...]
    outcome_seq: list,        # [[s,Δs,δ_O,ρ_O], ...]
    I_P: int = 0,
) -> str:
    """
    Serialize CPO evidence into structured text for CMR prompt.

    Cause: host speech + visual context
    Propagation: audience comments + interaction indicators
    Outcome: sales dynamics

    Returns a formatted string matching the CMR prompt template.
    """
    lines = []

    # ── Cause Evidence ──
    lines.append("## Cause Evidence")
    lines.append("")
    lines.append("### Host Speech (ASR Transcript)")
    lines.append(asr_text[:800] if asr_text else "(no speech data)")
    lines.append("")
    lines.append("### Visual Context")
    lines.append(visual_caption[:400] if visual_caption else "(no visual data)")
    lines.append("")

    # ── Propagation Evidence ──
    lines.append("## Propagation Evidence")
    lines.append("")
    lines.append("### Audience Comments (temporal order)")
    lines.append(comment_text[:500] if comment_text else "(no comment data)")
    lines.append("")
    lines.append(f"### Coordinated Interaction Indicator: I_P = {I_P}")
    lines.append("")
    lines.append("### Propagation Windows (5s each)")
    for w, feat in enumerate(propagation_seq):
        n, d, q, delta, rho, ip = feat
        surge = "SURGE" if delta == 1 else "normal"
        lines.append(
            f"  Window {w:2d} [{w*5:2d}-{(w+1)*5:2d}s]: "
            f"comments={n:.0f} dup_ratio={d:.3f} "
            f"amp_ratio={rho:.1f} {surge}"
        )
    lines.append("")

    # ── Outcome Evidence ──
    lines.append("## Outcome Evidence")
    lines.append("")
    lines.append("### Sales Windows (5s each)")
    for w, feat in enumerate(outcome_seq):
        s, ds, delta, rho = feat
        surge = "SURGE" if delta == 1 else "normal"
        lines.append(
            f"  Window {w:2d} [{w*5:2d}-{(w+1)*5:2d}s]: "
            f"cumulative={s:.1f} increment={ds:.1f} "
            f"amp_ratio={rho:.1f} {surge}"
        )

    return "\n".join(lines)
