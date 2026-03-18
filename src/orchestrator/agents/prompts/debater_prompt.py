"""Debater prompt builder. Round 1: no prior context. Round 2+: include all prior round summaries."""

DEBATER_R1_SYSTEM = """You are an EVIDENCE-BASED FACT-CHECKER.
Use ONLY the provided evidence. Do NOT use outside knowledge.

STEP 1 — BREAK CLAIM INTO KEY PARTS (2-4 parts):
For each part: find VERBATIM quote from evidence.
- Found and matches → COVERED
- Evidence explicitly contradicts → CONFLICT
- Not found in evidence → MISSING

STEP 2 — DECIDE:
- ALL parts COVERED → Support
- ANY part CONFLICT → Refute
- ANY part MISSING (no CONFLICT) → NEI

Output JSON only:
{
  "parts": [
    {"part": "...", "status": "COVERED|MISSING|CONFLICT", "quote": "verbatim OR null"}
  ],
  "verdict": "Support|Refute|NEI",
  "reasoning": "1-2 sentences"
}"""

DEBATER_R2_SYSTEM = """You are an EVIDENCE-BASED FACT-CHECKER in a multi-agent debate.
Review the arguments from other agents and provide your updated verdict.

You MUST output EXACTLY this format — no other text:
VERDICT: <Support|Refute|NEI>
REASONING: <2-4 sentences citing specific arguments you agree or disagree with>

Rules:
- Base your verdict ONLY on the provided evidence
- Cite specific arguments from other agents when updating your position
- Do NOT introduce external knowledge
- NEI means the evidence is insufficient to confirm or refute"""


def build_debater_prompt(
    statement: str,
    evidence: str,
    round_num: int,
    prior_rounds: list[dict],
    debater_id: str,
) -> list[dict]:
    """Build messages list for a debater. R1 uses JSON parts analysis; R2+ uses rebuttal format."""
    # Round 1: independent analysis — no prior context, structured JSON output
    # Round 2+: debate/rebuttal — includes prior rounds, plain VERDICT/REASONING output
    system = DEBATER_R1_SYSTEM if round_num == 1 else DEBATER_R2_SYSTEM
    user_content = f"CLAIM: {statement}\n\nEVIDENCE: {evidence}"

    if round_num > 1 and prior_rounds:
        prior_text = _format_prior_rounds(prior_rounds, debater_id)
        user_content += f"\n\n--- PRIOR DEBATE ROUNDS ---\n{prior_text}"
        user_content += "\n\nReview the above arguments and provide your updated verdict."

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


def _format_prior_rounds(prior_rounds: list[dict], current_debater_id: str) -> str:
    """Format prior round verdicts with anonymized agent IDs."""
    lines = []
    for r in prior_rounds:
        lines.append(f"[Round {r['round']}]")
        for agent in r["verdicts"]:
            tag = "You" if agent["agent_id"] == current_debater_id else f"Agent {agent['agent_id']}"
            lines.append(f"  {tag}: {agent['verdict']} — {agent['reasoning']}")
    return "\n".join(lines)
