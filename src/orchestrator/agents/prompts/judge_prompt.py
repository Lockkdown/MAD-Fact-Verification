from src.utils.constants import LABEL_NAMES

JUDGE_SYSTEM = f"""LANGUAGE RULE: YOU MUST RESPOND IN ENGLISH ONLY. Do NOT use Vietnamese or any other language regardless of the language of the claim or evidence.

You are a neutral fact-checking judge. Debaters have argued over a claim.
Your role is to evaluate argument QUALITY and synthesize the final verdict.

LABEL DEFINITIONS:
- Support: the provided evidence confirms the claim
- Refute: the provided evidence contradicts the claim
- NEI: the provided evidence is genuinely insufficient to confirm or refute the claim

You MUST output EXACTLY this format:
VERDICT: <{LABEL_NAMES[0]}|{LABEL_NAMES[1]}|{LABEL_NAMES[2]}>
REASONING: <cite which argument convinced you and why, in 2-4 sentences>

STRICT RULES:
- VERDICT must be exactly one of: {", ".join(LABEL_NAMES)}
- Base your verdict SOLELY on the provided evidence and debaters' arguments — do NOT introduce any external knowledge
- You MUST cite a specific argument from a debater
- When all debaters unanimously agree on a verdict, you MUST adopt that consensus verdict — do NOT override unanimous agreement"""


def build_judge_prompt(
    statement: str,
    evidence: str,
    all_rounds: list[dict],
    is_unanimous: bool = False,
    consensus_verdict: str | None = None,
) -> list[dict]:
    """Build messages list for the judge. Always called after all debate rounds."""
    debate_summary = _format_full_debate(all_rounds)
    unanimous_note = (
        f"NOTE: All debaters reached UNANIMOUS consensus on [{consensus_verdict}]. "
        f"You MUST return [{consensus_verdict}] as your verdict.\n\n"
        if is_unanimous and consensus_verdict is not None
        else ""
    )
    user_content = (
        f"CLAIM: {statement}\n\n"
        f"EVIDENCE: {evidence}\n\n"
        f"--- DEBATE TRANSCRIPT ---\n{debate_summary}\n\n"
        f"{unanimous_note}"
        f"Review the full debate transcript above. "
        f"Evaluate argument quality and deliver the final verdict."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def _format_full_debate(all_rounds: list[dict]) -> str:
    """Format the full debate transcript for the judge."""
    lines = []
    for r in all_rounds:
        lines.append(f"[Round {r['round']}]")
        for agent in r["verdicts"]:
            lines.append(f"  Agent {agent['agent_id']}: {agent['verdict']} — {agent['reasoning']}")
    return "\n".join(lines)
