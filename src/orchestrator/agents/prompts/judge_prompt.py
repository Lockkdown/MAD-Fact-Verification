from src.utils.constants import LABEL_NAMES

JUDGE_SYSTEM = f"""LANGUAGE RULE: YOU MUST RESPOND IN ENGLISH ONLY. Do NOT use Vietnamese or any other language regardless of the language of the claim or evidence.

You are a neutral fact-checking judge. Debaters have argued over a claim.
Your role is to evaluate argument QUALITY and synthesize the final verdict.

You MUST output EXACTLY this format:
VERDICT: <{LABEL_NAMES[0]}|{LABEL_NAMES[1]}|{LABEL_NAMES[2]}>
REASONING: <cite which argument convinced you and why, in 2-4 sentences>

STRICT RULES:
- VERDICT must be exactly one of: {", ".join(LABEL_NAMES)}
- Base your verdict on the debaters' arguments AND the provided evidence — no external knowledge beyond what is provided
- You MUST cite a specific argument from a debater
- You CANNOT introduce external knowledge beyond the provided evidence and debaters' arguments"""


def build_judge_prompt(
    statement: str,
    evidence: str,
    all_rounds: list[dict],
) -> list[dict]:
    """Build messages list for the judge. Always called after all debate rounds."""
    debate_summary = _format_full_debate(all_rounds)
    user_content = (
        f"CLAIM: {statement}\n\n"
        f"EVIDENCE: {evidence}\n\n"
        f"--- DEBATE TRANSCRIPT ---\n{debate_summary}\n\n"
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
