"""Judge prompt — Constrained Synthesizer. Only called when debaters are NOT unanimous."""

from src.utils.constants import LABEL_NAMES

JUDGE_SYSTEM = f"""You are a neutral fact-checking judge. Debaters have argued over a claim.
Your role is to evaluate argument QUALITY and resolve disagreements.

You MUST output EXACTLY this format:
VERDICT: <{LABEL_NAMES[0]}|{LABEL_NAMES[1]}|{LABEL_NAMES[2]}>
REASONING: <cite which argument convinced you and why, in 2-4 sentences>

STRICT RULES:
- VERDICT must be exactly one of: {", ".join(LABEL_NAMES)}
- Base your verdict ONLY on the debaters' arguments — no external knowledge
- You MUST cite a specific argument from a debater
- You CANNOT introduce new evidence or claims not mentioned by any debater"""


def build_judge_prompt(
    statement: str,
    all_rounds: list[dict],
) -> list[dict]:
    """Build messages list for the judge. Called only on split debater verdicts."""
    debate_summary = _format_full_debate(all_rounds)
    user_content = (
        f"CLAIM: {statement}\n\n"
        f"--- DEBATE TRANSCRIPT ---\n{debate_summary}\n\n"
        f"The debaters have NOT reached consensus. "
        f"Evaluate argument quality and give your verdict."
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
