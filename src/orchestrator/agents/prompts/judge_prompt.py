"""Judge prompt builder — constructs system and user messages for the JudgeAgent."""

import random

from src.utils.constants import LABEL_NAMES

_ARGUMENT_LABELS = ["A", "B", "C", "D", "E", "F"]

JUDGE_SYSTEM = f"""LANGUAGE RULE: YOU MUST RESPOND IN ENGLISH ONLY. Do NOT use Vietnamese or any other language regardless of the language of the claim or evidence.

You are a neutral fact-checking judge. Debaters have argued over a claim.
Your role is to evaluate argument QUALITY and evidence alignment to synthesize the final verdict.

LABEL DEFINITIONS:
- Support: the provided evidence confirms the claim
- Refute: the provided evidence contradicts the claim
- NEI: the provided evidence is genuinely insufficient to confirm or refute the claim

You MUST output EXACTLY this format:
VERDICT: <{LABEL_NAMES[0]}|{LABEL_NAMES[1]}|{LABEL_NAMES[2]}>
REASONING: <your evaluation in 2-4 sentences>

STRICT RULES:
- VERDICT must be exactly one of: {", ".join(LABEL_NAMES)}
- Base your verdict SOLELY on the provided evidence and debaters' arguments — do NOT introduce any external knowledge
- Do NOT count how many debaters support each position — evaluate reasoning quality only
- A single well-reasoned argument that correctly interprets the evidence outweighs multiple poorly-reasoned arguments
- NEI is valid ONLY when the evidence genuinely lacks information — not as a safe middle ground"""


def build_judge_prompt(
    statement: str,
    evidence: str,
    all_rounds: list[dict],
    is_unanimous: bool = False,
    consensus_verdict: str | None = None,
) -> list[dict]:
    """Build messages list for the judge. Always called after all debate rounds."""
    debate_summary = _format_debate_transcript(all_rounds, is_unanimous)

    if is_unanimous and consensus_verdict is not None:
        context_note = f"NOTE: All debaters reached unanimous consensus on [{consensus_verdict}].\n\n"
        task_instruction = (
            "Review the debate transcript above. "
            "Validate the consensus and explain your reasoning briefly."
        )
    else:
        context_note = ""
        task_instruction = (
            "Review the debate transcript above. Follow these steps:\n"
            "1. First, independently analyze what the EVIDENCE says about the CLAIM (2-3 sentences).\n"
            "2. Then, evaluate which side's arguments correctly align with the evidence.\n"
            "3. Deliver your verdict based on evidence-argument alignment — "
            "NOT on how many debaters hold each position."
        )

    user_content = (
        f"CLAIM: {statement}\n\n"
        f"EVIDENCE: {evidence}\n\n"
        f"--- DEBATE TRANSCRIPT ---\n{debate_summary}\n\n"
        f"{context_note}"
        f"{task_instruction}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]


def _format_debate_transcript(all_rounds: list[dict], is_unanimous: bool) -> str:
    """Format the debate transcript. Split cases use shuffled, anonymized argument labels."""
    rng = random.Random(42)
    lines = []
    for r in all_rounds:
        lines.append(f"[Round {r['round']}]")
        verdicts = list(r["verdicts"])
        if not is_unanimous:
            rng.shuffle(verdicts)
        for i, agent in enumerate(verdicts):
            label = f"Agent {agent['agent_id']}" if is_unanimous else f"Argument {_ARGUMENT_LABELS[i]}"
            lines.append(f"  {label}: {agent['verdict']} — {agent['reasoning']}")
    return "\n".join(lines)
