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
- Do NOT count how many debaters support each position — evaluate how well each argument is grounded in the provided evidence
- An argument that stays within what the evidence explicitly says deserves more weight than one that reads into or extrapolates from it
- NEI reflects genuine evidentiary insufficiency — assign it when warranted by the evidence, neither avoiding it nor defaulting to it. If the evidence addresses the core claim, minor unverified details alone do not warrant NEI"""


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
            "The debaters reached unanimous consensus. "
            "Verify this conclusion against the EVIDENCE — confirm it if the evidence supports it, "
            "or correct it if the evidence clearly points elsewhere. Explain briefly."
        )
    else:
        context_note = ""
        task_instruction = (
            "Review the debate transcript above. Follow these steps:\n"
            "1. Before engaging with the debate, determine what verdict the evidence alone supports — this is your working verdict.\n"
            "2. Check whether any debater has identified something in the evidence that you may have missed.\n"
            "3. Maintain your working verdict unless a debater's argument reveals something in the evidence that concretely changes the analysis."
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
