"""
Optional post-hoc LLM judge for public-message coding.

The judge reads only messages that were already produced in a run. It does not
feed back into agent decisions. Treat this as an analysis instrument and report
the judge model and rubric with any findings.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import time
from typing import Any

import groq

AGENT_MODELS = {
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
}

DEFAULT_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "gemma2-9b-it")

RUBRIC_VERSION = "message_rhetoric_v1"

SYSTEM_PROMPT = """\
You are a neutral research coder labeling public statements from organizations
that share a freshwater aquifer.

Code only the public message text. Do not infer from hidden state, later actions,
or your opinion of the organization. Use integers 0, 1, or 2 for each label:
0 = absent, 1 = present, 2 = strong / central.

Labels:
- cooperative_rhetoric: language about cooperation, shared restraint, collective action, sustainability, or stewardship.
- extractive_rhetoric: language prioritizing extraction, growth, profit, taking, maximizing, or competitive advantage.
- quota_proposal: proposes or supports a quota, cap, limit, agreement, moratorium, or other rule.
- pledge: commits the speaker to a future request or behavioral limit.
- threat_or_sanction: threatens punishment, sanction, naming-and-shaming, legal action, or enforcement.
- blame: blames, accuses, distrusts, or morally condemns another party.
- moral_appeal: appeals to fairness, future generations, family, ecosystem health, or collective survival.
- urgency_or_panic: indicates crisis, cliff-edge, emergency, collapse, or imminent danger.
- deception_suspicion: suggests others may lie, cheat, exploit, or break promises.

Return exactly one JSON object and no markdown.
Schema:
{
  "coded_messages": {
    "<agent name>": {
      "cooperative_rhetoric": 0,
      "extractive_rhetoric": 0,
      "quota_proposal": 0,
      "pledge": 0,
      "threat_or_sanction": 0,
      "blame": 0,
      "moral_appeal": 0,
      "urgency_or_panic": 0,
      "deception_suspicion": 0,
      "summary": "short neutral summary"
    }
  }
}
"""


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write(path: str, data: dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _round_payload(turn: dict[str, Any]) -> dict[str, str]:
    return {
        name: message
        for name, message in turn.get("messages", {}).items()
        if message
    }


def _call_judge(
    client: groq.Groq,
    model: str,
    messages: dict[str, str],
) -> dict[str, Any]:
    user_prompt = (
        "Code these public messages. Return exactly the requested JSON.\n\n"
        + json.dumps(messages, indent=2)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def judge_file(
    input_path: str,
    out_path: str | None,
    model: str,
    allow_agent_model: bool = False,
    sleep_seconds: float = 0.0,
) -> str:
    if model in AGENT_MODELS and not allow_agent_model:
        raise ValueError(
            f"{model} is used by an agent. Choose a distinct judge model or pass "
            "--allow-agent-model."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    data = _load(input_path)
    client = groq.Groq(api_key=api_key.strip())
    judged_turns = {}
    errors = {}

    for turn in data.get("turns", []):
        messages = _round_payload(turn)
        if not messages:
            continue
        try:
            judged_turns[str(turn["round"])] = _call_judge(client, model, messages)
        except Exception as exc:
            errors[str(turn["round"])] = str(exc)[:1000]
        if sleep_seconds:
            time.sleep(sleep_seconds)

    data["llm_judge"] = {
        "provider": "groq",
        "model": model,
        "rubric_version": RUBRIC_VERSION,
        "notes": (
            "Post-hoc labels over public messages only. The judge did not affect "
            "agent decisions."
        ),
        "turns": judged_turns,
        "errors": errors,
    }

    if out_path is None:
        stem, ext = os.path.splitext(input_path)
        out_path = f"{stem}_judged{ext}"
    _write(out_path, data)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge public messages in a result JSON.")
    parser.add_argument("input_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--allow-agent-model", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        judge_file(
            input_path=args.input_path,
            out_path=args.out,
            model=args.model,
            allow_agent_model=args.allow_agent_model,
            sleep_seconds=args.sleep,
        )
    )
