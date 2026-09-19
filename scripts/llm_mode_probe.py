"""Three postings, three output constraints, one local model.

    uv run python scripts/llm_mode_probe.py [--base-url URL] [--model NAME]

The benchmark arm constrains the reply with ``response_format.json_schema`` and
leaves the server to guarantee the shape. This probe shows what that guarantee
costs on a small model: the same weights, the same rubric and the same postings
are sent three ways.

* ``json_schema``: the constraint the benchmark uses.
* ``json_object``: JSON on, shape only described in the prompt.
* ``prompt``: no constraint at all, the shape stated in words.

Two of the three postings are not scams, and one of them is not a job posting at
all. An engine that answers the same label for all three is not reading the input,
whatever its accuracy happens to be on the real split.
"""

from __future__ import annotations

import argparse
import sys

import httpx

from triage.arms.llm import (
    CONFIDENCE_KEY,
    RESPONSE_FORMATS,
    response_schema,
    shape_instruction,
    system_prompt,
)
from triage.schema import by_id

POSTINGS: tuple[tuple[str, str], ...] = (
    (
        "obvious scam",
        "Data Entry Clerk - Work From Home\n\nEarn $500 per day. No experience "
        "needed. Pay a $50 onboarding fee for your starter kit and receive your "
        "first payment by wire transfer. Contact hr.hiring2024@gmail.com with "
        "your bank details.",
    ),
    (
        "ordinary engineering role",
        "Senior Backend Engineer, Payments\n\nStripe is hiring a Senior Backend "
        "Engineer to design and operate distributed payment systems in Go and "
        "Rust. 5+ years of experience building high-throughput services required. "
        "Bachelor's degree in Computer Science. Full-time, San Francisco.",
    ),
    (
        "not a job posting",
        "Grandma's Apple Pie\n\nMix 6 sliced apples with sugar, cinnamon and lemon "
        "juice. Fill the crust, dot with butter, cover with a lattice top and bake "
        "at 190C for 45 minutes until golden.",
    ),
)


def ask(
    client: httpx.Client,
    base_url: str,
    model: str,
    question,
    state: str,
    mode: str,
) -> str:
    questions = (question,)
    messages = [
        {"role": "system", "content": system_prompt(questions)},
        {"role": "user", "content": state},
    ]
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
    }
    if mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "triage",
                "strict": True,
                "schema": response_schema(questions),
            },
        }
    elif mode == "json_object":
        messages[0]["content"] += "\n" + shape_instruction(questions)
        payload["response_format"] = {"type": "json_object"}
    else:
        messages[0]["content"] += "\n" + shape_instruction(questions)

    response = client.post(f"{base_url}/chat/completions", json=payload)
    response.raise_for_status()
    body = response.json()
    return (body["choices"][0]["message"].get("content") or "").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm_mode_probe")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--question", default="fraudulent")
    args = parser.parse_args(argv)

    question = by_id(args.question)
    base_url = args.base_url.rstrip("/")
    with httpx.Client(timeout=300.0) as client:
        for mode in RESPONSE_FORMATS:
            print(f"\n{CONFIDENCE_KEY} mode: {mode}")
            for name, state in POSTINGS:
                try:
                    reply = ask(client, base_url, args.model, question, state, mode)
                except httpx.HTTPError as exc:
                    print(f"  {name:<26} request failed: {type(exc).__name__}: {exc}")
                    return 1
                print(f"  {name:<26} {reply[:160]}")
    print(
        "\nRead the column down: identical answers for all three rows mean the "
        "engine ignored the posting."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
