"""CLI runner.

    uv run python -m triage.run --arms all --split test
    uv run python -m triage.run --arms classifier_dev --questions fraudulent --limit 5

One JSONL per arm under ``results/``, a line per (record, question), plus
``results/_unavailable.json`` recording every arm that could not run and why.
Arms are loaded and run one at a time, so peak memory is one model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

from triage import data
from triage.arms.base import Arm, ArmPartial, ArmUnavailable
from triage.arms.classifier_dev import ClassifierDevArm, fraud_subsample
from triage.arms.llm import MAX_TOKENS, RESPONSE_FORMATS, LlmArm
from triage.arms.tfidf import TfidfArm
from triage.schema import Question, select

ALL_ARMS = ("classifier_dev", "tfidf", "gliner", "llm_local")
KNOWN_ARMS = ALL_ARMS + ("llm_hosted",)

#: Where to look for an API key when ``--api-key-env`` names one that is not in
#: the environment. Values are read, never printed.
ENV_FILES = (
    Path.home() / "Code" / "AutoApply" / "backend" / ".env",
    Path(".env"),
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="triage.run")
    parser.add_argument("--arms", default="all",
                        help=f"comma-separated, or 'all' = {','.join(ALL_ARMS)}")
    parser.add_argument("--split", default="test", choices=list(data.SPLITS))
    parser.add_argument("--questions", default=None,
                        help="comma-separated subset; default all four")
    parser.add_argument("--limit", type=int, default=None,
                        help="score a uniformly random sample of N records drawn "
                             "from the whole split with --seed (not a prefix: a "
                             "mirror's row order can be a batch order)")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tier", default="fast", choices=["fast", "smart"],
                        help="classifier.dev tier")
    parser.add_argument("--no-instructions", dest="instructions",
                        action="store_false", default=True,
                        help="drop the rubric field from classifier.dev requests")
    parser.add_argument("--fraud-subsample", type=int, default=None,
                        help="keep all positives plus random negatives, up to N records")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the --limit sample and the fraud subsample")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint (default http://127.0.0.1:8080)")
    parser.add_argument("--model", default="local")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None,
                        help="name of the env var holding the key (never logged)")
    parser.add_argument("--header", action="append", default=None,
                        metavar="NAME=VALUE",
                        help="extra request header for the LLM arm (e.g. x-opencode-session)")
    parser.add_argument("--response-format", default="json_schema",
                        choices=list(RESPONSE_FORMATS),
                        help="LLM arm output constraint; 'prompt' sends none")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="LLM arm output budget per completion; must cover a "
                             "reasoning model's reasoning plus the answer")
    return parser.parse_args(argv)


def _api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key
    if not args.api_key_env:
        return None
    value = os.environ.get(args.api_key_env)
    if value:
        return value
    from dotenv import dotenv_values

    for path in ENV_FILES:
        if not path.exists():
            continue
        found = dotenv_values(path).get(args.api_key_env)
        if found:
            return found
    raise SystemExit(
        f"{args.api_key_env} is not set in the environment and was not found in "
        + ", ".join(str(p) for p in ENV_FILES)
    )


def build_arm(name: str, args: argparse.Namespace) -> tuple[str, Arm]:
    """``(result label, arm)``. The label is what the report calls this run."""
    if name == "classifier_dev":
        label = "classifier_dev" if args.instructions else "classifier_dev_no_rubric"
        return label, ClassifierDevArm(
            tier=args.tier,
            use_instructions=args.instructions,
            cache_path=Path(args.out) / "cache.jsonl",
        )
    if name == "tfidf":
        return name, TfidfArm(split=args.split)
    if name == "gliner":
        from triage.arms.gliner import GlinerArm  # torch is only imported here

        return name, GlinerArm()
    if name in ("llm_local", "llm_hosted"):
        # A non-default output constraint is a different arm, not a silent change
        # to the same one, so it gets its own row and its own file.
        label = name if args.response_format == "json_schema" else f"{name}_{args.response_format}"
        base_url = args.base_url or "http://127.0.0.1:8080"
        return label, LlmArm(
            name=label,
            base_url=base_url,
            model=args.model,
            api_key=_api_key(args),
            concurrency=args.concurrency,
            response_format=args.response_format,
            max_tokens=args.max_tokens,
            extra_headers=_extra_headers(args.header),
        )
    raise SystemExit(f"unknown arm {name!r}; expected one of {', '.join(KNOWN_ARMS)}")


def _extra_headers(raw: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise SystemExit(f"--header expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        headers[name.strip()] = value.strip()
    return headers


def _arm_names(raw: str) -> list[str]:
    names = list(ALL_ARMS) if raw.strip() == "all" else [n.strip() for n in raw.split(",")]
    unknown = [n for n in names if n not in KNOWN_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; expected {', '.join(KNOWN_ARMS)}")
    return names


def _sample_window(
    records: list[dict[str, str]], limit: int | None, seed: int
) -> set[str] | None:
    """A seeded uniform sample of ``limit`` records, or every record.

    Deliberately a sample and not a prefix. The mirrors of this dataset can sit in
    the order they were uploaded to the source site, so the first N rows are a
    batch, not a random draw; the postings in this corpus also cluster (one
    recruiter's campaign can fill a whole page). The seed makes the draw
    reproducible, and the sampled record ids are exactly the ids present in the
    arm's result rows, so the frame is auditable after the fact.
    """
    if limit is None or limit >= len(records):
        return None
    rng = random.Random(seed)
    ids = [record["record_id"] for record in records]
    return set(rng.sample(ids, limit))


def _select_targets(
    questions: tuple[Question, ...],
    gold: dict[str, dict[str, str]],
    window: set[str] | None,
    fraud_subsample_size: int | None,
    seed: int,
) -> dict[str, set[str]]:
    """Per question, the record ids that need an answer.

    ``window`` restricts every question to the same slice of the split, so an arm
    always sees one comparable set of postings; a question is then scored on the
    postings inside that slice that actually have a gold label.
    """
    targets: dict[str, set[str]] = {}
    for question in questions:
        record_ids = list(gold[question.id])  # already in record order
        if window is not None:
            record_ids = [r for r in record_ids if r in window]
        if fraud_subsample_size and question.id == "fraudulent":
            positives = {r for r, g in gold[question.id].items() if g == "fraudulent"}
            record_ids = fraud_subsample(record_ids, positives, fraud_subsample_size, seed)
        targets[question.id] = set(record_ids)
    return targets


def _rows_for(
    label: str,
    split: str,
    questions: tuple[Question, ...],
    records: list[dict[str, str]],
    gold: dict[str, dict[str, str]],
    targets: dict[str, set[str]],
    answers: dict[str, dict[str, Any]],
    mean_latency: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = record["record_id"]
        for question in questions:
            if record_id not in targets[question.id]:
                continue
            answer = answers.get(question.id, {}).get(record_id)
            if answer is None:
                continue
            expected = gold[question.id][record_id]
            latency = None
            if answer.meta.get("latency_measured", True) is not False:
                latency = answer.meta.get("latency_ms")
                if latency is None:
                    latency = mean_latency.get(question.id)
            rows.append(
                {
                    "record_id": record_id,
                    "arm": label,
                    "question": question.id,
                    "split": split,
                    "value": answer.value,
                    "probabilities": answer.probabilities,
                    "confidence": answer.confidence,
                    "gold": expected,
                    "correct": answer.value == expected,
                    "latency_ms": latency,
                    "tokens_in": answer.tokens_in,
                    "tokens_out": answer.tokens_out,
                    "invalid": answer.invalid,
                    "meta": answer.meta,
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _record_status(path: Path, entries: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_frame(
    out_dir: Path,
    args: argparse.Namespace,
    questions: tuple[Question, ...],
    run_records: list[dict[str, str]],
) -> None:
    """Record the frame the arms were asked about, for the evaluator to report.

    Inferring the frame from record indices cannot tell a random sample from a
    prefix, and it cannot recover the seed, so the runner writes it down instead.
    A second run into the same directory that used the same split, limit and seed
    (a `--questions` top-up, say) merges into the existing file rather than
    replacing it, so the record stays a description of the whole directory. A run
    with a different frame replaces it, because then the older frame no longer
    describes what is in the directory.
    """
    path = out_dir / "_frame.json"
    frame = {
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed if args.limit is not None else None,
        "sampling": (
            "uniform random sample without replacement"
            if args.limit is not None
            else "every record of the split"
        ),
        "questions": [question.id for question in questions],
        "records_sent": len(run_records),
        "record_ids": [record["record_id"] for record in run_records],
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if existing and (
            existing.get("split"),
            existing.get("limit"),
            existing.get("seed"),
        ) == (frame["split"], frame["limit"], frame["seed"]):
            frame["questions"] = sorted(
                set(existing.get("questions", [])) | set(frame["questions"])
            )
            frame["record_ids"] = sorted(
                set(existing.get("record_ids", [])) | set(frame["record_ids"])
            )
            frame["records_sent"] = len(frame["record_ids"])
    path.write_text(json.dumps(frame, indent=2) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = select(args.questions.split(",") if args.questions else None)
    records = data.records(args.split)
    gold = data.golds(args.split, questions)
    window = _sample_window(records, args.limit, args.seed)
    targets = _select_targets(questions, gold, window, args.fraud_subsample, args.seed)

    needed = set().union(*targets.values()) if targets else set()
    run_records = [r for r in records if r["record_id"] in needed]
    _write_frame(out_dir, args, questions, run_records)
    scope = (
        "all records"
        if window is None
        else f"random sample of {len(window)} of {len(records)} records, seed {args.seed}"
    )
    print(
        f"{args.split} ({scope}): {len(run_records)} distinct postings sent, "
        + ", ".join(f"{q.id}={len(targets[q.id])} scored" for q in questions)
    )

    statuses: dict[str, Any] = {}
    for name in _arm_names(args.arms):
        label, arm = build_arm(name, args)
        ok, why = arm.available()
        if not ok:
            print(f"[{label}] unavailable: {why}")
            statuses[label] = {"status": "unavailable", "reason": why}
            continue

        started = time.perf_counter()
        try:
            answers = arm.run(run_records, questions, targets)
            status, reason = "ok", ""
        except ArmPartial as exc:
            answers = exc.answers
            status, reason = "partial", str(exc)
            print(f"[{label}] partial: {reason}")
        except ArmUnavailable as exc:
            print(f"[{label}] unavailable: {exc}")
            statuses[label] = {"status": "unavailable", "reason": str(exc)}
            arm.close()
            continue
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            print(f"[{label}] failed: {type(exc).__name__}: {exc}")
            statuses[label] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            arm.close()
            continue
        elapsed = time.perf_counter() - started

        mean_latency = {}
        for question in questions:
            answered = len(answers.get(question.id, {}))
            if answered:
                mean_latency[question.id] = elapsed * 1000 / answered

        rows = _rows_for(
            label, args.split, questions, records, gold, targets, answers, mean_latency
        )
        _write_jsonl(out_dir / f"{label}.jsonl", rows)

        meta = getattr(arm, "usage", None)
        usage = f" usage={meta}" if meta else ""
        covered = {
            q.id: sum(1 for r in rows if r["question"] == q.id) for q in questions
        }
        print(
            f"[{label}] {status}: {len(rows)} rows in {elapsed:.1f}s "
            f"({elapsed * 1000 / max(len(rows), 1):.0f} ms/row){usage} {covered}"
        )
        entry: dict[str, Any] = {"status": status, "rows": len(rows), "seconds": round(elapsed, 1)}
        if reason:
            entry["reason"] = reason
        statuses[label] = entry
        arm.close()

    _record_status(out_dir / "_unavailable.json", statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
