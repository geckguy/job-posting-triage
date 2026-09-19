"""Dataset loading and per-question views.

``uv run python -m triage.data --check`` reproduces the split sizes, the
per-question valid counts and the label distributions that the rest of the
project is calibrated against, and exits non-zero on any mismatch.

The code downloads at run time and never writes the posting text into the repo.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

from datasets import load_dataset

from triage.schema import QUESTION_IDS, QUESTIONS, Question, state

DATASET = "james-burton/fake_job_postings2"
SPLITS = ("train", "validation", "test")

#: Facts every downstream number is quoted against. Verified against the mirror
#: on 2026-09-19; ``--check`` fails loudly if the mirror ever shifts under us.
EXPECTED: dict[str, Any] = {
    "split_sizes": {"train": 10_816, "validation": 1_909, "test": 3_182},
    "test_n": {
        "fraudulent": 3_182,
        "required_experience": 1_992,
        "required_education": 1_680,
        "salary_range_stated": 3_182,
    },
    "test_fraudulent_positives": 138,
    "test_salary_range_stated": 570,
}


@lru_cache(maxsize=4)
def load(split: str):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return load_dataset(DATASET, split=split)


def records(split: str) -> list[dict[str, str]]:
    """One entry per posting: ``record_id`` plus the model-visible ``state``."""
    rows = load(split)
    out: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        out.append(
            {"record_id": f"{split}-{index}", "state": state(row)}
        )
    return out


def golds(
    split: str, questions: Iterable[Question] | None = None
) -> dict[str, dict[str, str]]:
    """``{question_id: {record_id: gold}}`` for every non-null gold in ``split``."""
    rows = load(split)
    questions = tuple(questions) if questions is not None else QUESTIONS
    out: dict[str, dict[str, str]] = {q.id: {} for q in questions}
    for index, row in enumerate(rows):
        record_id = f"{split}-{index}"
        for question in questions:
            value = question.gold(row)
            if value is not None:
                out[question.id][record_id] = value
    return out


def gold_frame(
    split: str, questions: Iterable[Question] | None = None
) -> list[dict[str, str]]:
    """One row per (record, question) that has a gold label.

    Each question is therefore scored on its own valid subset: the 1,190 test
    postings with no recorded experience level simply do not appear in the
    ``required_experience`` rows, rather than being scored against a guess.
    """
    frame: list[dict[str, str]] = []
    for question_id, per_record in golds(split, questions).items():
        for record_id, gold in per_record.items():
            frame.append(
                {"record_id": record_id, "question": question_id, "gold": gold}
            )
    return frame


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------


def _distribution(values: Iterable[str]) -> list[tuple[str, int]]:
    return Counter(values).most_common()


def check() -> int:
    failures: list[str] = []

    sizes = {split: len(load(split)) for split in SPLITS}
    print("split sizes")
    for split in SPLITS:
        expected = EXPECTED["split_sizes"][split]
        ok = sizes[split] == expected
        failures += [] if ok else [f"split {split}: {sizes[split]} != {expected}"]
        print(f"  {split:<11} {sizes[split]:>6}  expected {expected:>6}  "
              f"{'ok' if ok else 'MISMATCH'}")

    gold = golds("test")
    print("\ntest split, per question")
    for question in QUESTIONS:
        values = list(gold[question.id].values())
        expected = EXPECTED["test_n"][question.id]
        ok = len(values) == expected
        failures += [] if ok else [
            f"question {question.id}: n={len(values)} != {expected}"
        ]
        top = _distribution(values)
        majority = f"{top[0][0]} {top[0][1] / len(values):.1%}" if values else "n/a"
        print(f"  {question.id:<22} n={len(values):>5}  expected {expected:>5}  "
              f"{'ok' if ok else 'MISMATCH':<8} majority {majority}")
        if question.id in {"fraudulent", "salary_range_stated"}:
            print(f"    labels: {', '.join(f'{k}={v}' for k, v in top)}")
        else:
            print(f"    labels: {', '.join(f'{k}={v}' for k, v in sorted(top))}")

    positives = sum(1 for v in gold["fraudulent"].values() if v == "fraudulent")
    expected_positives = EXPECTED["test_fraudulent_positives"]
    ok = positives == expected_positives
    failures += [] if ok else [
        f"fraudulent positives: {positives} != {expected_positives}"
    ]
    print(
        f"\nfraudulent positives {positives}  expected {expected_positives}  "
        f"{'ok' if ok else 'MISMATCH'}  ({positives / sizes['test']:.2%} of test)"
    )

    stated = sum(1 for v in gold["salary_range_stated"].values() if v == "stated")
    expected_stated = EXPECTED["test_salary_range_stated"]
    ok = stated == expected_stated
    failures += [] if ok else [
        f"salary_range stated: {stated} != {expected_stated}"
    ]
    rate = stated / sizes["test"]
    print(
        f"salary_range stated   {stated}  expected {expected_stated}  "
        f"{'ok' if ok else 'MISMATCH'}  ({rate:.1%} stated, {1 - rate:.1%} not_stated)"
    )

    print()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS all dataset expectations reproduced")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage.data")
    parser.add_argument(
        "--check", action="store_true", help="verify the dataset against EXPECTED"
    )
    args = parser.parse_args(argv)
    if args.check:
        return check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
