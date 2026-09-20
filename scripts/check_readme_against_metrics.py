"""Check the README's results table against the metrics the report was built from.

    uv run python scripts/check_readme_against_metrics.py [--readme README.md]
                                                          [--metrics results/metrics.json]

The README quotes numbers from `results/report.md` by hand, and
`classifier.dev` is a shared service whose answers move by a record between
fetches, so a regenerated report can quietly disagree with the table. This exits
non-zero on a mismatch and names the cell, so the drift is found by a command
rather than by a reader.

Coverage is asserted rather than assumed, because a check that can pass vacuously
is worse than none:

* every arm that appears in the metrics file must appear in the README table, and
  a missing one is a failure naming it;
* every arm is compared on all seven columns, so the reported count is
  ``arms x 7`` and a skipped row cannot hide in a smaller number;
* a cell whose metric is null must read ``n/a`` in the README rather than being
  left to look checked.

The ``predicts fraud`` column is checked as a count derived from the confusion counts
that the per-class metrics imply (TP from support and recall, then FP from
precision), which is why a floor row with no precision must show zero.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

#: README cell index -> (value key, interval key). Precision and recall are read
#: from the per-class block for the positive label.
COLUMNS: dict[int, tuple[str, str | None, bool]] = {
    2: ("accuracy", None, False),
    3: ("f1_positive", "f1_ci95", False),
    4: ("precision", "precision_ci95", True),
    5: ("recall", "recall_ci95", True),
    6: ("pr_auc", None, False),
    7: ("ece", None, False),
}
PREDICTS_FRAUD_INDEX = 8
POSITIVE = "fraudulent"


def _arm_name(cell: str) -> str:
    """``"`majority_class` (floor)"`` -> ``"majority_class"``."""
    return cell.replace("`", "").replace("(floor)", "").strip()


def _parse(cell: str) -> tuple[float | None, list[float]]:
    """``"0.700 [0.59, 0.79]"`` -> ``(0.700, [0.59, 0.79])``; ``n/a`` -> ``(None, [])``."""
    numbers = re.findall(r"-?\d+\.\d+", cell)
    if not numbers:
        return None, []
    return float(numbers[0]), [float(x) for x in numbers[1:]]


def _parse_counts(cell: str) -> tuple[int, int] | None:
    """``"24/1000"`` -> ``(24, 1000)``; anything else -> ``None``."""
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", cell)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _metric(arm_metric: dict, key: str) -> float | None:
    """The value the report prints for a column.

    Precision is ``n/a`` exactly when the arm predicted no positive at all (the
    report's rule: the per-class precision exists but has no denominator), recall
    always has the positive support behind it.
    """
    per_class = (arm_metric.get("per_class") or {}).get(POSITIVE) or {}
    if key == "precision":
        counts = _flag_counts(arm_metric)
        if counts is None or counts[0] + counts[1] == 0:
            return None
        return per_class.get("precision")
    if key == "recall":
        return per_class.get("recall")
    return arm_metric.get(key)


def _flag_counts(arm_metric: dict) -> tuple[int, int] | None:
    """``(tp, fp)`` implied by support, recall and precision, or ``None``."""
    per_class = (arm_metric.get("per_class") or {}).get(POSITIVE) or {}
    support, recall = per_class.get("support"), per_class.get("recall")
    if support is None or recall is None:
        return None
    tp = round(support * recall)
    if tp == 0:
        return 0, 0
    precision = per_class.get("precision")
    if not precision:
        return None
    return tp, max(round(tp / precision) - tp, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_readme_against_metrics")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--question", default="fraudulent")
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args(argv)

    metrics = json.loads(pathlib.Path(args.metrics).read_text(encoding="utf-8"))
    block = metrics["per_question"][args.question]
    available = {**block["rows"], **block["baselines"]}

    table: dict[str, list[str]] = {}
    for line in pathlib.Path(args.readme).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        # The README has more than one table whose first cell is an arm name; only
        # the results table carries a cell per checked column.
        if len(cells) <= PREDICTS_FRAUD_INDEX:
            continue
        table[_arm_name(cells[0])] = cells

    failures: list[str] = []
    missing = sorted(set(available) - set(table))
    if missing:
        failures.append(
            f"these arms are in {args.metrics} but not in the README table: {missing}"
        )

    expected_n = len(available) * (len(COLUMNS) + 1)
    compared = 0
    for arm, arm_metric in available.items():
        cells = table.get(arm)
        if cells is None:
            continue
        for index, (key, interval_key, per_class) in COLUMNS.items():
            compared += 1
            expected = _metric(arm_metric, key)
            got, got_interval = _parse(cells[index])
            if expected is None:
                if cells[index].strip() != "n/a":
                    failures.append(
                        f"{arm}.{key}: metrics has no value, README says {cells[index]!r}"
                    )
                continue
            if got is None:
                failures.append(f"{arm}.{key}: README {cells[index]!r} vs metrics {expected}")
                continue
            if round(expected, args.digits) != got:
                failures.append(
                    f"{arm}.{key}: README {got} vs metrics {round(expected, args.digits)}"
                )
            interval = arm_metric.get(interval_key) if interval_key else None
            if interval:
                wanted = [round(x, 2) for x in interval]
                if [round(x, 2) for x in got_interval] != wanted:
                    failures.append(
                        f"{arm}.{interval_key}: README {got_interval} vs metrics {wanted}"
                    )
        compared += 1
        counts = _flag_counts(arm_metric)
        pair = _parse_counts(cells[PREDICTS_FRAUD_INDEX])
        if counts is None or pair is None:
            failures.append(
                f"{arm}.predicts_fraud: README {cells[PREDICTS_FRAUD_INDEX]!r} against derivable counts "
                f"{counts}"
            )
            continue
        tp, fp = counts
        # The README's cell is predicted positives over postings scored, so it is
        # the confusion total rather than either half of it.
        wanted_flags, wanted_n = tp + fp, arm_metric.get("n")
        if pair != (wanted_flags, wanted_n):
            failures.append(
                f"{arm}.predicts_fraud: README {cells[PREDICTS_FRAUD_INDEX]!r} vs derived "
                f"{wanted_flags}/{wanted_n}"
            )

    if compared != expected_n:
        failures.append(
            f"compared {compared} cells, expected {expected_n} "
            f"({len(available)} arms x {len(COLUMNS) + 1} columns)"
        )

    print(
        f"compared {compared} of {expected_n} expected cells across {len(available)} arms "
        f"in the README table against {args.metrics}"
    )
    if failures:
        for failure in failures:
            print(f"  MISMATCH {failure}")
        print(f"FAIL {len(failures)} problem(s)")
        return 1
    print("PASS every arm and every checked column matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
