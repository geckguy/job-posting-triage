"""Metrics, calibration, routing and cost for the job-posting-triage benchmark.

    uv run python -m triage.evaluate --results results
    uv run python -m triage.evaluate --results results --report
    uv run python -m triage.evaluate --results results --failures \
        --arm llm_local --question fraudulent

Everything is computed from ``results/*.jsonl`` plus the gold in
``triage/data.py``; nothing in this module is hardcoded and nothing that cannot
be computed is invented: every ``n/a`` cell carries its reason.

The arm identity of a row is its ``arm`` field, never the filename. ``cache.jsonl``
and ``_unavailable.json`` are not result files and are skipped when globbing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from triage import data
from triage.schema import ORDINAL, QUESTION_IDS, Question, by_id, select

#: Canonical arm order for every table; unknown arms are appended sorted.
ARM_ORDER: tuple[str, ...] = (
    "classifier_dev",
    "classifier_dev_no_rubric",
    "tfidf",
    "gliner",
    "llm_local",
    "llm_hosted",
)

FLOOR_MAJORITY = "majority_class"
FLOOR_DUMMY = "dummy_stratified"
FLOORS: tuple[str, ...] = (FLOOR_MAJORITY, FLOOR_DUMMY)

#: Thresholds swept by the routing table for the binary question.
TAUS: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95)

#: Hard leakage guard: an arm's ``salary_range_stated`` accuracy must sit within
#: this many percentage points of the majority rate for that question.
LEAK_TOLERANCE_PP = 3.0

#: Characters of posting text printed by ``--failures`` (stdout only).
FAILURE_STATE_CHARS = 300

#: Default caveat under the cost table. The evaluator cannot derive this from the
#: rows: latency was measured on whichever machine ran the arm, next to whatever
#: else was running there.
DEFAULT_COST_NOTE = (
    "Per-row latency is wall clock on the machine that ran the arm; arms that ran "
    "concurrently include each other's load in these numbers, so the projection is an "
    "upper bound for this machine rather than a benchmark."
)

REQUIRED_ROW_KEYS = ("record_id", "arm", "question", "value", "gold")

#: Metrics that need a distribution; ``None`` means "report n/a with a reason".
MetricValue = float | None


# --------------------------------------------------------------------------
# Row loading
# --------------------------------------------------------------------------


#: Arm-supplied reason strings are normalized before they enter the report: dashes
#: become ASCII, whitespace collapses, and the text is truncated.
INVALID_REASON_CHARS = 160
_DASHES = {"\u2014": " - ", "\u2013": " - ", "\u2212": "-", "\u2015": " - "}


def _clean_reason(text: str) -> str:
    for dash, replacement in _DASHES.items():
        text = text.replace(dash, replacement)
    text = " ".join(str(text).split())
    return text[:INVALID_REASON_CHARS]


#: How many distinct arm-reported reasons the report names before summarising the
#: rest; metrics.json always carries the full breakdown.
INVALID_REASONS_SHOWN = 4


def _reason_summary(reasons: Any) -> str:
    """``"12 x 'why'; 3 x 'other why'; and 5 more reason(s)"`` for the report."""
    ordered = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = "; ".join(f"{count} x '{reason}'" for reason, count in ordered[:INVALID_REASONS_SHOWN])
    rest = len(ordered) - INVALID_REASONS_SHOWN
    return shown + (f"; and {rest} more distinct reason(s)" if rest > 0 else "")


def _invalid_reason(row: dict) -> str:
    """Why an arm marked its own row invalid, from the row's ``meta``.

    A descriptive ``reason`` wins over a raw ``error``; ``finish_reason`` and
    ``max_tokens`` are appended when the arm records them, so a truncated or
    budget-exhausted answer is visible in the report rather than hidden behind a
    JSON parse error.
    """
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return "no meta on the row"
    primary = None
    for key in ("reason", "invalid_reason", "error"):
        value = meta.get(key)
        if value:
            primary = _clean_reason(str(value))
            break
    if primary is None:
        if meta.get("finish_reason"):
            primary = f"invalid with finish_reason={meta['finish_reason']}"
        else:
            primary = _clean_reason(
                "meta carries no reason; keys: " + ", ".join(sorted(meta))
            )
    extras = [
        f"{key}={meta[key]}"
        for key in ("finish_reason", "max_tokens")
        if meta.get(key) is not None and str(meta[key]) not in primary
    ]
    combined = primary + (f" [{', '.join(extras)}]" if extras else "")
    return combined[:INVALID_REASON_CHARS]


def _p_float(value: Any) -> float | None:
    """A probability in [0, 1], or ``None`` when the row does not carry one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0.0 or number > 1.0:  # NaN or out of range
        return None
    return number


@dataclass
class Loaded:
    """Everything ``results/`` gave us, and everything it did not."""

    rows: list[dict] = field(default_factory=list)
    arms: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    #: arm -> filenames its rows came from (they need not match the arm name).
    sources: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    files: list[tuple[str, int]] = field(default_factory=list)
    malformed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    status: dict[str, dict] = field(default_factory=dict)
    status_note: str = ""
    splits: Counter = field(default_factory=Counter)
    #: the runner's own record of the frame it used (``results/_frame.json``),
    #: preferred over anything this module can derive from record ids.
    frame_recorded: dict | None = None
    frame_note: str = ""

    @property
    def arm_names(self) -> list[str]:
        found = set(self.arms)
        ordered = [a for a in ARM_ORDER if a in found]
        return ordered + sorted(found - set(ordered))


def load_results(results_dir: Path) -> Loaded:
    loaded = Loaded()
    paths = sorted(p for p in results_dir.glob("*.jsonl") if p.name != "cache.jsonl")
    for path in paths:
        good = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                loaded.malformed[path.name] += 1
                continue
            if not isinstance(row, dict) or any(
                key not in row for key in REQUIRED_ROW_KEYS
            ):
                loaded.malformed[path.name] += 1
                continue
            arm = str(row.get("arm") or "unknown_arm")
            loaded.rows.append(row)
            loaded.arms[arm].append(row)
            if path.name not in loaded.sources[arm]:
                loaded.sources[arm].append(path.name)
            loaded.splits[str(row.get("split") or "?")] += 1
            good += 1
        loaded.files.append((path.name, good))

    frame_path = results_dir / "_frame.json"
    if frame_path.exists():
        try:
            recorded = json.loads(frame_path.read_text(encoding="utf-8"))
            if isinstance(recorded, dict):
                loaded.frame_recorded = recorded
            else:
                loaded.frame_note = "_frame.json is not a JSON object; ignored"
        except json.JSONDecodeError as exc:
            loaded.frame_note = f"_frame.json is unreadable ({exc}); ignored"
    else:
        loaded.frame_note = "no _frame.json in this directory"

    status_path = results_dir / "_unavailable.json"
    if status_path.exists():
        try:
            raw = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                loaded.status = raw
            else:
                loaded.status_note = "not a JSON object; ignored"
        except json.JSONDecodeError as exc:
            loaded.status_note = f"unreadable ({exc}); ignored"
    else:
        loaded.status_note = "no _unavailable.json in this directory"
    return loaded


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------


class GoldBook:
    """Gold labels per split: the dataset when available, the row otherwise.

    ``triage/data.py`` is authoritative, so when a row's ``(split, record_id)``
    exists in the dataset its gold overrides the row's own ``gold`` field; any
    disagreement is counted and reported as an integrity problem. Rows whose
    record is not in the dataset (synthetic fixtures, other splits) fall back to
    the ``gold`` field the row carries.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, dict[str, str]]] = {}
        self.errors: dict[str, str] = {}
        self.from_dataset = 0
        self.from_row = 0
        self.mismatch: Counter = Counter()

    def _split_gold(self, split: str) -> dict[str, dict[str, str]] | None:
        if split in self._cache:
            return self._cache[split]
        if split not in data.SPLITS:
            self.errors.setdefault(
                split, f"{split!r} is not a dataset split ({', '.join(data.SPLITS)})"
            )
            self._cache[split] = {}
            return {}
        try:
            gold = data.golds(split)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            self.errors[split] = f"{type(exc).__name__}: {exc}"
            self._cache[split] = {}
            return {}
        self._cache[split] = gold
        return gold

    def gold(self, row: dict, question: Question) -> tuple[str | None, str]:
        """``(gold, source)`` where source is ``dataset``, ``row`` or ``none``."""
        row_gold = row.get("gold")
        row_gold = None if row_gold is None else str(row_gold)
        split = _split_of(row)
        gold = self._split_gold(split) if split else None
        if gold and question.id in gold:
            found = gold[question.id].get(str(row["record_id"]))
            if found is not None:
                if row_gold is not None and row_gold != found:
                    self.mismatch[(question.id, split)] += 1
                self.from_dataset += 1
                return found, "dataset"
        if row_gold is None:
            return None, "none"
        self.from_row += 1
        return row_gold, "row"

    def train_values(self, question: Question) -> tuple[list[str], str]:
        """Every gold label for ``question`` in train+validation (the floor's fit set)."""
        values: list[str] = []
        for split in ("train", "validation"):
            gold = self._split_gold(split)
            if gold and question.id in gold:
                values.extend(gold[question.id].values())
        if values:
            return values, ""
        why = "; ".join(
            f"{s}: {self.errors.get(s, 'no gold')}" for s in ("train", "validation")
        )
        return [], f"could not load train+validation gold ({why})"


def _split_of(row: dict) -> str:
    split = row.get("split")
    if isinstance(split, str) and split:
        return split
    record_id = str(row.get("record_id") or "")
    prefix = record_id.rsplit("-", 1)[0] if "-" in record_id else ""
    return prefix


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class Scored:
    """One result row, with gold and the two unified probabilities resolved."""

    row: dict
    record_id: str
    split: str
    arm: str
    gold: str
    value: str
    correct: bool
    p_decision: float | None
    p_positive: float | None
    invalid: bool
    gold_source: str
    latency_ms: float | None
    tokens_in: int | None
    tokens_out: int | None


@dataclass
class ArmQuestion:
    """Every scored row of one arm on one question, plus what was dropped."""

    arm: str
    question: Question
    scored: list[Scored] = field(default_factory=list)
    no_gold: int = 0
    unparseable: int = 0
    correct_mismatch: int = 0
    gold_from_row: int = 0

    @property
    def n(self) -> int:
        return len(self.scored)

    @property
    def positives(self) -> int:
        positive = self.question.positive
        if positive is None:
            return 0
        return sum(1 for s in self.scored if s.gold == positive)


def p_decision(row: dict) -> float | None:
    """The arm's stated probability that its chosen answer is right.

    ``confidence if confidence is not None else probabilities[value]``: the
    same rule for every arm, so the calibration table compares like with like.
    """
    confidence = _p_float(row.get("confidence"))
    if confidence is not None:
        return confidence
    probabilities = row.get("probabilities")
    if isinstance(probabilities, dict):
        return _p_float(probabilities.get(str(row.get("value"))))
    return None


def p_positive(row: dict, question: Question) -> float | None:
    """The arm's implied probability of the positive label (binary questions only).

    ``probabilities[positive]`` when a distribution exists, else
    ``confidence if value == positive else 1 - confidence``.
    """
    positive = question.positive
    if positive is None:
        return None
    probabilities = row.get("probabilities")
    if isinstance(probabilities, dict):
        found = _p_float(probabilities.get(positive))
        if found is not None:
            return found
    confidence = _p_float(row.get("confidence"))
    if confidence is None:
        return None
    return confidence if str(row.get("value")) == positive else 1.0 - confidence


def build_arm_question(
    arm: str, question: Question, rows: Iterable[dict], goldbook: GoldBook
) -> ArmQuestion:
    aq = ArmQuestion(arm=arm, question=question)
    for row in rows:
        if str(row.get("question")) != question.id:
            continue
        gold, source = goldbook.gold(row, question)
        if gold is None:
            aq.no_gold += 1
            continue
        value = str(row.get("value") if row.get("value") is not None else "")
        correct = value == gold
        if row.get("correct") is not None and bool(row["correct"]) is not correct:
            aq.correct_mismatch += 1
        if source == "row":
            aq.gold_from_row += 1
        if value not in question.labels:
            aq.unparseable += 1
        aq.scored.append(
            Scored(
                row=row,
                record_id=str(row.get("record_id")),
                split=_split_of(row),
                arm=arm,
                gold=gold,
                value=value,
                correct=correct,
                p_decision=p_decision(row),
                p_positive=p_positive(row, question),
                invalid=bool(row.get("invalid")),
                gold_source=source,
                latency_ms=_p_float(row.get("latency_ms"))
                if row.get("latency_ms") is not None
                else None,
                tokens_in=row.get("tokens_in"),
                tokens_out=row.get("tokens_out"),
            )
        )
    return aq


# --------------------------------------------------------------------------
# Confidence intervals (binary question only)
# --------------------------------------------------------------------------

#: Two-sided 95 percent normal quantile, hard coded so no scipy is needed.
Z95 = 1.959963984540054

#: Bootstrap resamples for the F1 interval; the RNG is seeded so reruns agree.
BOOTSTRAP_RESAMPLES = 2000


def wilson_interval(successes: int, trials: int) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion, two-sided 95 percent.

    ``None`` when there are no trials: a proportion without a denominator is not
    a number, and the caller reports it as n/a with a reason.
    """
    if trials <= 0:
        return None
    n = float(trials)
    p_hat = successes / n
    z2 = Z95 * Z95
    centre = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    half = (
        Z95 * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    ) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Linear interpolation percentile (the numpy/matplotlib convention)."""
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_f1_interval(
    gold_positive: Sequence[bool],
    predicted_positive: Sequence[bool],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Deterministic bootstrap interval for the positive-class F1.

    Rows are resampled with replacement ``resamples`` times using
    ``random.Random(seed)`` and F1 is recomputed each time, so the same rows always
    give the same interval. Percentiles use linear interpolation.
    """
    n = len(gold_positive)
    if n == 0:
        return None
    buckets: list[tuple[bool, bool]] = list(zip(gold_positive, predicted_positive))
    tp_idx = [i for i, (g, p) in enumerate(buckets) if g and p]
    fp_idx = [i for i, (g, p) in enumerate(buckets) if (not g) and p]
    fn_idx = [i for i, (g, p) in enumerate(buckets) if g and (not p)]
    rng = random.Random(seed)
    population = range(n)
    values: list[float] = []
    for _ in range(resamples):
        counts = Counter(rng.choices(population, k=n))
        tp = sum(counts[i] for i in tp_idx)
        fp = sum(counts[i] for i in fp_idx)
        fn = sum(counts[i] for i in fn_idx)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        values.append(
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
    values.sort()
    return (_percentile(values, 0.025), _percentile(values, 0.975))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def calibration(pairs: Sequence[tuple[float, bool]]) -> dict[str, Any]:
    """Brier score, 10-bin equal-width ECE, and the bin table."""
    if not pairs:
        return {"n": 0, "brier": None, "ece": None, "bins": []}
    brier = mean((p - (1.0 if c else 0.0)) ** 2 for p, c in pairs)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    total = len(pairs)
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        members = [
            (p, c)
            for p, c in pairs
            if (low <= p < high) or (index == 9 and p >= high)
        ]
        if not members:
            bins.append({"low": low, "high": high, "n": 0, "conf": None, "acc": None})
            continue
        conf = mean(p for p, _ in members)
        acc = mean(1.0 if c else 0.0 for _, c in members)
        ece += len(members) / total * abs(conf - acc)
        bins.append(
            {"low": low, "high": high, "n": len(members), "conf": conf, "acc": acc}
        )
    return {"n": total, "brier": brier, "ece": ece, "bins": bins}


def _macro_f1(
    question: Question, y_true: Sequence[str], y_pred: Sequence[str]
) -> MetricValue:
    from sklearn.metrics import f1_score

    if not y_true:
        return None
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(question.labels),
            average="macro",
            zero_division=0,
        )
    )


def _positive_f1(tp: int, fp: int, fn: int) -> float:
    """F1 of the positive class from the confusion counts.

    Computed from the counts rather than sklearn so that a prediction outside
    the declared label set (an arm returning junk) cannot make the call raise
    ``ValueError: Target is multiclass``; it is already counted as wrong by
    accuracy. ``zero_division=0`` semantics.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def per_class(
    question: Question, y_true: Sequence[str], y_pred: Sequence[str]
) -> list[dict]:
    """Per-label precision/recall/F1/support over the question's declared labels.

    Precision is ``None`` rather than ``0.0`` when the arm made no prediction of
    that label: sklearn's ``zero_division=0`` would report a number for a
    quantity that has no denominator (there is nothing to divide by), which
    contradicts the ``n/a`` the positive-class column of the metrics table
    prints for the same situation. Recall and F1 keep sklearn's convention.
    """
    from sklearn.metrics import precision_recall_fscore_support

    if not y_true:
        return []
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(question.labels), zero_division=0
    )
    predicted = Counter(y_pred)
    return [
        {
            "label": label,
            "precision": float(p) if predicted.get(label) else None,
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(question.labels, precision, recall, f1, support)
    ]


@dataclass
class Metrics:
    arm: str
    question: Question
    n: int
    accuracy: MetricValue
    macro_f1: MetricValue
    classes: list[dict]
    invalid_rate: MetricValue
    #: ``[{"metric": str, "reason": str}]``: every metric of this arm on this
    #: question that could not be produced, with its reason. Rendered into
    #: report.md section 8 and into metrics.json's ``n_a``.
    na: list[dict] = field(default_factory=list)
    # binary
    pr_auc: MetricValue = None
    #: 95 percent intervals on the positive class, from the same counts
    precision_ci95: tuple[float, float] | None = None
    recall_ci95: tuple[float, float] | None = None
    f1_ci95: tuple[float, float] | None = None
    #: how many positives this frame has, for the interval reader
    ci_note: str | None = None
    f1_positive: MetricValue = None
    precision_positive: MetricValue = None
    recall_positive: MetricValue = None
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    #: rows where the arm produced no label from the declared set (empty, junk, or
    #: flagged invalid). They are neither a false positive nor a true negative:
    #: an empty reply made no call at all, and scoring it as a correct negative
    #: would be exactly the failure this benchmark exists to catch.
    no_call: int = 0
    #: of ``no_call``: rows the arm flagged ``invalid``, and rows whose gold is the
    #: positive label (so they cost recall as well as accuracy).
    no_call_invalid: int = 0
    no_call_positives: int = 0
    positives: int = 0
    # ordinal
    mae: MetricValue = None
    plus_minus_1: MetricValue = None
    ordinal_n: int = 0
    ordinal_excluded: int = 0
    #: reason -> count for rows the arm itself marked ``invalid``.
    invalid_reasons: dict[str, int] = field(default_factory=dict)
    # calibration
    cal_n: int = 0
    cal_excluded: int = 0
    #: of the excluded rows: how many are ``invalid`` and how many are valid
    #: answers that simply state no probability.
    excluded_invalid: int = 0
    excluded_valid: int = 0
    brier: MetricValue = None
    ece: MetricValue = None
    cal_bins: list[dict] = field(default_factory=list)


def metrics_for(aq: ArmQuestion) -> Metrics:
    question = aq.question
    scored = aq.scored
    y_true = [s.gold for s in scored]
    y_pred = [s.value for s in scored]
    m = Metrics(
        arm=aq.arm,
        question=question,
        n=len(scored),
        accuracy=mean(1.0 if s.correct else 0.0 for s in scored) if scored else None,
        macro_f1=_macro_f1(question, y_true, y_pred),
        classes=per_class(question, y_true, y_pred),
        invalid_rate=(sum(1 for s in scored if s.invalid) / len(scored))
        if scored
        else None,
    )
    if not scored:
        m.na.append(
            {
                "metric": "all",
                "reason": f"no scored rows for {question.id} "
                f"({aq.no_gold} row(s) without gold)",
            }
        )
        return m

    for s in scored:
        if s.invalid:
            reason = _invalid_reason(s.row)
            m.invalid_reasons[reason] = m.invalid_reasons.get(reason, 0) + 1
    m.excluded_invalid = sum(
        1 for s in scored if s.p_decision is None and s.invalid
    )
    m.excluded_valid = sum(
        1 for s in scored if s.p_decision is None and not s.invalid
    )

    pairs = [(s.p_decision, s.correct) for s in scored if s.p_decision is not None]
    m.cal_n = len(pairs)
    m.cal_excluded = len(scored) - len(pairs)
    if pairs:
        summary = calibration(pairs)
        m.brier, m.ece, m.cal_bins = summary["brier"], summary["ece"], summary["bins"]
    else:
        reason = (
            f"no row of {aq.arm} carries a usable p_decision for {question.id} "
            "(confidence and probabilities are both absent or null)"
        )
        m.na.append({"metric": "brier", "reason": reason})
        m.na.append({"metric": "ece", "reason": reason})

    if question.positive is not None:
        positive = question.positive
        labels = set(question.labels)
        m.positives = sum(1 for gold in y_true if gold == positive)
        answered = [s for s in scored if s.value in labels]
        m.tp = sum(1 for s in answered if s.value == positive and s.gold == positive)
        m.fp = sum(1 for s in answered if s.value == positive and s.gold != positive)
        m.fn = sum(1 for s in answered if s.value != positive and s.gold == positive)
        m.tn = sum(1 for s in answered if s.value != positive and s.gold != positive)
        m.no_call = len(scored) - len(answered)
        m.no_call_invalid = sum(1 for s in scored if s.value not in labels and s.invalid)
        m.no_call_positives = sum(
            1 for s in scored if s.value not in labels and s.gold == positive
        )
        # F1 keeps its definition over every scored row, so a no-call row counts as
        # a missed positive, matching recall's denominator (all gold positives).
        m.f1_positive = _positive_f1(m.tp, m.fp, m.positives - m.tp)
        m.precision_positive = m.tp / (m.tp + m.fp) if (m.tp + m.fp) else None
        m.recall_positive = m.tp / m.positives if m.positives else None
        m.precision_ci95 = wilson_interval(m.tp, m.tp + m.fp)
        m.recall_ci95 = wilson_interval(m.tp, m.positives)
        m.f1_ci95 = bootstrap_f1_interval(
            [s.gold == positive for s in scored], [s.value == positive for s in scored]
        )
        m.ci_note = (
            f"{m.positives} positive{'s' if m.positives != 1 else ''} in this frame"
        )
        for metric, value in (
            ("precision_ci95", m.precision_ci95),
            ("recall_ci95", m.recall_ci95),
            ("f1_ci95", m.f1_ci95),
        ):
            if value is None:
                m.na.append(
                    {
                        "metric": metric,
                        "reason": "no positives (or no positive predictions) in the "
                        "scored subset, so the interval has no denominator",
                    }
                )
        scored_pairs = [
            (1 if s.gold == positive else 0, s.p_positive)
            for s in scored
            if s.p_positive is not None
        ]
        if not scored_pairs:
            m.na.append(
                {
                    "metric": "pr_auc",
                    "reason": f"no {aq.arm} row carries p_positive for {question.id}",
                }
            )
        elif len({label for label, _ in scored_pairs}) < 2:
            m.na.append(
                {
                    "metric": "pr_auc",
                    "reason": "the scored subset is single-class after dropping rows "
                    f"without p_positive ({len(scored_pairs)} of {len(scored)} rows kept)",
                }
            )
        else:
            from sklearn.metrics import average_precision_score

            m.pr_auc = float(
                average_precision_score(
                    [label for label, _ in scored_pairs],
                    [p for _, p in scored_pairs],
                )
            )
    else:
        reason = (
            f"{question.id} is a multi-label question: it has no positive class, so "
            "positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent "
            "intervals) are not defined for it and are not quantified"
        )
        for metric in ("pr_auc", "f1_positive", "precision_positive", "recall_positive"):
            m.na.append({"metric": metric, "reason": reason})

    if question.ordinal:
        usable = [
            (ORDINAL[s.gold], ORDINAL[s.value])
            for s in scored
            if s.gold in ORDINAL and s.value in ORDINAL
        ]
        m.ordinal_n = len(usable)
        m.ordinal_excluded = len(scored) - len(usable)
        if usable:
            m.mae = mean(abs(a - b) for a, b in usable)
            m.plus_minus_1 = mean(1.0 if abs(a - b) <= 1 else 0.0 for a, b in usable)
        else:
            reason = (
                "no row has both an ordinal gold and an ordinal prediction "
                f"({len(scored)} scored rows)"
            )
            m.na.append({"metric": "mae", "reason": reason})
            m.na.append({"metric": "within_one", "reason": reason})
    return m


# --------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------


def floor_rows(
    question: Question,
    scored: Sequence[Scored],
    goldbook: GoldBook,
) -> tuple[list[dict], list[str]]:
    """``majority_class`` and ``dummy_stratified`` rows for one question.

    The majority label is the most frequent gold **in the scored subset** (the
    same subset every arm is scored on); the stratified dummy is fitted on the
    train+validation gold for the question, so it is a genuine no-skill floor
    rather than a copy of the test gold.
    """
    rows: list[dict] = []
    notes: list[dict] = []
    if not scored:
        return rows, [{"metric": "floor", "reason": "no scored rows"}]

    record_ids = [s.record_id for s in scored]
    split = scored[0].split
    gold_by_record = {s.record_id: s.gold for s in scored}
    counts = Counter(s.gold for s in scored)
    majority, majority_n = counts.most_common(1)[0]
    rows.append(
        _floor_rows_for(
            FLOOR_MAJORITY,
            question,
            split,
            record_ids,
            [majority] * len(record_ids),
            None,
            gold_by_record,
        )
    )
    notes.append(
        {
            "metric": FLOOR_MAJORITY,
            "reason": f"always predicts '{majority}' "
            f"({majority_n}/{len(scored)} = {majority_n / len(scored):.1%} of the scored "
            "subset) and carries no distribution, so brier, ece, pr_auc and the routing "
            "sweep are null for it",
        }
    )

    values, why = goldbook.train_values(question)
    if not values:
        notes.append({"metric": FLOOR_DUMMY, "reason": why})
        return rows, notes
    import numpy as np
    from sklearn.dummy import DummyClassifier

    fit_x = np.zeros((len(values), 1))
    forest = DummyClassifier(strategy="stratified", random_state=0).fit(fit_x, values)
    predict_x = np.zeros((len(record_ids), 1))
    predicted = [str(v) for v in forest.predict(predict_x)]
    proba = forest.predict_proba(predict_x)
    classes = [str(c) for c in forest.classes_]
    probabilities = [
        {label: float(p) for label, p in zip(classes, row)} for row in proba
    ]
    rows.append(
        _floor_rows_for(
            FLOOR_DUMMY,
            question,
            split,
            record_ids,
            predicted,
            probabilities,
            gold_by_record,
        )
    )
    notes.append(
        {
            "metric": FLOOR_DUMMY,
            "reason": "sklearn DummyClassifier(strategy='stratified', random_state=0) "
            f"fitted on {len(values)} train+validation gold rows, predicting on the "
            f"{len(record_ids)} scored records; its predict_proba is one-hot on the "
            "sampled label, so it is a certainty-asserting floor",
        }
    )
    return rows, notes


def _floor_rows_for(
    arm: str,
    question: Question,
    split: str,
    record_ids: Sequence[str],
    values: Sequence[str],
    probabilities: list[dict[str, float]] | None,
    gold_by_record: dict[str, str],
) -> dict:
    rows: list[dict] = []
    for index, (record_id, value) in enumerate(zip(record_ids, values)):
        gold = gold_by_record.get(record_id)
        rows.append(
            {
                "record_id": record_id,
                "arm": arm,
                "question": question.id,
                "split": split,
                "value": value,
                "probabilities": None
                if probabilities is None
                else probabilities[index],
                "confidence": None,
                "gold": gold,
                "correct": value == gold,
                "latency_ms": None,
                "tokens_in": None,
                "tokens_out": None,
                "invalid": False,
                "meta": {"synthetic": "floor"},
            }
        )
    return {"arm": arm, "question": question, "rows": rows}


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def routing_table(
    question: Question,
    arm_metrics: dict[str, Metrics],
    arm_rows: dict[str, ArmQuestion],
) -> list[dict]:
    """Coverage/precision/recall at each threshold, per arm. Binary question only."""
    positive = question.positive
    if positive is None:
        return []
    out: list[dict] = []
    for arm, metrics in arm_metrics.items():
        rows = arm_rows[arm].scored
        positives = metrics.positives
        no_score = sum(1 for s in rows if s.p_positive is None)
        for tau in TAUS:
            acted = [
                s for s in rows if s.p_positive is not None and s.p_positive >= tau
            ]
            caught = [s for s in acted if s.gold == positive]
            entry: dict[str, Any] = {
                "arm": arm,
                "tau": tau,
                "n": len(rows),
                "no_score": no_score,
                "acted": len(acted),
                "coverage": (len(acted) / len(rows)) if rows else None,
                "tp": len(caught),
                "fp": len(acted) - len(caught),
                "positives": positives,
                "positives_acted": len(caught),
                "recall": (len(caught) / positives) if positives else None,
                "precision": (len(caught) / len(acted)) if acted else None,
                "fn_missed": positives - len(caught),
                "note": "",
            }
            if not rows:
                entry["note"] = "no scored rows"
            elif not acted:
                entry["note"] = (
                    f"nothing reaches p_positive >= {tau:g} "
                    f"({no_score} row(s) carry no p_positive)"
                    if no_score
                    else f"nothing reaches p_positive >= {tau:g}"
                )
            out.append(entry)
    return out


# --------------------------------------------------------------------------
# Cost and latency
# --------------------------------------------------------------------------


def latency_stats(rows: Sequence[dict]) -> dict[str, Any]:
    """p50/p95 latency per row, mean tokens per row, per-posting latency."""
    per_row: list[float] = []
    for row in rows:
        value = row.get("latency_ms")
        if value is None:
            continue
        try:
            per_row.append(float(value))
        except (TypeError, ValueError):
            continue
    stats: dict[str, Any] = {"n_rows": len(rows), "n_latency": len(per_row)}
    if per_row:
        ordered = sorted(per_row)
        stats["mean_ms"] = mean(per_row)
        stats["p50_ms"] = _quantile(ordered, 0.50)
        stats["p95_ms"] = _quantile(ordered, 0.95)
    else:
        stats["mean_ms"] = stats["p50_ms"] = stats["p95_ms"] = None
    tokens_in = [
        int(row["tokens_in"]) for row in rows if row.get("tokens_in") is not None
    ]
    tokens_out = [
        int(row["tokens_out"]) for row in rows if row.get("tokens_out") is not None
    ]
    stats["n_tokens"] = max(len(tokens_in), len(tokens_out))
    stats["tokens_in"] = mean(tokens_in) if tokens_in else None
    stats["tokens_out"] = mean(tokens_out) if tokens_out else None

    # Cost per posting: one completion can answer several questions and the
    # runner repeats that completion's timing on each row it produced, so
    # identical per-record values are counted once.
    by_record: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("latency_ms") is None:
            continue
        try:
            by_record[str(row["record_id"])].append(round(float(row["latency_ms"]), 3))
        except (TypeError, ValueError):
            continue
    stats["n_records"] = len(by_record)
    stats["posting_ms"] = (
        mean(sum(sorted(set(values))) for values in by_record.values())
        if by_record
        else None
    )
    return stats


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy/matplotlib convention)."""
    if not ordered:
        raise ValueError("empty")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------


def fmt(value: MetricValue, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_ci(value: tuple[float, float] | None) -> str:
    """``"[0.60, 0.85]"`` for an interval, ``n/a`` when it has no denominator."""
    if value is None:
        return "n/a"
    return f"[{value[0]:.2f}, {value[1]:.2f}]"


def fmt_pct(value: MetricValue, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        cells = ["" if cell is None else str(cell) for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ordered_arms(names: Iterable[str]) -> list[str]:
    found = set(names)
    ordered = [a for a in ARM_ORDER if a in found]
    return ordered + sorted(found - set(ordered))


# --------------------------------------------------------------------------
# Frame check
# --------------------------------------------------------------------------


def frame_check(analysis: Analysis) -> dict[str, dict]:
    """Which records each arm answered for each question, and whether they agree.

    Comparing arms only makes sense on the same postings: an arm that answered a
    different set of records has numbers that are not directly comparable with the
    others. The majority set is the record set the largest number of arms
    answered (ties broken by the alphabetically first arm owning it), and each
    mismatch is measured against it. Floors are not part of the frame: they are
    scored on the scored subset of one arm by construction.
    """
    frame: dict[str, dict] = {}
    for question in analysis.questions:
        per_arm = {
            arm: {s.record_id for s in aq.scored}
            for arm, aq in analysis.scored.get(question.id, {}).items()
            if arm not in FLOORS
        }
        if not per_arm:
            continue
        sets = list(per_arm.values())
        union = set().union(*sets)
        shared = set.intersection(*sets) if sets else set()
        tally = Counter(frozenset(ids) for ids in per_arm.values())
        best_count = max(tally.values())
        owners = sorted(
            arm for arm, ids in per_arm.items() if tally[frozenset(ids)] == best_count
        )
        majority_arm = owners[0]
        majority_set = per_arm[majority_arm]
        mismatched = sorted(arm for arm, ids in per_arm.items() if ids != majority_set)
        frame[question.id] = {
            "union": len(union),
            "shared": len(shared),
            "per_arm": {arm: len(ids) for arm, ids in sorted(per_arm.items())},
            "majority_set_size": len(majority_set),
            "majority_arms": owners,
            "majority_arm": majority_arm,
            "mismatch": bool(mismatched),
            "mismatched_arms": mismatched,
            "symdiff_vs_majority": {
                arm: len(per_arm[arm] ^ majority_set) for arm in mismatched
            },
            "only_in_arm": {
                arm: len(per_arm[arm] - majority_set) for arm in mismatched
            },
            "missing_from_arm": {
                arm: len(majority_set - per_arm[arm]) for arm in mismatched
            },
        }
    return frame


def frame_warnings(analysis: Analysis) -> list[str]:
    """One ``WARN`` line per arm that answered a different set of records."""
    lines: list[str] = []
    for question_id, entry in analysis.frame.items():
        if not entry["mismatch"]:
            continue
        aligned = ", ".join(entry["majority_arms"])
        lines.append(
            f"WARN frame `{question_id}`: {len(entry['mismatched_arms'])} of "
            f"{len(entry['per_arm'])} arm(s) answered a different set of records "
            f"(union {entry['union']}, shared {entry['shared']}, majority set "
            f"{entry['majority_set_size']} answered by {aligned})"
        )
        for arm in entry["mismatched_arms"]:
            lines.append(
                f"WARN frame `{question_id}`: arm `{arm}` answered "
                f"{entry['per_arm'][arm]} record(s), symmetric difference "
                f"{entry['symdiff_vs_majority'][arm]} vs the majority set of "
                f"{entry['majority_set_size']} "
                f"({entry['only_in_arm'][arm]} record(s) it answered that the others did "
                f"not, {entry['missing_from_arm'][arm]} it did not answer that they did); "
                "arms scored on a different set of postings are not directly comparable."
            )
    return lines


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


@dataclass
class Analysis:
    results_dir: Path
    loaded: Loaded
    goldbook: GoldBook
    questions: tuple[Question, ...]
    #: question id -> arm -> ArmQuestion (real arms, then floors)
    scored: dict[str, dict[str, ArmQuestion]] = field(default_factory=dict)
    metrics: dict[str, dict[str, Metrics]] = field(default_factory=dict)
    guard: list[dict] = field(default_factory=list)
    guard_ok: bool = True
    #: question id -> the arm whose scored subset the floors and the gold
    #: distribution of that question are computed on (the largest arm).
    subset_reference: dict[str, str] = field(default_factory=dict)
    #: question id -> the record frame every arm answered, and any arm that did
    #: not answer it. See ``frame_check``.
    frame: dict[str, dict] = field(default_factory=dict)
    #: the caveat printed under the cost table (``--cost-note``).
    cost_note: str = DEFAULT_COST_NOTE
    #: ``{"arm", "question", "metric", "reason"}``: every metric that could not be
    #: produced, and every scoring caveat that changes a number. Single source for
    #: report.md section 8 and for metrics.json's ``n_a``.
    na: list[dict] = field(default_factory=list)

    def arms_for(self, question_id: str) -> list[str]:
        return list(self.metrics.get(question_id, {}))

    def note(
        self,
        reason: str,
        metric: str,
        arm: str | None = None,
        question: str | None = None,
    ) -> None:
        entry = {
            "arm": arm,
            "question": question,
            "metric": metric,
            "reason": reason,
        }
        if entry not in self.na:
            self.na.append(entry)


def analyse(results_dir: Path, cost_note: str = DEFAULT_COST_NOTE) -> Analysis:
    loaded = load_results(results_dir)
    goldbook = GoldBook()
    analysis = Analysis(
        results_dir=results_dir,
        loaded=loaded,
        goldbook=goldbook,
        questions=select(None),
        cost_note=cost_note,
    )

    for question in analysis.questions:
        arm_rows = {
            arm: build_arm_question(arm, question, rows, goldbook)
            for arm, rows in loaded.arms.items()
        }
        scored: dict[str, ArmQuestion] = {
            arm: aq for arm, aq in arm_rows.items() if aq.n
        }
        metrics: dict[str, Metrics] = {
            arm: metrics_for(aq) for arm, aq in scored.items()
        }

        if scored:
            reference = max(scored.values(), key=lambda aq: aq.n)
            analysis.subset_reference[question.id] = reference.arm
            floors, notes = floor_rows(question, reference.scored, goldbook)
            for note in notes:
                analysis.note(note["reason"], note["metric"], question=question.id)
            reference_ids = {s.record_id for s in reference.scored}
            reference_gold = {s.record_id: s.gold for s in reference.scored}
            for floor in floors:
                aq = ArmQuestion(arm=floor["arm"], question=question)
                for row in floor["rows"]:
                    gold = reference_gold.get(row["record_id"])
                    if gold is None or row["record_id"] not in reference_ids:
                        continue
                    aq.scored.append(
                        Scored(
                            row=row,
                            record_id=row["record_id"],
                            split=row["split"],
                            arm=aq.arm,
                            gold=gold,
                            value=row["value"],
                            correct=row["value"] == gold,
                            p_decision=p_decision(row),
                            p_positive=p_positive(row, question),
                            invalid=False,
                            gold_source="floor",
                            latency_ms=None,
                            tokens_in=None,
                            tokens_out=None,
                        )
                    )
                if not aq.scored:
                    continue
                sizes = {a: x.n for a, x in scored.items()}
                if len(set(sizes.values())) > 1:
                    analysis.note(
                        "the floor is scored on the "
                        f"{len(aq.scored)}-row subset of the largest arm "
                        f"(`{reference.arm}`); the arms are on different subsets ("
                        + ", ".join(f"{a}={n}" for a, n in sorted(sizes.items()))
                        + "), so floor and arm rows are not on identical record sets",
                        "floor_subset",
                        arm=aq.arm,
                        question=question.id,
                    )
                scored[aq.arm] = aq
                metrics[aq.arm] = metrics_for(aq)

        analysis.scored[question.id] = scored
        analysis.metrics[question.id] = metrics
        for arm, aq in scored.items():
            for entry in metrics[arm].na:
                analysis.note(
                    entry["reason"], entry["metric"], arm=arm, question=question.id
                )
            if aq.no_gold:
                analysis.note(
                    f"{aq.no_gold} row(s) skipped because gold is null",
                    "scored_rows",
                    arm=arm,
                    question=question.id,
                )
            if aq.unparseable:
                analysis.note(
                    f"{aq.unparseable} row(s) carry a value outside the declared label "
                    "set; they are scored as wrong, and a per-class table cannot place "
                    "them in any class",
                    "unparseable_value",
                    arm=arm,
                    question=question.id,
                )
            if aq.correct_mismatch:
                analysis.note(
                    f"{aq.correct_mismatch} row(s) have a stored `correct` that "
                    "disagrees with value==gold; the recomputed value is used",
                    "correct_field",
                    arm=arm,
                    question=question.id,
                )
            if aq.gold_from_row:
                analysis.note(
                    f"{aq.gold_from_row} row(s) scored against the row's own `gold` "
                    "field because their record is not in the dataset split",
                    "gold_source",
                    arm=arm,
                    question=question.id,
                )
            if metrics[arm].n and metrics[arm].invalid_rate:
                invalid = sum(1 for s in aq.scored if s.invalid)
                reasons = _reason_summary(metrics[arm].invalid_reasons)
                analysis.note(
                    f"invalid_json_rate = {fmt_pct(metrics[arm].invalid_rate)}: "
                    f"{invalid} of {metrics[arm].n} rows failed the arm's own output "
                    f"validation and are scored wrong (arm-reported reasons: {reasons}); "
                    f"{metrics[arm].excluded_invalid} of them are also excluded from "
                    "Brier/ECE because they carry no probability",
                    "invalid_json_rate",
                    arm=arm,
                    question=question.id,
                )

    analysis.guard = leakage_guard(analysis)
    covered = {entry["arm"] for entry in analysis.guard}
    for arm in ordered_arms(analysis.loaded.arms):
        if arm not in covered and arm not in FLOORS:
            analysis.note(
                "produced no `salary_range_stated` rows, so the leakage guard does "
                "not cover it",
                "leakage_guard",
                arm=arm,
                question="salary_range_stated",
            )
    for arm, entry in sorted(loaded.status.items()):
        if entry.get("status") not in (None, "ok") or arm not in loaded.arms:
            analysis.note(
                "arm status "
                f"`{entry.get('status', '?')}`"
                + (
                    f": {entry['reason']}"
                    if entry.get("reason")
                    else " with no reason recorded"
                ),
                "arm",
                arm=arm,
            )
    analysis.guard_ok = all(entry["verdict"] != "FAIL" for entry in analysis.guard)

    analysis.frame = frame_check(analysis)
    for question_id, entry in analysis.frame.items():
        for arm in entry["mismatched_arms"]:
            analysis.note(
                f"answered {entry['per_arm'][arm]} record(s) for this question, "
                f"symmetric difference {entry['symdiff_vs_majority'][arm]} against the "
                f"majority set of {entry['majority_set_size']} record(s) "
                f"(union {entry['union']}, shared {entry['shared']}); arms scored on a "
                "different set of postings are not directly comparable",
                "frame",
                arm=arm,
                question=question_id,
            )
    return analysis


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------


def leakage_guard(analysis: Analysis) -> list[dict]:
    """``salary_range_stated`` accuracy must sit within tolerance of the floor.

    ``salary_range`` is platform metadata, not posting text, so no arm that only
    sees the posting can legitimately beat the majority rate; an arm far above
    it has the gold in its input. Floors are excluded: they are built from the
    gold itself, so the check would be vacuous on them.
    """
    question_id = "salary_range_stated"
    metrics = analysis.metrics.get(question_id, {})
    guard: list[dict] = []
    for arm in analysis.arms_for(question_id):
        if arm in FLOORS:
            continue
        m = metrics[arm]
        counts = Counter(s.gold for s in analysis.scored[question_id][arm].scored)
        majority, majority_n = counts.most_common(1)[0]
        majority_rate = majority_n / m.n
        delta = (m.accuracy - majority_rate) * 100.0
        if delta > LEAK_TOLERANCE_PP:
            verdict = "FAIL"
            note = (
                f"{delta:.1f}pp ABOVE the majority rate ('{majority}'): the answer is "
                "platform metadata, so a gap this large means the gold reached the "
                "model's input"
            )
        elif delta < -LEAK_TOLERANCE_PP:
            verdict = "BELOW"
            note = (
                f"{abs(delta):.1f}pp BELOW the majority rate ('{majority}'): below the "
                "floor, not leakage. The arm is not answering the question that was "
                "asked (see the no-rubric ablation) or is not reading the text"
            )
        else:
            verdict = "PASS"
            note = (
                f"within {LEAK_TOLERANCE_PP:g}pp above the majority rate ('{majority}')"
                + (
                    " and at most "
                    f"{abs(delta):.1f}pp below it"
                    if delta < 0
                    else ""
                )
            )
        guard.append(
            {
                "arm": arm,
                "n": m.n,
                "accuracy": m.accuracy,
                "majority": majority_rate,
                "majority_label": majority,
                "majority_count": majority_n,
                "delta_pp": delta,
                "tolerance_pp": LEAK_TOLERANCE_PP,
                "verdict": verdict,
                "note": note,
            }
        )
    return guard


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------


METRIC_DEFS = """\
`p_decision`: the arm's stated probability that its chosen answer is right:
`confidence` if the row carries one, else `probabilities[value]`. Rows where it is
`null` are excluded from Brier/ECE and counted in the *excluded* column. This is
the same rule for every arm: classifier.dev's calibrated `confidence` for Jev, the
score of the predicted label for tfidf/gliner, and the LLM's self-reported
`confidence` field.

`p_positive` (binary questions only): the arm's implied P(fraudulent):
`probabilities["fraudulent"]` when a distribution exists, else `confidence` when
the prediction is `fraudulent` and `1 - confidence` when it is `legitimate`. Used
for PR-AUC and for the routing sweep. Multi-label questions have no positive
class, so those metrics are `n/a` there.

Brier = mean over scored rows of `(p_decision - correct)^2`, with `correct` as
0/1. ECE = 10 equal-width bins on [0, 1],
`sum_b (n_b / N) * |mean(p_decision)_b - accuracy_b|`; the last bin is closed on
the right so `p_decision == 1.0` is counted. A row whose `p_decision` is `null`
leaves both and is counted as excluded.

`invalid_json_rate` = share of rows with `invalid == true` (the arm's own output
failed its parse/validation; such rows are wrong by construction). It is a column
in the tables, never hidden.

Floors: `majority_class` always predicts the most frequent gold label of the
scored subset and carries no distribution. `dummy_stratified` is
`sklearn.dummy.DummyClassifier(strategy="stratified", random_state=0)` fitted on
the train+validation gold rows for that question and evaluated on the scored
records, so it carries probabilities and is also a calibration floor; in this
sklearn version its `predict_proba` is one-hot on the sampled label, i.e. it
asserts certainty.

Pitfalls this table invites:

* a `1.0`-heavy `p_decision` column gives a low Brier for a reason that has
  nothing to do with being right: read it next to accuracy;
* `not_applicable` is a real gold label for `required_experience`: it is scored in
  accuracy and macro-F1, and excluded from MAE and +/-1 because it is not a
  position on the ladder;
* macro-F1 is computed over the question's declared label set, so a label nobody
  predicted still contributes a zero instead of disappearing;
* `p_positive` derived from `1 - confidence` inherits the arm's confidence
  semantics: it is an implied value, not a measured distribution.
"""


def _section_run(analysis: Analysis) -> str:
    loaded = analysis.loaded
    lines = ["## 1. What was run", ""]
    lines.append(
        md_table(
            ["file", "rows read", "misparsed"],
            [
                [name, count, loaded.malformed.get(name, 0)]
                for name, count in loaded.files
            ]
            or [["(no readable JSONL files)", 0, 0]],
        )
    )
    lines.append("")
    lines.append(
        md_table(
            ["arm", "rows", "source file(s)", "splits", "status", "reason"],
            [
                [
                    arm,
                    len(rows),
                    ", ".join(loaded.sources.get(arm, [])),
                    ", ".join(sorted({str(r.get("split")) for r in rows})),
                    (loaded.status.get(arm) or {}).get(
                        "status", "no entry in _unavailable.json"
                    ),
                    (loaded.status.get(arm) or {}).get("reason", ""),
                ]
                for arm, rows in sorted(
                    loaded.arms.items(), key=lambda kv: ordered_arms([kv[0]])
                )
            ]
            or [["(no arms)", 0, "", "", "", ""]],
        )
    )
    lines.append("")
    extra: list[str] = []
    for arm, entry in sorted(loaded.status.items()):
        if arm in loaded.arms:
            continue
        extra.append(
            f"* `{arm}` is in `_unavailable.json` with status "
            f"`{entry.get('status', '?')}` and produced no rows"
            + (
                f": {entry['reason']}"
                if entry.get("reason")
                else " and no reason was recorded"
            )
            + "."
        )
    for arm, names in sorted(loaded.sources.items()):
        if any(name != f"{arm}.jsonl" for name in names):
            extra.append(
                f"* `{arm}` rows came from file(s) {', '.join(names)}: the arm identity "
                "is the row's `arm` field, not the filename."
            )
    if loaded.status_note.startswith("no "):
        extra.append(
            f"* {loaded.status_note}: arms that failed without leaving rows are not "
            "visible here."
        )
    elif loaded.status_note.startswith("unreadable"):
        extra.append(f"* `_unavailable.json` was {loaded.status_note}.")
    if extra:
        lines += extra + [""]
    lines.append(
        f"Gold source: {analysis.goldbook.from_dataset} row(s) scored against the "
        f"dataset gold from `triage/data.py`, {analysis.goldbook.from_row} against the "
        "`gold` field of their own result row (records absent from that dataset split, "
        "e.g. synthetic fixtures)."
    )
    if analysis.goldbook.mismatch:
        lines.append("")
        lines.append(
            "**Integrity problem**: the row's `gold` disagrees with the dataset gold for "
            + ", ".join(
                f"`{question}` ({split}) x{count}"
                for (question, split), count in sorted(analysis.goldbook.mismatch.items())
            )
            + "; the dataset value was used, and the affected arm's rows should be "
            "regenerated before the numbers are trusted."
        )
    if analysis.goldbook.errors:
        lines.append("")
        lines.append(
            "Dataset gold could not be read for: "
            + ", ".join(
                f"`{split}` ({reason})"
                for split, reason in sorted(analysis.goldbook.errors.items())
            )
            + ". Rows fall back to their own `gold` field and both floors are `n/a`."
        )
    lines.append("")
    lines.append("### Frame: which records each arm answered")
    lines.append("")
    if analysis.frame:
        lines.append(
            "A question is scored only where its gold exists, so the sample size per "
            "question differs by design. What must not differ is which records an arm "
            "answered for the same question: an arm on a different set of postings has "
            "rows that are not directly comparable with the others. `shared` is the "
            "intersection across every arm, `majority set` is the record set that the "
            "largest number of arms answered, and mismatches are measured against it."
        )
        lines.append("")
        frame_rows = []
        for question_id, entry in analysis.frame.items():
            frame_rows.append(
                [
                    question_id,
                    len(entry["per_arm"]),
                    entry["union"],
                    entry["shared"],
                    entry["majority_set_size"],
                    ", ".join(f"{arm}={count}" for arm, count in entry["per_arm"].items()),
                    "match"
                    if not entry["mismatch"]
                    else "WARN: "
                    + ", ".join(
                        f"{arm} (symdiff {entry['symdiff_vs_majority'][arm]})"
                        for arm in entry["mismatched_arms"]
                    ),
                ]
            )
        lines.append(
            md_table(
                ["question", "arms", "union", "shared", "majority set",
                 "records per arm", "frame"],
                frame_rows,
            )
        )
        lines.append("")
        lines.append(
            "Frame source: "
            + (
                f"`{_display_path(analysis.results_dir)}/_frame.json`, the runner's own "
                "record."
                if analysis.loaded.frame_recorded is not None
                else "derived from the record ids in the results; no frame was recorded, "
                "so this cannot distinguish a random sample from a prefix."
            )
        )
        lines.append("")
        warnings = frame_warnings(analysis)
        if warnings:
            lines += warnings + [""]
        else:
            lines.append(
                "Every arm answered the same records for every question it scored: no "
                "WARN."
            )
            lines.append("")
    else:
        lines.append("n/a: no arm produced scorable rows, so there is no frame.")
        lines.append("")
    lines.append(
        "Questions with scored rows: "
        + (
            ", ".join(
                f"`{q}` ({sum(m.n for m in analysis.metrics[q].values())} arm-rows)"
                for q in QUESTION_IDS
                if analysis.metrics.get(q)
            )
            or "(none)"
        )
        + "."
    )
    return "\n".join(lines)


def _question_table(
    analysis: Analysis, question: Question
) -> str:
    metrics = analysis.metrics[question.id]
    lines = [f"### `{question.id}` ({question.kind}, {len(question.labels)} labels)", ""]
    headers = ["arm", "n", "accuracy", "macro-F1", "invalid_json_rate"]
    if question.positive:
        headers += [
            "#positives",
            f"F1({question.positive})",
            f"precision({question.positive})",
            f"recall({question.positive})",
            "PR-AUC (AP)",
            "precision_ci95",
            "recall_ci95",
            "f1_ci95",
            "ci_note",
            "TP/FP/FN/TN",
            "no_call",
        ]
    if question.ordinal:
        headers += ["MAE", "+/-1 acc", "MAE n", "MAE excluded"]
    if question.id == "salary_range_stated":
        headers += ["majority rate", "accuracy - majority (pp)"]
    rows: list[list[Any]] = []
    for arm, m in metrics.items():
        counts = Counter(s.gold for s in analysis.scored[question.id][arm].scored)
        majority_rate = counts.most_common(1)[0][1] / m.n if counts and m.n else None
        row: list[Any] = [
            arm,
            m.n,
            fmt(m.accuracy),
            fmt(m.macro_f1),
            fmt(m.invalid_rate),
        ]
        if question.positive:
            row += [
                m.positives,
                fmt(m.f1_positive),
                fmt(m.precision_positive),
                fmt(m.recall_positive),
                fmt(m.pr_auc),
                fmt_ci(m.precision_ci95),
                fmt_ci(m.recall_ci95),
                fmt_ci(m.f1_ci95),
                m.ci_note or "n/a",
                f"{m.tp}/{m.fp}/{m.fn}/{m.tn}",
                m.no_call,
            ]
        if question.ordinal:
            row += [fmt(m.mae), fmt(m.plus_minus_1), m.ordinal_n, m.ordinal_excluded]
        if question.id == "salary_range_stated":
            row += [
                fmt(majority_rate),
                fmt(
                    (m.accuracy - majority_rate) * 100
                    if majority_rate is not None and m.accuracy is not None
                    else None,
                    1,
                ),
            ]
        rows.append(row)
    lines.append(md_table(headers, rows))
    lines.append("")
    lines.append(
        "Per-class detail, over the same scored rows as the table above (all n of them, "
        "not a per-label subset): precision/recall/F1 come from sklearn with the "
        "question's declared label set, so a label nobody predicted contributes a zero to "
        "macro-F1 rather than disappearing. A precision cell is `n/a` when the arm never "
        f"predicted that label: {PRECISION_UNDEFINED}."
    )
    lines.append("")
    class_rows: list[list[Any]] = []
    for arm, m in metrics.items():
        for entry in m.classes:
            class_rows.append(
                [
                    arm,
                    entry["label"],
                    fmt(entry["precision"]),
                    fmt(entry["recall"]),
                    fmt(entry["f1"]),
                    entry["support"],
                ]
            )
    lines.append(
        md_table(
            ["arm", "label", "precision", "recall", "F1", "support"], class_rows
        )
    )
    lines.append("")
    if question.ordinal:
        lines.append(
            "MAE and +/-1 use `triage.schema.ORDINAL` ("
            + ", ".join(f"`{k}={v}`" for k, v in ORDINAL.items())
            + "); rows whose gold or prediction is `not_applicable`, empty or outside the "
            "label set are excluded from both and counted in *MAE excluded*. "
            "`not_applicable` still counts in accuracy and macro-F1."
        )
        lines.append("")
    if question.id == "salary_range_stated":
        lines.append(
            "Control question: the answer is platform metadata, so an arm reading only the "
            "posting cannot beat the majority rate. See the leakage guard in section 9."
        )
        lines.append("")
    if question.positive is not None:
        positives = next(
            (m.positives for m in metrics.values() if m.positives), 0
        )
        lines.append(
            "Confusion convention: the four cells are read over every scored row and "
            "they sum to n only after the `no_call` column, which counts rows where the "
            "arm produced no label from the declared set (an empty reply, a junk value, "
            "or a row it flagged invalid). A `no_call` row is neither a false positive "
            "nor a true negative: an empty reply made no call at all, so scoring it as a "
            "correct negative would be the exact failure this benchmark exists to catch. "
            "`no_call` rows are wrong in accuracy, they count as missed positives when "
            "their gold is the positive label (so recall and the positive-class F1 carry "
            "them), and `invalid_json_rate` counts the subset of them the arm itself "
            "flagged. Check the arithmetic with: `accuracy x n == TP + TN` and "
            "`TP + FP + FN + TN + no_call == n`."
        )
        lines.append("")
        mismatched = [
            (arm, m) for arm, m in metrics.items()
            if m.accuracy is not None and m.n and abs(
                m.accuracy * m.n - (m.tp + m.tn)
            ) > 1e-6
        ]
        if mismatched:
            lines.append(
                "Reconciliation FAIL for: "
                + ", ".join(
                    f"`{arm}` (accuracy x n = {m.accuracy * m.n:.1f} against "
                    f"TP + TN = {m.tp + m.tn})"
                    for arm, m in mismatched
                )
            )
            lines.append("")
        lines.append(
            "Intervals: `precision_ci95` and `recall_ci95` are two-sided 95 percent Wilson "
            "score intervals on the counts already in the table (TP/(TP+FP) and "
            "TP/positives); `f1_ci95` is a deterministic bootstrap ("
            f"{BOOTSTRAP_RESAMPLES} resamples of the scored rows with random.Random(0), "
            "percentiles by linear interpolation), so a rerun gives the same interval. "
            f"`ci_note` records the frame: {positives} positives in this frame."
        )
        lines.append("")
        lines.append(
            f"With {positives} positives in this frame, the positive-class estimates carry "
            "intervals roughly plus or minus ten percentage points wide (a little wider for "
            "precision, see the ci95 columns), the arms' intervals separate on recall and "
            "nearly separate on precision, and the same two arms were also run over the "
            "whole split "
            "where the prevalence is 4.3 percent and the point estimates matched "
            "(precision 0.74 both, recall 0.66 versus 0.65), so the ordering does not "
            "depend on the frame or on rebalancing. That whole-split comparison comes from "
            "a run outside this results directory; the intervals above are computed from "
            "the rows here."
        )
        lines.append("")
        lines.append(
            "Intervals in this frame: "
            + "; ".join(
                f"`{arm}` precision {fmt_ci(m.precision_ci95)} recall {fmt_ci(m.recall_ci95)}"
                for arm, m in metrics.items()
            )
            + "."
        )
        lines.append("")
        lines.append(
            "The same caveat applies to the multi-label questions "
            "(`required_experience`, `required_education`): their scored subsets are larger, "
            "so the intervals would be narrower, and they are not quantified here because "
            "precision and recall over a multi-label argmax are not a single binomial "
            "proportion."
        )
        lines.append("")
    return "\n".join(lines)


def _section_questions(analysis: Analysis) -> str:
    lines = ["## 3. Per-question metrics", ""]
    lines.append(
        "Every table carries both floors: `majority_class` (most frequent gold label of "
        "the scored subset) and `dummy_stratified` (fitted on train+validation gold, see "
        "section 2). No arm is shown without them."
    )
    lines.append("")
    for question in analysis.questions:
        if not analysis.metrics.get(question.id):
            lines.append(f"### `{question.id}`: n/a")
            lines.append("")
            lines.append(
                "No arm produced a row for this question in this results directory, so "
                "there is no table: nothing is invented to fill it."
            )
            lines.append("")
            continue
        lines.append(_question_table(analysis, question))
    return "\n".join(lines)


def _section_calibration(analysis: Analysis) -> str:
    lines = ["## 4. Calibration", ""]
    lines.append(
        "Brier and ECE are computed on `p_decision` (section 2), which is why an arm that "
        "cannot state a probability shows `n/a` rather than a substituted zero."
    )
    lines.append("")
    rows: list[list[Any]] = []
    arm_pairs: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    arm_rows: Counter = Counter()
    arm_invalid: Counter = Counter()
    arm_excluded_invalid: Counter = Counter()
    arm_excluded_valid: Counter = Counter()
    arm_invalid_reasons: dict[str, Counter] = defaultdict(Counter)
    for question in analysis.questions:
        for arm, m in analysis.metrics.get(question.id, {}).items():
            rows.append(
                [
                    arm,
                    question.id,
                    m.n,
                    m.cal_n,
                    m.cal_excluded,
                    m.excluded_invalid,
                    m.excluded_valid,
                    fmt(m.brier),
                    fmt(m.ece),
                    fmt(m.invalid_rate),
                ]
            )
            for scored in analysis.scored[question.id][arm].scored:
                arm_rows[arm] += 1
                arm_invalid[arm] += int(scored.invalid)
                if scored.p_decision is None:
                    if scored.invalid:
                        arm_excluded_invalid[arm] += 1
                    else:
                        arm_excluded_valid[arm] += 1
                else:
                    arm_pairs[arm].append((scored.p_decision, scored.correct))
            for reason, count in m.invalid_reasons.items():
                arm_invalid_reasons[arm][reason] += count
    for arm in ordered_arms(arm_rows):
        summary = calibration(arm_pairs.get(arm, []))
        rows.append(
            [
                f"**{arm} (all questions)**",
                "all",
                arm_rows[arm],
                summary["n"],
                arm_rows[arm] - summary["n"],
                arm_excluded_invalid[arm],
                arm_excluded_valid[arm],
                fmt(summary["brier"]),
                fmt(summary["ece"]),
                fmt(arm_invalid[arm] / arm_rows[arm] if arm_rows[arm] else None),
            ]
        )
    lines.append(
        md_table(
            [
                "arm",
                "question",
                "n scored",
                "n with p_decision",
                "excluded (null p_decision)",
                "of which invalid",
                "excluded but valid",
                "Brier",
                "ECE",
                "invalid_json_rate",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Excluded rows are never dropped silently. *of which invalid* counts rows the arm "
        "itself marked `invalid` (its own output failed to parse or validate) that also "
        "carry no probability; *excluded but valid* counts rows that answered normally yet "
        "state no probability at all, which is why Brier/ECE cannot include them. Both sets "
        "still count as wrong in accuracy, macro-F1 and every other scored metric, and the "
        "n/a list in section 8 names them per arm and question."
    )
    lines.append("")
    invalid_lines = []
    for arm in ordered_arms(arm_rows):
        if not arm_invalid[arm]:
            continue
        reasons = _reason_summary(arm_invalid_reasons[arm])
        invalid_lines.append(
            f"* `{arm}`: {arm_invalid[arm]} of {arm_rows[arm]} rows are `invalid` "
            f"({fmt_pct(arm_invalid[arm] / arm_rows[arm])} of its rows), and "
            f"{arm_excluded_invalid[arm]} of those are excluded from calibration; "
            f"arm-reported reasons: {reasons}."
        )
    if invalid_lines:
        lines.append("Invalid rows, named:")
        lines.append("")
        lines += invalid_lines
        lines.append("")
    fraudulent = analysis.metrics.get("fraudulent")
    if fraudulent:
        lines.append("### Reliability bins, `fraudulent`")
        lines.append("")
        bin_rows: list[list[Any]] = []
        for arm, m in fraudulent.items():
            for index, entry in enumerate(m.cal_bins):
                if not entry["n"]:
                    continue
                bin_rows.append(
                    [
                        arm,
                        f"[{entry['low']:.1f}, {entry['high']:.1f}"
                        + ("]" if index == 9 else ")"),
                        entry["n"],
                        fmt(entry["conf"], 3),
                        fmt(entry["acc"], 3),
                        fmt(
                            abs(entry["conf"] - entry["acc"])
                            if entry["conf"] is not None and entry["acc"] is not None
                            else None,
                            3,
                        ),
                    ]
                )
        lines.append(
            md_table(
                ["arm", "bin", "n", "mean p_decision", "empirical accuracy", "gap"],
                bin_rows,
            )
            if bin_rows
            else "n/a: no arm produced a usable p_decision on `fraudulent`."
        )
        lines.append("")
        lines.append(
            "Empty bins are omitted from the table; the ECE above sums over all ten bins, "
            "empty ones contributing zero."
        )
    else:
        lines.append(
            "n/a: no arm produced `fraudulent` rows in this results directory, so no "
            "reliability table can be shown."
        )
    lines.append("")
    return "\n".join(lines)


def _section_routing(analysis: Analysis) -> str:
    lines = ["## 5. Routing (`fraudulent`)", ""]
    metrics = analysis.metrics.get("fraudulent")
    if not metrics:
        lines.append(
            "n/a: no `fraudulent` rows in this results directory. Routing is defined only "
            "for the binary question."
        )
        return "\n".join(lines)
    lines.append(
        "Acting means **auto-flag as scam**: the record leaves the human review queue and "
        "is treated as fraud. The sweep runs over `p_positive` (section 2). Coverage is the "
        "share of the evaluated records acted on; the remainder is what a human still has "
        "to read. A row with no `p_positive` can never be acted on, so it counts against "
        "coverage."
    )
    lines.append("")
    table = routing_table(by_id("fraudulent"), metrics, analysis.scored["fraudulent"])
    rows: list[list[Any]] = []
    for entry in table:
        rows.append(
            [
                entry["arm"],
                f"{entry['tau']:g}",
                entry["acted"],
                fmt_pct(entry["coverage"]),
                entry["tp"],
                entry["fp"],
                fmt(entry["precision"]),
                fmt(entry["recall"]),
                entry["fn_missed"],
                f"{entry['positives_acted']}/{entry['positives']}",
                fmt(
                    entry["positives_acted"] / entry["positives"]
                    if entry["positives"]
                    else None
                ),
                entry["note"],
            ]
        )
    lines.append(
        md_table(
            [
                "arm",
                "tau",
                "acted",
                "coverage",
                "TP (scams flagged)",
                "FP (legitimate flagged)",
                "precision",
                "recall",
                "FN (scams missed)",
                "positives acted/total",
                "positives acted rate",
                "note",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "`FN (scams missed)` counts positives below the threshold: the frauds the "
        "acted-on set does not catch. `positives acted/total` is the share of the "
        "positives that survive the filter, given as rate and count. At a threshold above "
        "every score nothing is acted on: coverage 0, precision and recall `n/a` with the "
        "reason stated, and no division by zero."
    )
    lines.append("")
    lines.append(
        "classifier.dev is a shared service and its probabilities shift slightly between "
        "fetches of the same posting, so the threshold-dependent cells above can move by "
        "one record between reruns while the label-based tables in section 3 do not. Treat "
        "the routing cells as indicative at the high thresholds, where the `acted` column "
        "shows how few postings carry the decision."
    )
    lines.append("")
    return "\n".join(lines)


def _section_cost(analysis: Analysis) -> str:
    lines = ["## 6. Cost and latency", ""]
    rows: list[list[Any]] = []
    projection: list[str] = []
    for arm in ordered_arms(analysis.loaded.arms):
        if arm in FLOORS:
            continue
        stats = latency_stats(analysis.loaded.arms[arm])
        rows.append(
            [
                arm,
                stats["n_rows"],
                stats["n_latency"],
                fmt(stats["p50_ms"], 1),
                fmt(stats["p95_ms"], 1),
                fmt(stats["mean_ms"], 1),
                fmt(stats["tokens_in"], 1) if stats["tokens_in"] is not None else "n/a",
                fmt(stats["tokens_out"], 1)
                if stats["tokens_out"] is not None
                else "n/a",
                stats["n_tokens"] or "n/a",
            ]
        )
        if stats["posting_ms"] is None:
            projection.append(f"* `{arm}`: n/a: no `latency_ms` on any row.")
            continue
        seconds = stats["posting_ms"] / 1000.0
        projection.append(
            f"* `{arm}`: {stats['n_records']} distinct records at "
            f"{stats['posting_ms']:.1f} ms per posting -> "
            f"10,000 x {stats['posting_ms']:.1f} ms = {10_000 * seconds / 60:.1f} min "
            f"({10_000 * seconds / 3600:.2f} h) serial"
            + (
                f", {10_000 * stats['tokens_out']:,.0f} completion tokens at "
                f"{stats['tokens_out']:.1f} tokens per row"
                if stats["tokens_out"] is not None
                else ""
            )
            + "."
        )
    lines.append(
        md_table(
            [
                "arm",
                "rows",
                "rows with latency",
                "p50 latency_ms",
                "p95 latency_ms",
                "mean latency_ms",
                "mean tokens_in",
                "mean tokens_out",
                "rows with token counts",
            ],
            rows or [["(no arms)", 0, 0, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]],
        )
    )
    lines.append("")
    lines.append(
        "Latency is per result row, as the runner recorded it. `llm_local`/`llm_hosted` "
        "answer all four questions in one completion and the runner repeats that "
        "completion's timing on each of the four rows, so when projecting to postings the "
        "identical per-record values are counted once and one posting is one completion."
    )
    lines.append("")
    lines.append(f"Cost note: {analysis.cost_note}")
    lines.append("")
    lines.append("### Projection to 10,000 postings/day")
    lines.append("")
    lines += projection or ["* n/a: no arm produced rows."]
    classifier_rows = analysis.loaded.arms.get("classifier_dev", [])
    if classifier_rows:
        records = {str(r.get("record_id")) for r in classifier_rows}
        questions = sorted({str(r.get("question")) for r in classifier_rows})
        per_question = Counter(str(r.get("question")) for r in classifier_rows)
        lines.append(
            f"* `classifier_dev`: {len(classifier_rows)} rows = {len(classifier_rows)} "
            "classifications spent, one question-answer each ("
            + ", ".join(f"{q}={per_question[q]}" for q in questions)
            + f") over {len(records)} distinct record(s). Answers are cached in "
            "`results/cache.jsonl` keyed by "
            "`sha256(question, labels, instructions, tier, state)`, so reruns and `--limit` "
            "top-ups spend no additional quota. classifier.dev's free fast tier allows "
            "20,000 classifications/day per IP, so this run used "
            f"{len(classifier_rows) / 20_000:.1%} of one day's quota; the number of "
            "records walked is set by the runner's `--limit`, and the cost of a window is "
            "the sum of its per-question rows."
        )
    else:
        lines.append("* `classifier_dev`: n/a, no rows for this arm in this directory.")
    lines.append("")
    return "\n".join(lines)


def _section_ablation(analysis: Analysis) -> str:
    lines = ["## 7. Instructions ablation", ""]
    plain, no_rubric = "classifier_dev", "classifier_dev_no_rubric"
    if not analysis.loaded.arms.get(no_rubric):
        lines.append(
            f"n/a: no `{no_rubric}` rows in this results directory. Produce them with "
            f"`uv run python -m triage.run --arms {plain} --questions fraudulent "
            "--no-instructions` and re-run the evaluator; the ablation is reported whatever "
            "its direction."
        )
        return "\n".join(lines)
    lines.append(
        f"Both variants exist. `{plain}` sends the `instructions` rubric field to "
        f"classifier.dev; `{no_rubric}` drops it. Accuracy and F1 are shown side by side "
        "for the questions they cover, irrespective of which way the difference goes."
    )
    lines.append("")
    rows: list[list[Any]] = []
    for question in analysis.questions:
        for arm in (plain, no_rubric):
            m = analysis.metrics.get(question.id, {}).get(arm)
            if m is None:
                continue
            rows.append(
                [
                    question.id,
                    arm,
                    m.n,
                    fmt(m.accuracy),
                    fmt(m.macro_f1),
                    fmt(m.f1_positive) if question.positive else "n/a",
                    fmt(m.brier),
                ]
            )
    if not rows:
        lines.append("n/a: neither variant produced scorable rows.")
    else:
        lines.append(
            md_table(
                ["question", "arm", "n", "accuracy", "macro-F1", "F1(positive)", "Brier"],
                rows,
            )
        )
        lines.append("")
        for question in analysis.questions:
            a = analysis.metrics.get(question.id, {}).get(plain)
            b = analysis.metrics.get(question.id, {}).get(no_rubric)
            if a is None or b is None or a.accuracy is None or b.accuracy is None:
                continue
            lines.append(
                f"* `{question.id}`: dropping the rubric changes accuracy by "
                f"{(b.accuracy - a.accuracy) * 100:+.1f}pp "
                f"({fmt(a.accuracy)} -> {fmt(b.accuracy)})."
            )
    lines.append("")
    return "\n".join(lines)


def _section_na(analysis: Analysis) -> str:
    lines = ["## 8. Not available (n/a)", ""]
    lines.append(
        "Every metric that could not be computed and every scoring caveat, with its "
        "reason. Nothing here is a substituted model, dataset or metric. The same list is "
        "written machine-readably as `n_a` in `metrics.json`."
    )
    lines.append("")
    if not analysis.na:
        lines.append("* Nothing was n/a in this run.")
        lines.append("")
        return "\n".join(lines)
    for entry in sorted(
        analysis.na,
        key=lambda e: (e["question"] or "", e["arm"] or "", e["metric"]),
    ):
        scope = " / ".join(
            part
            for part in (
                f"`{entry['arm']}`" if entry["arm"] else "",
                f"`{entry['question']}`" if entry["question"] else "",
            )
            if part
        )
        lines.append(
            f"* {scope + ': ' if scope else ''}`{entry['metric']}`: {entry['reason']}"
        )
    lines.append("")
    return "\n".join(lines)


def _section_guard(analysis: Analysis) -> str:
    lines = ["## 9. Leakage guard (`salary_range_stated`)", ""]
    lines.append(
        "`salary_range` is platform metadata, not posting text, so an arm that sees only "
        "`title + description` cannot legitimately beat the majority rate. The guard is "
        "therefore deliberately one-directional:"
    )
    lines.append("")
    lines.append(
        f"* **PASS** when `accuracy <= floor + {LEAK_TOLERANCE_PP:g}pp`: no evidence that "
        "the gold reached the model's input."
    )
    lines.append(
        f"* **FAIL** when any arm is ABOVE `floor + {LEAK_TOLERANCE_PP:g}pp`: that is the "
        "only direction leakage can push accuracy, so the process prints FAIL and exits "
        "non-zero."
    )
    lines.append(
        f"* **BELOW** (`accuracy < floor - {LEAK_TOLERANCE_PP:g}pp`) does not fail: it is "
        "below the floor, not leakage. That arm is not answering the question that was "
        "asked (see the no-rubric ablation) or is not reading the text, and it is "
        "reported with its own note."
    )
    lines.append("")
    if not analysis.guard:
        lines.append(
            "n/a: no arm produced `salary_range_stated` rows in this results directory, so "
            "the guard has nothing to check and does not report PASS by default."
        )
        return "\n".join(lines)
    floors = {
        (entry["majority_label"], round(entry["majority"], 6)): entry
        for entry in analysis.guard
    }
    lines.append(
        "Floor compared against: "
        + "; ".join(
            f"{rate:.3f} (majority label '{label}', "
            f"{entry['majority_count']}/{entry['n']} of the records present in these "
            "results)"
            for (label, rate), entry in sorted(floors.items())
        )
        + ". It is computed from the gold of the records actually present in the results, "
        "not from the full split, and 'above' is the only direction that can indicate "
        "leakage."
    )
    lines.append("")
    rows = [
        [
            entry["arm"],
            entry["n"],
            fmt(entry["accuracy"]),
            fmt(entry["majority"]),
            fmt(entry["delta_pp"], 1),
            entry["verdict"],
            entry["note"],
        ]
        for entry in analysis.guard
    ]
    lines.append(
        md_table(
            ["arm", "n", "accuracy", "floor (majority rate)", "delta (pp)", "verdict",
             "note"],
            rows,
        )
    )
    lines.append("")
    skipped = [arm for arm in analysis.arms_for("salary_range_stated") if arm in FLOORS]
    if skipped:
        lines.append(
            "Floors are excluded from the guard: "
            + ", ".join(f"`{a}`" for a in skipped)
            + " are built from the gold itself, so the check would be vacuous on them (the "
            "stratified floor also samples labels without looking at the text, so its "
            "accuracy sits below the majority rate by construction)."
        )
        lines.append("")
    failing = [e for e in analysis.guard if e["verdict"] == "FAIL"]
    lines.append(
        "Verdict: **"
        + ("FAIL" if failing else "PASS")
        + "** ("
        + ", ".join(f"{entry['arm']}={entry['verdict']}" for entry in analysis.guard)
        + ")."
    )
    return "\n".join(lines)


def build_report(analysis: Analysis) -> str:
    header = [
        "# job-posting-triage: evaluation report",
        "",
        f"Results directory: `{analysis.results_dir}`. Arms found: "
        f"{', '.join(f'`{a}`' for a in analysis.loaded.arm_names) or '(none)'}. "
        "Arms in this report: "
        f"{', '.join(f'`{a}`' for a in ordered_arms(analysis.loaded.arms)) or '(none)'}.",
        "",
        "Alongside this file, `--report` writes `metrics.json`: the same numbers, from "
        "the same code path, for machine consumption (`n_a` there is the list in "
        "section 8).",
        "",
    ]
    sections = [
        _section_run(analysis),
        "## 2. Metric definitions",
        "",
        METRIC_DEFS,
        _section_questions(analysis),
        _section_calibration(analysis),
        _section_routing(analysis),
        _section_cost(analysis),
        _section_ablation(analysis),
        _section_na(analysis),
        _section_guard(analysis),
    ]
    return "\n".join(header + sections) + "\n"


# --------------------------------------------------------------------------
# metrics.json
# --------------------------------------------------------------------------

#: Used wherever a precision has no denominator: the arm predicted nothing with
#: that label, so the ratio does not exist. Both the metrics table and the
#: per-class table say this, with the same words.
PRECISION_UNDEFINED = (
    "no row of this arm was predicted with this label, so precision has no "
    "denominator and is undefined"
)

#: Fallback reasons for a null in metrics.json when no specific reason was
#: registered during the analysis. Every null in the payload gets one of these
#: or a specific one from ``analysis.na``, so nothing is ever silently missing.
GENERIC_NA: dict[str, str] = {
    "accuracy": "no scored rows, so accuracy cannot be computed",
    "macro_f1": "no scored rows, so macro-F1 cannot be computed",
    "per_class": "no scored rows, so no per-class metrics exist",
    "pr_auc": "no row of this arm carries p_positive for this question",
    "f1_positive": "no scored rows, so the positive-class F1 cannot be computed",
    "precision_ci95": "no positive predictions in the scored subset, so the interval has "
    "no denominator",
    "recall_ci95": "no positives in the scored subset, so the interval has no denominator",
    "f1_ci95": "no scored rows, so the bootstrap interval has nothing to resample",
    "ci_note": "not a binary question: the positive-class intervals are only defined for "
    "`fraudulent`, and the multi-label caveat is stated in report section 3",
    "precision_positive": PRECISION_UNDEFINED,
    "per_class_precision": PRECISION_UNDEFINED,
    "recall_positive": "the scored subset contains no positive gold",
    "mae": "no row has both an ordinal gold and an ordinal prediction",
    "within_one": "no row has both an ordinal gold and an ordinal prediction",
    "brier": "no row of this arm carries a usable p_decision for this question",
    "ece": "no row of this arm carries a usable p_decision for this question",
    "latency_p50": "no row of this arm carries latency_ms for this question",
    "latency_p95": "no row of this arm carries latency_ms for this question",
    "tokens_in_mean": "no row of this arm carries a prompt token count",
    "tokens_out_mean": "no row of this arm carries a completion token count",
    "invalid_reasons": "no row of this arm is marked invalid for this question",
    "coverage": "no scored rows for this question",
    "precision": "nothing reaches this threshold, so precision is undefined",
    "recall": "nothing reaches this threshold, or there are no positives to catch",
    "classifications": "classifications are a classifier.dev unit: this arm's rows are "
    "not classifier.dev classifications",
    "requests": "the runner records result rows, not HTTP requests, so the request "
    "count was never measured",
    "seconds": "the runner recorded no wall clock time for this arm "
    "(no entry in _unavailable.json)",
    "split": "the rows carry no split field",
    "window": "no frame was recorded in results/_frame.json and the record ids do not "
    "allow a window to be derived (they are not all of the form '<split>-<index>')",
    "ablation": "no classifier_dev_no_rubric rows, so there is no ablation pair",
}


def _display_path(path: Path) -> str:
    """A short path for the report: relative when it is inside the cwd."""
    try:
        relative = path.relative_to(Path.cwd())
        return str(relative)
    except ValueError:
        return str(path)


def _recorded_frame_note(recorded: dict, display: str) -> str:
    """The frame the runner recorded, said back in one line."""
    split = recorded.get("split")
    limit = recorded.get("limit")
    seed = recorded.get("seed")
    sampling = recorded.get("sampling") or "sampling method not stated"
    ids = recorded.get("record_ids")
    questions = recorded.get("questions")
    if limit is None:
        head = (
            f"{sampling} of split {split}"
            if "whole" in str(sampling).lower()
            else f"{sampling} over the whole split {split}"
        )
    else:
        head = f"{sampling} of {limit} records from split {split}"
    if seed is not None:
        head += f" with seed {seed}"
    counts = []
    if isinstance(ids, list):
        counts.append(f"{len(ids)} record ids")
    elif recorded.get("records_sent") is not None:
        counts.append(f"{recorded['records_sent']} records sent")
    if isinstance(questions, list):
        counts.append(f"{len(questions)} question{'s' if len(questions) != 1 else ''}")
    return f"the runner recorded this frame in {display}: {head}" + (
        ", " + ", ".join(counts) if counts else ""
    )


def _window_from_records(record_ids: Iterable[str], split: str | None) -> tuple[int | None, str]:
    """``(window, note)``: the window derived from record ids, never assumed.

    The highest index seen plus one is reported, but a set of record ids cannot
    tell a random sample from a prefix and cannot reconstruct the draw, so the
    note says so; ``results/_frame.json`` is the authoritative record when the
    runner wrote one.
    """
    if not split:
        return None, "no split on the rows, so no window can be derived"
    pattern = f"{split}-"
    indexes = []
    for record_id in record_ids:
        if not record_id.startswith(pattern):
            return None, (
                f"record ids are not all of the form '{split}-<index>', so the window "
                "cannot be derived from them"
            )
        tail = record_id[len(pattern):]
        if not tail.isdigit():
            return None, (
                f"record ids are not all of the form '{split}-<index>', so the window "
                "cannot be derived from them"
            )
        indexes.append(int(tail))
    if not indexes:
        return None, "no records in this results directory"
    return max(indexes) + 1, (
        f"derived as the highest record index seen plus one over the {len(indexes)} "
        "record ids in the results; no frame was recorded, so this cannot distinguish a "
        "random sample from a prefix and the draw cannot be reconstructed"
    )


#: Verbatim selection rules for the showcase hero record.
HERO_RULE_DISAGREEMENT = (
    "the lowest record_id among records with gold fraudulent where at least two arms "
    "disagree on the fraudulent label"
)
HERO_RULE_FALLBACK = (
    "the lowest record_id with gold fraudulent: no record with gold fraudulent has two "
    "arms disagreeing on the fraudulent label"
)
HERO_STATE_CHARS = 400


def _record_order(record_id: str) -> tuple:
    """Order ids so that '<prefix>-<index>' sorts by index, everything else by text.

    The runner walks a split in record order, so index order is the order the arms
    saw the records in; ids that are not '<prefix>-<index>' fall back to text
    order. Either way the choice is deterministic.
    """
    head, _, tail = record_id.rpartition("-")
    if head and tail.isdigit():
        return (0, head, int(tail), "")
    return (1, record_id, 0, record_id)


def _state_excerpt(split: str, record_id: str, limit: int = HERO_STATE_CHARS) -> str | None:
    """The first ``limit`` characters of the state the arms saw, from triage.data."""
    if split not in data.SPLITS:
        return None
    try:
        records = data.records(split)
    except Exception:  # noqa: BLE001 - reported as null plus an n_a entry
        return None
    for record in records:
        if record["record_id"] == record_id:
            return record["state"][:limit]
    return None


def _hero(analysis: Analysis) -> tuple[dict | None, str | None]:
    """``(hero, null_reason)``: the deterministic showcase record, or ``None``.

    Built from the same scored rows the tables use, so it cannot disagree with
    them. Floors are not engines: they never produced a label from a posting.
    """
    rows_by_record: dict[str, list[Scored]] = defaultdict(list)
    for arm, aq in analysis.scored.get("fraudulent", {}).items():
        if arm in FLOORS:
            continue
        for scored in aq.scored:
            if scored.gold == "fraudulent":
                rows_by_record[scored.record_id].append(scored)
    if not rows_by_record:
        return None, None

    def disagrees(record_id: str) -> bool:
        values = {s.value for s in rows_by_record[record_id] if s.value}
        return len(values) >= 2

    candidates = sorted(rows_by_record, key=_record_order)
    chosen = next((rid for rid in candidates if disagrees(rid)), None)
    if chosen is None:
        chosen = candidates[0]
        rule = HERO_RULE_FALLBACK
    else:
        rule = HERO_RULE_DISAGREEMENT

    engines = []
    for scored in sorted(rows_by_record[chosen], key=lambda s: s.arm):
        engines.append(
            {
                "arm": scored.arm,
                "value": scored.value,
                "correct": scored.correct,
                "confidence": _p_float(scored.row.get("confidence")),
                "p_positive": scored.p_positive,
                "p_decision": scored.p_decision,
            }
        )
    split = rows_by_record[chosen][0].split
    excerpt = _state_excerpt(split, chosen)
    hero = {
        "selection_rule": rule,
        "record_order": "record_id ordered by numeric index when ids follow "
        "'<prefix>-<index>', otherwise by text; this matches the order the runner "
        "walked the split in",
        "record_id": chosen,
        "split": split,
        "gold": "fraudulent",
        "state_excerpt": excerpt,
        "state_excerpt_chars": HERO_STATE_CHARS if excerpt is not None else None,
        "state_chars": len(excerpt) if excerpt is not None else None,
        "state_source": "triage.data.records(split), truncated to "
        f"{HERO_STATE_CHARS} characters",
        "engines": engines,
        "engines_excluded": list(FLOORS),
        "disagreeing_engine_values": sorted(
            {s.value for s in rows_by_record[chosen] if s.value}
        ),
    }
    return hero, (
        None
        if excerpt is not None
        else f"the state of record {chosen!r} is not in dataset split {split!r}, so no "
        "state excerpt can be shown (the row's own data was still scored)"
    )


def build_metrics_json(analysis: Analysis) -> dict:
    """The machine-readable dump: exactly the numbers report.md shows.

    Built from the same ``Analysis`` object and the same helpers the report uses
    (``metrics_for``, ``calibration``, ``routing_table``, ``latency_stats``,
    ``leakage_guard``), so there is no second implementation of any metric. Every
    key that could not be computed is ``null`` and carries an entry in ``n_a``.
    """
    loaded = analysis.loaded
    arms = ordered_arms(loaded.arms)
    per_arm_records = {
        arm: {str(row.get("record_id")) for row in rows}
        for arm, rows in loaded.arms.items()
    }
    all_records = set().union(*per_arm_records.values()) if per_arm_records else set()
    splits = Counter(
        str(row.get("split")) for rows in loaded.arms.values() for row in rows
    )
    split = splits.most_common(1)[0][0] if splits else None
    display = _display_path(analysis.results_dir)
    recorded = loaded.frame_recorded
    if recorded is not None:
        window = recorded.get("limit")
        window_note = _recorded_frame_note(recorded, f"{display}/_frame.json")
        frame_source = f"{display}/_frame.json"
    else:
        window, window_note = _window_from_records(all_records, split)
        frame_source = "derived from record indices"

    na: list[dict] = [dict(entry) for entry in analysis.na]
    seen = {(e["arm"], e["question"], e["metric"]) for e in na}
    specific = {
        (entry["arm"], entry["question"], entry["metric"]): entry["reason"]
        for entry in analysis.na
    }
    floor_reasons = {
        (entry["question"], entry["metric"]): entry["reason"]
        for entry in analysis.na
        if entry["metric"] in FLOORS
    }

    def note(arm: str | None, question: str | None, metric: str) -> None:
        key = (arm, question, metric)
        if key in seen:
            return
        seen.add(key)
        reason = (
            specific.get(key)
            or specific.get((arm, None, metric))
            or specific.get((None, question, metric))
            or (floor_reasons.get((question, arm)) if arm in FLOORS else None)
            or GENERIC_NA.get(metric)
            or f"not computed for arm {arm!r} on question {question!r}"
        )
        na.append({"arm": arm, "question": question, "metric": metric, "reason": reason})

    def payload(arm: str, question_id: str) -> dict | None:
        question = by_id(question_id)
        m = analysis.metrics.get(question_id, {}).get(arm)
        if m is None:
            return None
        rows = analysis.scored[question_id][arm].scored
        stats = latency_stats([s.row for s in rows])
        out: dict[str, Any] = {
            "n": m.n,
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "macro_f1_labels": list(question.labels),
            "per_class": {
                entry["label"]: {
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "f1": entry["f1"],
                    "support": entry["support"],
                }
                for entry in m.classes
            }
            or None,
            "per_class_precision_undefined": [
                entry["label"] for entry in m.classes if entry["precision"] is None
            ]
            or None,
            "pr_auc": m.pr_auc,
        }
        if question.positive is not None:
            # Intervals are defined for the binary question only: a multi-label
            # argmax is not a single binomial proportion, so report section 3
            # states that caveat instead of inventing an interval.
            out.update(
                {
                    "precision_ci95": list(m.precision_ci95) if m.precision_ci95 else None,
                    "recall_ci95": list(m.recall_ci95) if m.recall_ci95 else None,
                    "f1_ci95": list(m.f1_ci95) if m.f1_ci95 else None,
                    "ci_note": m.ci_note,
                }
            )
        out.update(
            {
            "f1_positive": m.f1_positive,
            "precision_positive": m.precision_positive,
            "recall_positive": m.recall_positive,
            "positive_label": question.positive,
            "positives": m.positives,
            "confusion": {
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "tn": m.tn,
                "no_call": m.no_call,
                "no_call_invalid": m.no_call_invalid,
                "no_call_positives": m.no_call_positives,
                "sum": m.tp + m.fp + m.fn + m.tn + m.no_call,
            },
            "mae": m.mae,
            "within_one": m.plus_minus_1,
            "ordinal_n": m.ordinal_n,
            "ordinal_excluded": m.ordinal_excluded,
            "invalid_json_rate": m.invalid_rate,
            "invalid_reasons": dict(
                sorted(m.invalid_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            or None,
            "invalid_rows": sum(m.invalid_reasons.values()),
            "brier": m.brier,
            "ece": m.ece,
            "calibration_n": m.cal_n,
            "calibration_excluded": m.cal_excluded,
            "calibration_excluded_invalid": m.excluded_invalid,
            "calibration_excluded_valid": m.excluded_valid,
            "latency_p50": stats["p50_ms"],
            "latency_p95": stats["p95_ms"],
            "latency_n": stats["n_latency"],
            "tokens_in_mean": stats["tokens_in"],
            "tokens_out_mean": stats["tokens_out"],
            "rows": stats["n_rows"],
            }
        )
        for entry in m.classes:
            if entry["precision"] is None:
                na.append(
                    {
                        "arm": arm,
                        "question": question_id,
                        "metric": f"per_class_precision[{entry['label']}]",
                        "reason": PRECISION_UNDEFINED,
                    }
                )
        for key, value in out.items():
            if value is None:
                note(arm, question_id, key)
        return out

    per_question: dict[str, Any] = {}
    for question in analysis.questions:
        metrics = analysis.metrics.get(question.id)
        if not metrics:
            per_question[question.id] = {
                "labels": list(question.labels),
                "kind": question.kind,
                "gold_distribution": None,
                "majority_label": None,
                "rows": {},
                "baselines": {},
                "scored_records": 0,
                "subset_arm": None,
            }
            for key in ("gold_distribution", "majority_label", "subset_arm"):
                note(None, question.id, key)
            continue
        reference = analysis.subset_reference.get(question.id)
        reference_rows = (
            analysis.scored[question.id][reference].scored
            if reference in analysis.scored.get(question.id, {})
            else []
        )
        counts = Counter(s.gold for s in reference_rows)
        majority = counts.most_common(1)[0][0] if counts else None
        entry: dict[str, Any] = {
            "labels": list(question.labels),
            "kind": question.kind,
            "gold_distribution": dict(counts) if counts else None,
            "majority_label": majority,
            "scored_records": len(reference_rows),
            "subset_arm": reference,
            "rows": {},
            "baselines": {},
        }
        for arm in metrics:
            target = "baselines" if arm in FLOORS else "rows"
            body = payload(arm, question.id)
            if body is not None:
                entry[target][arm] = body
        for key, value in entry.items():
            if value is None:
                note(None, question.id, key)
        per_question[question.id] = entry

    routing: dict[str, list[dict]] = {}
    fraudulent = analysis.metrics.get("fraudulent")
    if fraudulent:
        for row in routing_table(
            by_id("fraudulent"), fraudulent, analysis.scored["fraudulent"]
        ):
            body = {
                "tau": row["tau"],
                "coverage": row["coverage"],
                "precision": row["precision"],
                "recall": row["recall"],
                "false_negatives": row["fn_missed"],
                "positives_acted_on": row["positives_acted"],
                "positives_total": row["positives"],
                "acted": row["acted"],
                "scored": row["n"],
                "records_without_p_positive": row["no_score"],
                "note": row["note"],
            }
            routing.setdefault(row["arm"], []).append(body)
            for key in ("coverage", "precision", "recall"):
                if body[key] is None:
                    note(row["arm"], "fraudulent", key)

    calibration_bins: dict[str, dict[str, list[dict]]] = {}
    for question in analysis.questions:
        for arm, m in analysis.metrics.get(question.id, {}).items():
            bins = []
            for index, entry in enumerate(m.cal_bins):
                if entry["n"] is None:
                    continue
                bins.append(
                    {
                        "bin": [entry["low"], entry["high"]],
                        "n": entry["n"],
                        "mean_p": entry["conf"],
                        "accuracy": entry["acc"],
                    }
                )
                if not entry["n"]:
                    note(arm, question.id, "calibration_bin")
            calibration_bins.setdefault(arm, {})[question.id] = bins

    guard = {
        entry["arm"]: {
            "question": "salary_range_stated",
            "n": entry["n"],
            "accuracy": entry["accuracy"],
            "floor": entry["majority"],
            "delta_pp": entry["delta_pp"],
            "passed": entry["verdict"] != "FAIL",
            "verdict": entry["verdict"],
            "note": entry["note"],
        }
        for entry in analysis.guard
    }

    cost: dict[str, dict] = {}
    for arm in arms:
        status = loaded.status.get(arm) or {}
        is_classifier = arm.startswith("classifier_dev")
        body = {
            "classifications": len(loaded.arms[arm]) if is_classifier else None,
            "requests": None,
            "seconds": status.get("seconds"),
            "rows": len(loaded.arms[arm]),
            "records": len(per_arm_records[arm]),
        }
        for key in ("classifications", "requests", "seconds"):
            if body[key] is None:
                note(arm, None, key)
        cost[arm] = body

    ablation: dict[str, Any] | None = None
    if loaded.arms.get("classifier_dev_no_rubric"):
        ablation = {}
        for question in analysis.questions:
            pair = {
                "classifier_dev": payload("classifier_dev", question.id),
                "classifier_dev_no_rubric": payload(
                    "classifier_dev_no_rubric", question.id
                ),
            }
            if pair["classifier_dev"] is None and pair["classifier_dev_no_rubric"] is None:
                continue
            for arm, body in pair.items():
                if body is None:
                    note(arm, question.id, "ablation")
            ablation[question.id] = pair
        if not ablation:
            ablation = None
            note(None, None, "ablation")

    unavailable = {}
    for arm, status in sorted(loaded.status.items()):
        if status.get("status") in (None, "ok") and arm in loaded.arms:
            continue
        unavailable[arm] = {
            "status": status.get("status"),
            "reason": status.get("reason"),
        }

    if split is None:
        note(None, None, "split")
    if window is None:
        reason = (
            "the recorded frame covers the whole split with no limit, so there is no "
            "window"
            if recorded is not None
            else GENERIC_NA["window"]
        )
        na.append({"arm": None, "question": None, "metric": "window", "reason": reason})
        seen.add((None, None, "window"))

    recorded_ids = recorded.get("record_ids") if recorded else None
    if isinstance(recorded_ids, list) and recorded_ids:
        recorded_set = {str(record_id) for record_id in recorded_ids}
        unseen = sorted(all_records - recorded_set)
        if unseen:
            na.append(
                {
                    "arm": None,
                    "question": None,
                    "metric": "frame_recorded",
                    "reason": f"{len(unseen)} record id(s) in the results are not in the "
                    f"frame recorded in {display}/_frame.json (for example "
                    f"{', '.join(unseen[:3])}); the rows and the recorded frame disagree",
                }
            )
            seen.add((None, None, "frame_recorded"))

    hero, hero_reason = _hero(analysis)
    if hero_reason:
        na.append(
            {
                "arm": None,
                "question": "fraudulent",
                "metric": "hero.state_excerpt",
                "reason": hero_reason,
            }
        )
        seen.add((None, "fraudulent", "hero.state_excerpt"))

    payload: dict[str, Any] = {
        "split": split,
        "window": window,
        "frame_source": frame_source,
        "frame": analysis.frame,
        "window_note": window_note,
        "records_sent": len(all_records),
        "records_per_arm": {arm: len(ids) for arm, ids in per_arm_records.items()},
        "row_count": len(loaded.rows),
        "arm_count": len(arms),
        "arms": arms,
        "unavailable": unavailable,
        "per_question": per_question,
        "routing": routing,
        "leakage_guard": guard,
        "calibration_bins": calibration_bins,
        "cost": cost,
        "cost_note": analysis.cost_note,
        "cost_note_default": analysis.cost_note == DEFAULT_COST_NOTE,
        "ablation": ablation,
        "n_a": sorted(
            na,
            key=lambda e: (
                e["arm"] or "",
                e["question"] or "",
                e["metric"],
                e["reason"],
            ),
        ),
        "results_dir": str(analysis.results_dir),
        "files_read": {name: count for name, count in loaded.files},
        "gold_source": {
            "from_dataset": analysis.goldbook.from_dataset,
            "from_row_field": analysis.goldbook.from_row,
            "dataset_errors": dict(analysis.goldbook.errors),
            "mismatches": {
                f"{question} ({split_name})": count
                for (question, split_name), count in sorted(
                    analysis.goldbook.mismatch.items()
                )
            },
        },
    }
    if hero is not None:
        payload["hero"] = hero
    return payload


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def write_plots(analysis: Analysis, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    written: list[Path] = []
    arms = [
        a for a in ordered_arms(analysis.loaded.arms) if a not in FLOORS
    ]
    questions = [q.id for q in analysis.questions if analysis.metrics.get(q.id)]
    x = np.arange(len(questions)) if questions else np.arange(1)
    width = 0.8 / max(len(arms), 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for panel, metric, title in (
        (axes[0][0], "brier", "Brier"),
        (axes[0][1], "ece", "ECE"),
    ):
        for index, arm in enumerate(arms):
            heights = []
            for question_id in questions:
                m = analysis.metrics.get(question_id, {}).get(arm)
                value = getattr(m, metric) if m is not None else None
                heights.append(np.nan if value is None else value)
            panel.bar(x + index * width, heights, width, label=arm)
        panel.set_title(f"{title} by question and arm (lower is better)")
        panel.set_xticks(x + width * max(len(arms) - 1, 0) / 2)
        panel.set_xticklabels(questions, rotation=20, ha="right")
        if arms:
            panel.legend(fontsize=7)
    for panel, question_id in (
        (axes[1][0], "fraudulent"),
        (axes[1][1], "salary_range_stated"),
    ):
        panel.set_title(f"Reliability curve, {question_id}")
        panel.set_xlabel("p_decision")
        panel.set_ylabel("empirical accuracy")
        panel.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        panel.grid(alpha=0.3)
        plotted = False
        for arm in ordered_arms(analysis.metrics.get(question_id, {})):
            m = analysis.metrics[question_id][arm]
            bins = [entry for entry in m.cal_bins if entry["n"]]
            if not bins:
                continue
            panel.plot(
                [entry["conf"] for entry in bins],
                [entry["acc"] for entry in bins],
                marker="o",
                linewidth=1,
                label=f"{arm} (n={m.cal_n})",
            )
            plotted = True
        panel.legend(fontsize=7)
        if not plotted:
            panel.text(0.5, 0.5, "n/a: no p_decision", ha="center", va="center")
    fig.suptitle("job-posting-triage calibration", fontsize=13)
    fig.tight_layout()
    calibration_png = out_dir / "calibration.png"
    fig.savefig(calibration_png, dpi=130)
    plt.close(fig)
    written.append(calibration_png)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = analysis.metrics.get("fraudulent")
    if metrics:
        table = routing_table(
            by_id("fraudulent"), metrics, analysis.scored["fraudulent"]
        )
        by_arm: dict[str, list[dict]] = defaultdict(list)
        for entry in table:
            by_arm[entry["arm"]].append(entry)
        for arm, entries in by_arm.items():
            if arm in FLOORS:
                continue
            points = sorted(
                (
                    (entry["coverage"], entry["precision"], entry["recall"])
                    for entry in entries
                    if entry["coverage"] is not None
                )
            )
            if not points:
                continue
            for panel, position in ((axes[0], 1), (axes[1], 2)):
                panel.plot(
                    [p[0] for p in points],
                    [np.nan if p[position] is None else p[position] for p in points],
                    marker="o",
                    label=arm,
                )
        for panel, title in (
            (axes[0], "precision of the auto-flagged set"),
            (axes[1], "recall of scams (positives caught)"),
        ):
            panel.set_xlabel("coverage (share of records auto-flagged)")
            panel.set_title(
                f"{title}\npoints are tau = " + ", ".join(f"{t:g}" for t in TAUS)
            )
            panel.set_ylim(-0.05, 1.05)
            panel.set_xlim(-0.05, 1.05)
            panel.grid(alpha=0.3)
            panel.legend(fontsize=8)
        axes[1].set_ylabel("recall")
    else:
        for panel in axes:
            panel.text(0.5, 0.5, "n/a: no fraudulent rows", ha="center", va="center")
    fig.suptitle("job-posting-triage routing, fraudulent", fontsize=13)
    fig.tight_layout()
    routing_png = out_dir / "routing.png"
    fig.savefig(routing_png, dpi=130)
    plt.close(fig)
    written.append(routing_png)
    return written


# --------------------------------------------------------------------------
# Failure dump
# --------------------------------------------------------------------------


def failure_blocks(
    analysis: Analysis, arm: str, question_id: str, limit: int = 0
) -> int:
    """Mispredictions, then the positives left unacted-on at tau=0.9."""
    question = by_id(question_id)
    scored = analysis.scored.get(question_id, {}).get(arm)
    print(f"# failures: arm={arm} question={question_id}")
    print(
        "# one block per mispredicted record (gold != prediction). The posting text is "
        "read from the dataset with triage.data and truncated; it is printed to stdout "
        "only, never written into results/."
    )
    if scored is None:
        print(
            f"no scorable rows for arm={arm!r} question={question_id!r} in "
            f"{analysis.results_dir} (arms present: "
            f"{', '.join(ordered_arms(analysis.loaded.arms)) or 'none'})"
        )
        return 1
    wrong = [s for s in scored.scored if not s.correct]
    if limit:
        wrong = wrong[:limit]
    print(f"# {len(wrong)} mispredicted of {scored.n} scored rows")
    states = _states_for({s.split for s in wrong})
    for s in wrong:
        print("")
        print(f"record_id   {s.record_id}  (split {s.split})")
        print(f"gold        {s.gold}")
        print(f"prediction  {s.value or '<empty>'}")
        print(f"p_positive  {fmt(s.p_positive)}")
        print(f"p_decision  {fmt(s.p_decision)}")
        print(f"invalid     {s.invalid}")
        state = states.get((s.split, s.record_id))
        if state:
            print(f"state       first {FAILURE_STATE_CHARS} chars, from the dataset")
            print("  " + " ".join(state.split())[:FAILURE_STATE_CHARS])
        else:
            print("state       n/a: record not in the dataset split")

    if question.positive is not None:
        positives = [s for s in scored.scored if s.gold == question.positive]
        missed = [
            s for s in positives if s.p_positive is None or s.p_positive < 0.9
        ]
        print("")
        print(
            "# positives not acted on at tau=0.9 (the false negatives of the auto-flag "
            f"set): {len(missed)} of {len(positives)}"
        )
        for s in missed[: limit or None]:
            print(
                f"  {s.record_id}  gold={s.gold} prediction={s.value or '<empty>'} "
                f"p_positive={fmt(s.p_positive)} p_decision={fmt(s.p_decision)}"
            )
    return 0


def _states_for(splits: Iterable[str]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for split in sorted(set(splits)):
        if split not in data.SPLITS:
            continue
        try:
            for record in data.records(split):
                out[(split, record["record_id"])] = record["state"]
        except Exception as exc:  # noqa: BLE001 - printed, never fatal
            print(f"# state n/a for split {split}: {type(exc).__name__}: {exc}")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="triage.evaluate")
    parser.add_argument("--results", default="results", help="directory holding *.jsonl")
    parser.add_argument(
        "--report",
        action="store_true",
        help="write report.md, calibration.png and routing.png into --results",
    )
    parser.add_argument(
        "--failures", action="store_true", help="print mispredicted records (stdout only)"
    )
    parser.add_argument("--arm", default=None, help="with --failures: which arm")
    parser.add_argument("--question", default=None, help="with --failures: which question")
    parser.add_argument(
        "--limit", type=int, default=0, help="with --failures: cap the blocks printed"
    )
    parser.add_argument(
        "--leak-tolerance-pp",
        type=float,
        default=LEAK_TOLERANCE_PP,
        help="leakage guard tolerance in percentage points",
    )
    parser.add_argument(
        "--cost-note",
        default=DEFAULT_COST_NOTE,
        help="caveat printed under the cost table and stored as cost_note in "
        "metrics.json (default: latency is this machine's wall clock, so the "
        "projection is an upper bound, not a benchmark)",
    )
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> int:
    global LEAK_TOLERANCE_PP
    LEAK_TOLERANCE_PP = args.leak_tolerance_pp
    results_dir = Path(args.results)
    if not results_dir.is_dir():
        print(f"FAIL --results {results_dir} is not a directory")
        return 2

    analysis = analyse(results_dir, cost_note=args.cost_note)
    report = build_report(analysis)
    print(report)

    if args.report:
        report_path = results_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        metrics_path = results_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(build_metrics_json(analysis), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        written = write_plots(analysis, results_dir)
        print(
            "wrote: " + ", ".join(str(p) for p in [report_path, metrics_path, *written])
        )

    print("---")
    for warning in frame_warnings(analysis):
        print(warning)
    print(
        f"results: {results_dir}; arms found: "
        + (", ".join(ordered_arms(analysis.loaded.arms)) or "(none)")
    )
    for entry in analysis.guard:
        print(
            f"guard {entry['verdict']:<4} {entry['arm']}: accuracy="
            f"{fmt(entry['accuracy'])} majority={fmt(entry['majority'])} "
            f"delta={fmt(entry['delta_pp'], 1)}pp: {entry['note']}"
        )
    if not analysis.guard:
        print(
            "leakage guard: n/a: no salary_range_stated rows to check, so it is not "
            "assessed (and not reported as PASS)"
        )
        return 0
    if not analysis.guard_ok:
        print(
            "FAIL leakage guard: at least one arm's salary_range_stated accuracy is more "
            f"than {LEAK_TOLERANCE_PP:g}pp ABOVE the majority floor, which is the only "
            "direction leakage can push it"
        )
        return 1
    below = [e["arm"] for e in analysis.guard if e["verdict"] == "BELOW"]
    print(
        f"PASS leakage guard for {len(analysis.guard)} arm(s), one-directional "
        f"(no arm above floor + {LEAK_TOLERANCE_PP:g}pp)"
        + (f"; below the floor but not failing: {', '.join(below)}" if below else "")
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.failures:
        if not args.arm or not args.question:
            print("FAIL --failures requires --arm and --question")
            return 2
        results_dir = Path(args.results)
        if not results_dir.is_dir():
            print(f"FAIL --results {results_dir} is not a directory")
            return 2
        return failure_blocks(analyse(results_dir), args.arm, args.question, args.limit)
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
