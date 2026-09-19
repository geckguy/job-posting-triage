"""Triage arbitrary postings with the real arms.

    uv run python -m triage.demo --source linkedin --n 200 \
        --arm classifier_dev,llm_hosted --questions fraudulent
    uv run python -m triage.demo --source linkedin --n 20 --questions fraudulent
    uv run python -m triage.demo --source dataset --n 5 --split test

Arms are built through ``triage.run.build_arm`` -- the single construction site --
and run with ``targets=None``, because a demo posting has no gold to narrow the
work by. The run writes one Markdown artifact at ``<out>/demo_<source>.md``; it is
rendered in memory and written once, so an interrupted run never leaves a
half-written file.

Nothing here is scored. The LinkedIn sample is unlabeled, so the artifact reports
flags, confidence and agreement -- never accuracy.
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from statistics import fmean
from typing import Any, Iterable
from urllib.parse import urlparse

from triage import data, run
from triage.arms.base import Answer, ArmPartial, ArmUnavailable
from triage.arms.classifier_dev import API_URL, ClassifierDevArm
from triage.schema import Question, select

LINKEDIN_DATASET = "xanderios/linkedin-job-postings"
LINKEDIN_SPLIT = "train"
#: The demo state is deliberately shorter than ``schema.STATE_CHAR_LIMIT``: this
#: is a reading exercise on real postings, and the artifact quotes the same text
#: the arms saw.
LINKEDIN_STATE_LIMIT = 4_000
SNIPPET_CHARS = 200
POSITIVE_LABEL = "fraudulent"
NEGATIVE_LABEL = "legitimate"
DEFAULT_THRESHOLD = 0.9


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="triage.demo")
    parser.add_argument("--source", default="dataset", choices=["dataset", "linkedin"])
    parser.add_argument("--n", type=int, default=200,
                        help="how many postings to triage")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the uniform random draw of postings")
    parser.add_argument("--arm", default="classifier_dev",
                        help=f"comma-separated; {' , '.join(run.KNOWN_ARMS)}")
    parser.add_argument("--questions", default="fraudulent",
                        help="comma-separated subset; default fraudulent")
    parser.add_argument("--split", default="test", choices=list(data.SPLITS),
                        help="only used by --source dataset")
    parser.add_argument("--out", default="results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="p_positive at or above which a posting is a high-confidence flag")
    parser.add_argument("--tier", default="fast", choices=["fast", "smart"],
                        help="classifier.dev tier")
    parser.add_argument("--no-instructions", dest="instructions",
                        action="store_false", default=True)
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint (default http://127.0.0.1:8080)")
    parser.add_argument("--model", default="local")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default=None,
                        help="name of the env var holding the key (never logged)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="LLM arm output budget; forwarded to triage.run when given "
                             "(the default there is sized for reasoning models)")
    parser.add_argument("--header", action="append", default=None, metavar="NAME=VALUE",
                        help="extra request header for the LLM arms (the value is never logged)")
    parser.add_argument("--response-format", default=None,
                        help="LLM arm output constraint; forwarded to triage.run when given")
    return parser.parse_args(argv)


def runner_args(args: argparse.Namespace) -> argparse.Namespace:
    """A ``triage.run`` namespace carrying the settings this CLI exposes."""
    runner = run.parse_args([])
    runner.arms = args.arm
    runner.questions = args.questions
    runner.out = args.out
    runner.tier = args.tier
    runner.instructions = args.instructions
    runner.split = args.split
    runner.base_url = args.base_url
    runner.model = args.model
    runner.api_key = args.api_key
    runner.api_key_env = args.api_key_env
    runner.concurrency = args.concurrency
    # Forwarded only when given, so this keeps working if triage.run grows or
    # renames its LLM options.
    for name in ("header", "response_format", "max_tokens"):
        value = getattr(args, name, None)
        if value:
            setattr(runner, name, value)
    return runner


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def linkedin_state(title: str, location: str, description: str) -> str:
    """``title``\\n``location``\\n\\n``description`` -- missing fields dropped."""
    head = "\n".join(part for part in (title, location) if part)
    body = description[:LINKEDIN_STATE_LIMIT]
    return f"{head}\n\n{body}".strip()


def _linkedin_posting(index: int, row: dict[str, Any]) -> dict[str, str]:
    title = _clean(row.get("title"))
    location = _clean(row.get("location"))
    description = _clean(row.get("description"))
    # Both columns exist in this mirror; either may be empty on a given row.
    url = _clean(row.get("job_posting_url")) or _clean(row.get("application_url"))
    return {
        "record_id": f"linkedin-{index}",
        "title": title,
        "url": url,
        "state": linkedin_state(title, location, description),
    }


def load_postings(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """A uniform random draw of ``--n`` postings, plus what the draw covered.

    A prefix of a split is a biased frame: this corpus clusters, so the first
    rows over-represent some campaigns (a run of one employer's postings, for
    instance) and the artifact would report that cluster rather than the corpus.
    Indices are drawn without replacement from the whole split, so the sample is
    uniform over rows and reproducible from the seed.
    """
    n = max(0, args.n)
    if args.source == "linkedin":
        from datasets import load_dataset

        dataset = load_dataset(LINKEDIN_DATASET, split=LINKEDIN_SPLIT)
        total = dataset.num_rows
        drawn = draw_indices(total, n, args.seed)
        postings = [_linkedin_posting(index, dataset[index]) for index in drawn]
    else:
        records = data.records(args.split)
        total = len(records)
        drawn = draw_indices(total, n, args.seed)
        postings = [
            {
                "record_id": records[index]["record_id"],
                "title": records[index]["state"].split("\n\n", 1)[0].strip(),
                "url": "",  # the Kaggle mirror has no posting URL
                "state": records[index]["state"],
            }
            for index in drawn
        ]
    return postings, {"total": total, "drawn": drawn, "seed": args.seed}


def draw_indices(total: int, n: int, seed: int) -> list[int]:
    """``n`` row indices drawn uniformly without replacement, in ascending order.

    Sorted so the artifact's rows follow dataset order whatever the draw was;
    the sample itself is the draw, not the first ``n``.
    """
    take = min(n, total)
    if take <= 0:
        return []
    return sorted(random.Random(seed).sample(range(total), take))


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def p_positive(answer: Answer) -> float | None:
    """P(the positive class, i.e. ``fraudulent``) for one answer.

    ``probabilities[POSITIVE_LABEL]`` when the arm reports a distribution;
    otherwise its confidence in its own label, mirrored when the label is the
    negative one (``legitimate`` at 0.9 implies P(fraudulent) = 0.1). ``None``
    when the arm reported neither -- rendered ``n/a``, never a fabricated 0.
    """
    probabilities = answer.probabilities
    if probabilities:
        score = probabilities.get(POSITIVE_LABEL)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
    confidence = answer.confidence
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        confidence = float(confidence)
        return confidence if answer.value == POSITIVE_LABEL else 1.0 - confidence
    return None


def endpoint_host(arm: Any) -> str:
    """The host this arm actually talks to; empty when it talks to nothing."""
    base = getattr(arm, "base_url", None)
    if base:
        return urlparse(str(base)).netloc or str(base)
    if isinstance(arm, ClassifierDevArm):
        return urlparse(API_URL).netloc
    return ""


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def display_label(label: str, host: str) -> str:
    """What the artifact calls this arm, given the endpoint it really used.

    An ``llm_hosted`` arm pointed at loopback is the local llama-server reached
    through the OpenAI-compatible path -- the same model as ``llm_local`` -- and
    calling it "hosted" would be a false claim in the README.
    """
    name = host.split(":")[0].strip("[]").lower()
    if label.startswith("llm_") and name in LOOPBACK_HOSTS:
        parts = label.split("_", 2)
        suffix = f"_{parts[2]}" if len(parts) == 3 else ""
        return f"llm_local_openai_api{suffix}"
    return label


def run_arms(
    names: list[str],
    runner: argparse.Namespace,
    postings: list[dict[str, str]],
    questions: tuple[Question, ...],
) -> list[dict[str, Any]]:
    """Run each arm; an arm that cannot run is reported, never fatal."""
    records = [{"record_id": p["record_id"], "state": p["state"]} for p in postings]
    entries: list[dict[str, Any]] = []
    for name in names:
        entry: dict[str, Any] = {
            "requested": name,
            "label": name,
            "display": name,
            "endpoint": "",
            "status": "ok",
            "reason": "",
            "answers": {},
            "seconds": 0.0,
            "usage": None,
        }
        entries.append(entry)
        try:
            label, arm = run.build_arm(name, runner)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            entry["status"] = "unavailable"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[{name}] unavailable: {entry['reason']}")
            continue

        entry["label"] = label
        entry["endpoint"] = endpoint_host(arm)
        entry["display"] = display_label(label, entry["endpoint"])
        started = time.perf_counter()
        try:
            ok, why = arm.available()
            if not ok:
                entry["status"] = "unavailable"
                entry["reason"] = why
            else:
                entry["answers"] = arm.run(records, questions, targets=None)
        except ArmPartial as exc:
            entry["status"] = "partial"
            entry["reason"] = str(exc)
            entry["answers"] = exc.answers
        except ArmUnavailable as exc:
            entry["status"] = "unavailable"
            entry["reason"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            entry["status"] = "failed"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            entry["seconds"] = round(time.perf_counter() - started, 1)
            usage = getattr(arm, "usage", None)
            if usage:
                entry["usage"] = dict(usage)
            arm.close()

        answered = sum(len(v) for v in entry["answers"].values())
        note = f" -- {entry['reason']}" if entry["reason"] else ""
        print(f"[{entry['display']}] {entry['status']}: {answered} answers "
              f"in {entry['seconds']}s{note}")
    return entries


def _answer(entry: dict[str, Any], question_id: str, record_id: str) -> Answer | None:
    return entry["answers"].get(question_id, {}).get(record_id)


def _label(entry: dict[str, Any], question_id: str, record_id: str) -> str | None:
    """The arm's usable label, or ``None`` for no answer / invalid output."""
    answer = _answer(entry, question_id, record_id)
    if answer is None or not answer.value or answer.invalid:
        return None
    return answer.value


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[:limit] + "..."
    return flat.replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _source_line(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    n = len(sample.get("drawn", ()))
    where = (f"`{LINKEDIN_DATASET}` (split `{LINKEDIN_SPLIT}`)"
             if args.source == "linkedin" else f"`{data.DATASET}` (split `{args.split}`)")
    return (f"{where}, {sample['total']:,} rows; uniform random draw of {n} "
            f"(`random.Random({sample['seed']}).sample`, no replacement)")


def _coverage_line(args: argparse.Namespace, sample: dict[str, Any]) -> str:
    drawn = sample.get("drawn") or []
    if not drawn:
        return "- **Rows drawn**: none"
    span = f"{drawn[0]}-{drawn[-1]}"
    return (f"- **Rows drawn**: {len(drawn)} of {sample['total']:,}, indices {span} "
            f"(record ids are `{('linkedin' if args.source == 'linkedin' else args.split)}-"
            f"<row index>`)")


def render(
    args: argparse.Namespace,
    postings: list[dict[str, str]],
    questions: tuple[Question, ...],
    entries: list[dict[str, Any]],
    sample: dict[str, Any],
) -> str:
    record_ids = [p["record_id"] for p in postings]
    question_ids = [q.id for q in questions]
    n = len(postings)

    lines: list[str] = [
        f"# Demo -- {'LinkedIn' if args.source == 'linkedin' else 'labeled'} postings",
        "",
        f"- **Source**: {_source_line(args, sample)}",
        f"- **n**: {n} postings",
        _coverage_line(args, sample),
        f"- **Seed**: `{sample['seed']}` (same seed and `--n` reproduce this exact sample)",
        "- **Questions asked**: " + ", ".join(f"`{qid}`" for qid in question_ids),
        f"- **High-confidence threshold**: `p_positive >= {args.threshold}`",
        "- **Arms**: " + ", ".join(f"`{e['display']}`" for e in entries),
        f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"by `python -m triage.demo`",
        "",
        "The sample is a uniform random draw over the whole split, not the first rows: a prefix "
        "of this corpus is a biased frame, because postings cluster (a single employer's campaign "
        "can occupy a run of consecutive rows), so a prefix would report that campaign's flag "
        "rate as if it were the corpus's.",
        "",
        "## Arms",
        "",
    ]

    arm_rows = []
    for entry in entries:
        answered = sum(len(v) for v in entry["answers"].values())
        usable = sum(1 for answers in entry["answers"].values() for a in answers.values()
                     if a.value and not a.invalid)
        invalid = sum(1 for answers in entry["answers"].values() for a in answers.values()
                      if a.invalid)
        note = entry["reason"] or "ok"
        if entry["usage"]:
            usage = entry["usage"]
            spent = usage.get("classifications")
            cached = usage.get("cached")
            if spent is not None:
                note = (f"{note} | classifications: {spent}"
                        + (f", cached: {cached}" if cached else ""))
        requested = (f"`{entry['requested']}`" if entry["requested"] != entry["display"]
                     else "same")
        arm_rows.append([f"`{entry['display']}`", requested,
                         entry["endpoint"] or "(in-process)", entry["status"],
                         str(answered), str(usable), str(invalid), f"{entry['seconds']:.1f}",
                         note])
    lines += _table(
        ["arm (artifact name)", "requested as", "endpoint host", "status", "returned",
         "usable", "invalid", "seconds", "notes"], arm_rows)
    lines += [
        "",
        "The endpoint column is the host each arm actually talked to, so a hosted run cannot be "
        "confused with a local one; API keys are never recorded. `(in-process)` means the model "
        "or the features ran on this machine with no network call.",
        "",
        "`returned` counts the answers an arm handed back; `usable` counts those with a non-empty "
        "`value` that the arm did not itself mark `invalid`; `invalid` counts the rest. An arm "
        "that returned answers and 0 usable ones answered nothing, and `status: ok` there means "
        "only that the arm did not raise, not that its answers are labels. Those postings are "
        "counted as `(no answer)`, never as a negative label.",
        "",
        "`unavailable`/`failed` means the arm produced nothing here and the reason is printed "
        "above; no other model, dataset or metric was substituted for it.",
        "",
        "`llm_local_openai_api` is an arm requested as `llm_hosted` whose base URL resolved to "
        "loopback: the local llama-server reached through the OpenAI-compatible HTTP path, i.e. "
        "the same model as `llm_local`, not a hosted model. Rows produced that way are never "
        "called hosted.",
        "",
    ]
    lines += _invalid_reasons(entries)
    lines += [
        "## Probability convention",
        "",
        "`p_positive` = P(`fraudulent`) for one answer: `probabilities[\"fraudulent\"]` when the arm "
        "reports a distribution (classifier_dev, tfidf, gliner); otherwise the arm's self-reported "
        "confidence in its own label, mirrored when the label is the negative one "
        "(`legitimate` at 0.9 implies `p_positive` 0.1) -- the LLM arms. `n/a` means the arm "
        "reported neither field, so the posting cannot be thresholded for that arm. `confidence` "
        "is always the arm's confidence in the label it chose.",
        "",
    ]

    # -- per-question label counts ----------------------------------------
    for question in questions:
        labels = [*question.labels, "(no answer)"]
        rows = []
        for entry in entries:
            counts = dict.fromkeys(labels, 0)
            for record_id in record_ids:
                value = _label(entry, question.id, record_id)
                counts[value if value in counts else "(no answer)"] += 1
            rows.append([f"`{entry['display']}`", *[str(counts[label]) for label in labels]])
        lines += [f"## Labels -- `{question.id}`", "",
                  f"Labels: " + ", ".join(f"`{label}`" for label in question.labels), ""]
        lines += _table(["arm", *labels], rows)
        lines += [""]

    # -- fraud-specific analysis ------------------------------------------
    if POSITIVE_LABEL in question_ids:
        lines += _fraud_sections(postings, entries, args.threshold)
    else:
        lines += [
            "## Flag analysis",
            "",
            f"`{POSITIVE_LABEL}` was not among the questions asked, so the flag rate, the "
            "agreement table and the high-confidence list are not defined for this run. "
            "Re-run with `--questions fraudulent` for those.",
            "",
        ]

    lines += _unlabeled_note(args, sample)
    return "\n".join(lines) + "\n"


def _mirrored_hit(
    entries: list[dict[str, Any]], record_id: str, threshold: float
) -> bool:
    """Did a confidence-only arm land this posting above ``threshold`` by mirroring?

    Only true for a *negative* call: a confidence-only arm that called the
    posting positive reaches the threshold through its own label, not the mirror.
    """
    for entry in entries:
        answer = _answer(entry, POSITIVE_LABEL, record_id)
        if answer is None or not answer.value or answer.invalid:
            continue
        if answer.probabilities or answer.confidence is None:
            continue
        if answer.value == POSITIVE_LABEL:
            continue
        score = _score(entry, record_id)
        if score is not None and score >= threshold:
            return True
    return False


def _mirror_stats(
    entries: list[dict[str, Any]], record_ids: list[str], threshold: float
) -> dict[str, int]:
    """How much of the threshold count comes from confidence-only arms.

    An arm that reports no distribution gets its ``p_positive`` by mirroring its
    own confidence, so a *low-confidence negative* call lands above the threshold.
    That is arithmetic, not evidence, and the artifact says so with counts.
    Counted only over arms that report no distribution at all, so the
    denominators describe those arms rather than the whole run.
    """
    stats = {"mirrored": 0, "low_conf_negatives": 0, "confidence_only_rows": 0,
             "usable_rows": 0}
    for entry in entries:
        answers = entry["answers"].get(POSITIVE_LABEL, {})
        if not any(answer.probabilities is None and answer.confidence is not None
                   for answer in answers.values()):
            continue  # this arm reports a distribution; the mirror never applies
        for record_id in record_ids:
            answer = _answer(entry, POSITIVE_LABEL, record_id)
            if answer is None or not answer.value or answer.invalid:
                continue
            stats["usable_rows"] += 1
            if answer.probabilities or answer.confidence is None:
                continue
            stats["confidence_only_rows"] += 1
            if answer.value != POSITIVE_LABEL and answer.confidence < 0.5:
                stats["low_conf_negatives"] += 1
                score = _score(entry, record_id)
                if score is not None and score >= threshold:
                    stats["mirrored"] += 1
    return stats


def _mirror_note(stats: dict[str, int], threshold: float, any_arm: int) -> list[str]:
    if not stats["mirrored"]:
        return []
    share = f"{stats['mirrored']}/{any_arm}" if any_arm else str(stats["mirrored"])
    return [
        f"**Read this before quoting that number.** {share} of the `p_positive >= {threshold}` "
        f"hits come from an arm that reports no distribution: it labelled the posting "
        f"`{NEGATIVE_LABEL}` while stating confidence below 0.5, and the mirror rule turns a "
        f"low-confidence `{NEGATIVE_LABEL}` call into a high `p_positive` "
        f"(`{NEGATIVE_LABEL}` at 0.01 implies `p_positive` 0.99). That arm gave "
        f"{stats['usable_rows']} usable answers, {stats['confidence_only_rows']} of which became "
        f"mirrored scores, {stats['low_conf_negatives']} of those from low-confidence "
        f"`{NEGATIVE_LABEL}` calls. So the threshold count measures how unsure the arm is, not "
        f"how many postings look fraudulent, and quoting it as a flag rate would be false. For "
        f"an arm in that state, read its label column (and its confidence directly) instead of "
        f"its mirrored score.",
        "",
    ]


def _fraud_sections(postings, entries, threshold: float) -> list[str]:
    record_ids = [p["record_id"] for p in postings]
    n = len(postings)
    lines: list[str] = []

    rows = []
    for entry in entries:
        flagged = 0
        scored: list[float] = []
        confidences: list[float] = []
        for record_id in record_ids:
            if _label(entry, POSITIVE_LABEL, record_id) == POSITIVE_LABEL:
                flagged += 1
            score = _score(entry, record_id)
            if score is not None:
                scored.append(score)
            answer = _answer(entry, POSITIVE_LABEL, record_id)
            if answer is not None and answer.confidence is not None:
                confidences.append(answer.confidence)
        rate = flagged / n if n else 0.0
        rows.append([
            f"`{entry['display']}`",
            f"{flagged}/{n}",
            _fmt(rate, 3),
            f"{len(scored)}/{n}",
            _fmt(fmean(scored)) if scored else "n/a",
            _fmt(fmean(confidences)) if confidences else "n/a",
        ])
    lines += [
        "## Fraud-flag rate",
        "",
        "Flagged = the arm's `value` is `fraudulent` (not a threshold on probability). "
        "`scored` is how many postings the arm produced a `p_positive` for; mean confidence is "
        "the arm's own certainty in the label it chose, and a mean near 0 with a mean `p_positive` "
        "near 1 is the degenerate-confidence pattern described under the flag table below.",
        "",
    ]
    lines += _table(
        ["arm", "flagged", "rate", "scored", "mean p_positive", "mean confidence"], rows)
    lines += [""]

    # pairwise agreement
    agree_rows = []
    for i, left in enumerate(entries):
        for right in entries[i + 1:]:
            compared = agreeing = both = 0
            for record_id in record_ids:
                a = _label(left, POSITIVE_LABEL, record_id)
                b = _label(right, POSITIVE_LABEL, record_id)
                if a is None or b is None:
                    continue
                compared += 1
                agreeing += int(a == b)
                both += int(a == POSITIVE_LABEL and b == POSITIVE_LABEL)
            rate = agreeing / compared if compared else None
            agree_rows.append([
                f"`{left['display']}`", f"`{right['display']}`",
                str(compared), str(agreeing), _fmt(rate, 3), str(both),
            ])
    lines += [
        "## Agreement on the fraud label",
        "",
        "Compared = postings where both arms returned a label, so rows with an empty `value` "
        "drop out of the pair rather than counting as disagreement.",
        "",
    ]
    answering = [e["display"] for e in entries
                 if any(_label(e, POSITIVE_LABEL, rid) is not None for rid in record_ids)]
    if agree_rows and len(answering) >= 2:
        lines += _table(["arm A", "arm B", "compared", "agree", "agreement", "both flagged"],
                        agree_rows)
    else:
        if not answering:
            why = "No arm returned a label here, so there is no pair to compare."
        elif len(answering) == 1:
            why = (f"Only `{answering[0]}` returned labels here, so there is no pair "
                   "to compare.")
        else:
            why = ("No pair of arms returned a label for the same posting, so there is "
                   "nothing to compare.")
        lines += [why + " This demo does not score gold, so no accuracy is reported."]
    lines += [""]

    scoring: list[tuple[str, set[str]]] = []
    for entry in entries:
        scores = {rid: _score(entry, rid) for rid in record_ids}
        if not any(score is not None for score in scores.values()):
            continue
        scoring.append((entry["display"],
                        {rid for rid, score in scores.items()
                         if score is not None and score >= threshold}))
    every = sorted(set.intersection(*(ids for _, ids in scoring))) if scoring else []
    any_arm = sorted(set().union(*(ids for _, ids in scoring))) if scoring else []
    labelled = sorted({rid for entry in entries for rid in record_ids
                       if _label(entry, POSITIVE_LABEL, rid) == POSITIVE_LABEL})
    listed = sorted(set(any_arm) | set(labelled))
    lines += [
        f"## Postings flagged at `p_positive >= {threshold}`, or labelled `{POSITIVE_LABEL}`",
        "",
        f"At `p_positive >= {threshold}` by every arm that could score a posting: "
        f"**{len(every)}/{n}** ({', '.join('`%s`' % label for label, _ in scoring) or 'no arm scored'}). "
        f"At `p_positive >= {threshold}` by at least one arm: **{len(any_arm)}/{n}**. "
        f"Labelled `{POSITIVE_LABEL}` by at least one arm, at any score: **{len(labelled)}/{n}**.",
        "",
    ]
    mirror = _mirror_stats(entries, record_ids, threshold)
    lines += _mirror_note(mirror, threshold, len(any_arm))
    if not listed:
        lines += [
            f"No posting reached `p_positive >= {threshold}` for any arm, and no arm labelled any "
            f"posting `{POSITIVE_LABEL}`.",
            "",
        ]
        return lines

    by_id = {p["record_id"]: p for p in postings}
    header = ["#", "record_id", "why listed", "title", "url"]
    for entry in entries:
        header += [f"`{entry['display']}`", f"`{entry['display']}` p_positive"]
    header.append(f"state (first {SNIPPET_CHARS} chars)")
    rows = []
    for position, record_id in enumerate(listed, start=1):
        posting = by_id[record_id]
        if record_id in any_arm:
            why = f"p>={threshold}"
            if _mirrored_hit(entries, record_id, threshold):
                why = f"p>={threshold} (mirrored conf)"
        else:
            score = max((s for s in (_score(e, record_id) for e in entries)
                         if s is not None), default=None)
            why = f"label only, p={_fmt(score, 2)}"
        row = [str(position), f"`{record_id}`", why,
               posting["title"].replace("|", "\\|"),
               posting["url"] or "(no url)"]
        for entry in entries:
            answer = _answer(entry, POSITIVE_LABEL, record_id)
            score = _score(entry, record_id)
            if answer is None or not answer.value:
                row += ["(no answer)", "n/a"]
            elif answer.invalid:
                row += ["(invalid output)", _fmt(score, 3)]
            else:
                row += [f"{answer.value} ({_fmt(answer.confidence, 2)})", _fmt(score, 3)]
        row.append(_snippet(posting["state"]))
        rows.append(row)
    lines += _table(header, rows)
    lines += [
        "",
        "`n/a` in the `p_positive` column means that arm returned neither a distribution nor a "
        "confidence for the posting, so the threshold cannot be evaluated for it; it is not "
        "counted as a low score. A row can appear with "
        f"`why listed = p>={threshold}` while an arm's *label* is `legitimate`: that would mean "
        "the arm reported a distribution whose positive mass reached the threshold. Rows listed "
        "as `label only` are near-ties, not confident flags: an argmax at `p_positive` 0.5x can "
        "carry a calibrated confidence near zero, and the two numbers mean different things "
        "(mass on the positive class vs. certainty in the label chosen).",
        "",
    ]
    return lines


def _score(entry: dict[str, Any], record_id: str) -> float | None:
    answer = _answer(entry, POSITIVE_LABEL, record_id)
    return None if answer is None else p_positive(answer)


def _invalid_reasons(entries: list[dict[str, Any]]) -> list[str]:
    """The exact reason every invalid answer failed, so a reader can act on it.

    An invalid count without its reasons is a number nobody can debug, and for a
    reasoning model the usual reason is the output budget running out before the
    JSON is emitted.
    """
    blocks: list[str] = []
    for entry in entries:
        reasons: Counter[str] = Counter()
        for answers in entry["answers"].values():
            for answer in answers.values():
                if not answer.invalid:
                    continue
                meta = answer.meta or {}
                reason = str(meta.get("error") or "no error recorded")
                finish = meta.get("finish_reason")
                if finish and "finish_reason" not in reason:
                    reason = f"{reason} (finish_reason={finish})"
                reasons[reason] += 1
        if not reasons:
            continue
        blocks += [f"### Invalid answers: `{entry['display']}`", ""]
        for reason, count in reasons.most_common():
            blocks.append(f"- **{count}x** {_snippet(reason, 300)}")
        blocks.append("")
    if not blocks:
        return []
    return ["## Why answers were invalid", "",
            "Every answer an arm marked `invalid` is excluded from its labels and counted as "
            "`(no answer)`. The reasons below are verbatim from the arm's own metadata:", ""] + blocks


def _unlabeled_note(args: argparse.Namespace, sample: dict[str, Any]) -> list[str]:
    if args.source == "linkedin":
        return [
            "## This sample is UNLABELED",
            "",
            "Nothing above is accuracy. There is no gold fraud label for these LinkedIn postings, "
            "and the demo never invents one: the numbers are **flag counts, flag rates and "
            "agreement between arms** on this sample only. They say nothing about the population "
            "of LinkedIn postings, nothing about any arm's error rate, and two arms agreeing is "
            "not evidence that either is right -- a shared failure mode looks exactly like "
            "agreement. Accuracy for a given arm lives in `results/report.md`, on the labeled "
            "Kaggle test split.",
            "",
            "Caveats specific to this artifact: the state each arm saw is "
            f"`title`\\n`location`\\n\\n`description[:{LINKEDIN_STATE_LIMIT}]` and nothing else "
            "(no salary columns, no company data), so a posting whose scam cues sit past the "
            "truncation point looks clean to every arm; the sample is a uniform random draw of "
            f"{len(sample.get('drawn', ()))} rows from the whole split under a fixed seed, which "
            "makes it reproducible and unbiased in expectation but still one sample, so its rates "
            "carry sampling error and are not corpus rates; and `job_posting_url` / "
            "`application_url` are carried through unverified -- this demo did not fetch any of "
            "them.",
            "",
        ]
    return [
        "## This source is labeled -- but this run did not score it",
        "",
        f"`{data.DATASET}` ships a `fraudulent` column, so accuracy is computable here; "
        "`triage.run` + `triage.evaluate` are the scoring path. This artifact deliberately "
        "reports only what it computed: flag counts and agreement, without reading gold. "
        "Read the flag rates as a sanity check on the demo wiring, and `results/report.md` "
        "for the metrics.",
        "",
    ]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    names = [name.strip() for name in args.arm.split(",") if name.strip()]
    unknown = [name for name in names if name not in run.KNOWN_ARMS]
    if unknown:
        raise SystemExit(
            f"unknown arm(s) {unknown}; expected {', '.join(run.KNOWN_ARMS)}"
        )

    questions = select(args.questions.split(",") if args.questions else None)
    postings, sample = load_postings(args)
    if not postings:
        raise SystemExit(f"--n {args.n} selected no postings from {args.source}")
    print(f"{args.source}: {len(postings)} of {sample['total']} postings drawn with "
          f"seed {sample['seed']}, " + ", ".join(f"{q.id}" for q in questions))

    entries = run_arms(names, runner_args(args), postings, questions)
    path = Path(args.out) / f"demo_{args.source}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(args, postings, questions, entries, sample), encoding="utf-8")
    print(f"wrote {path}")

    productive = [e for e in entries
                  if any(a.value and not a.invalid
                         for answers in e["answers"].values() for a in answers.values())]
    if not productive:
        print("no arm produced a usable label; the artifact records why")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
