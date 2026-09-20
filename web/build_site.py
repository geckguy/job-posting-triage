#!/usr/bin/env python3
"""Generate the static showcase site for the job-posting-triage benchmark.

Reads ``results/metrics.json`` (plus ``calibration.png``, ``routing.png`` and
``report.md`` from the same directory) and writes ``docs/index.html``,
``docs/.nojekyll``, copies of those assets and a copy of ``web/styles.css``.

Every number on the page comes from ``metrics.json``. Nothing is estimated,
interpolated or defaulted: a value that is ``null`` renders as ``n/a`` together
with the reason text from the ``n_a`` list, and a top-level key the page needs
is a hard error.

    uv run python web/build_site.py
    uv run python web/build_site.py --results /tmp/site_fixture --out /tmp/site_out
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

#: Where the generated site points for source and reproduction.
REPO_URL = "https://github.com/geckguy/job-posting-triage"

#: The question whose table, calibration and routing the page is built around.
FRAUD = "fraudulent"

#: Top-level keys the page cannot render without. A missing one is fatal.
REQUIRED_KEYS = (
    "split",
    "window",
    "records_sent",
    "arms",
    "unavailable",
    "per_question",
    "routing",
    "leakage_guard",
    "calibration_bins",
    "cost",
    "n_a",
)

#: Files the page needs next to metrics.json. All are copied into the site.
REQUIRED_FILES = ("calibration.png", "routing.png", "report.md")

#: The stylesheet lives in web/ and is copied into the site on every build.
STYLESHEET = Path(__file__).with_name("styles.css")

SECTIONS = (
    ("measurement", "The measurement"),
    ("fraud", "Fraud triage"),
    ("example", "One posting, four verdicts"),
    ("engines", "The four engines"),
    ("calibration", "Calibration"),
    ("routing", "Routing and cost"),
    ("checks", "Method checks"),
    ("limits", "What this does not prove"),
    ("reproduce", "Reproduce it"),
)

#: The section ids build_page writes, in page order. The rail is generated from
#: SECTIONS while the markup is written by hand, so the two lists can drift. A
#: renamed anchor costs nothing visible: the rail script drops any link whose
#: target is missing, and the nav item then scrolls nowhere in silence. build_page
#: refuses to build when the lists disagree.
RENDERED_SECTIONS = (
    "measurement",
    "fraud",
    "example",
    "engines",
    "calibration",
    "routing",
    "checks",
    "limits",
    "reproduce",
)

#: Arms fitted on this corpus rather than asked zero-shot. The page marks them in
#: the fraud table and in the headline numbers, since a fitted row is not a
#: zero-shot result and reads as one otherwise.
FITTED_ARMS = frozenset({"tfidf"})

#: What each arm is, sourced from triage/arms/*.py and the runner's defaults.
#: The hosted arm carries its response-format in the label, so both spellings
#: share one description.
HOSTED_ARM_INFO = (
    "The same LLM code path pointed at a hosted model over an OpenAI-compatible endpoint, kept "
    "as its own row so the two are never confused. The gateway rejects json_schema, so the "
    "reply shape goes in the prompt and the reply is validated afterwards."
)

ARM_INFO: dict[str, str] = {
    "classifier_dev": (
        "Jev 1.13.0, served by classifier.dev. One request carries up to 1,000 postings and "
        "returns a probability for every label of a question in a single pass. No tokens generated."
    ),
    "classifier_dev_no_rubric": (
        "The same Jev arm with the per-question rubric text dropped from the request. This is the "
        "ablation for what the instruction text buys."
    ),
    "gliner": (
        "fastino/gliner2.5-base-v1, a 194M-parameter zero-shot encoder. All four questions are "
        "compiled into one schema and read in one forward pass per posting. Its window is 512 "
        "subwords, so long postings are truncated before they are scored."
    ),
    "llm_local": (
        "Qwen2.5-1.5B-Instruct Q4_K_M, served by llama-server on this machine. One chat "
        "completion per posting answers all four questions under a strict JSON schema at "
        "temperature 0. The model reports its own confidence."
    ),
    "llm_hosted": HOSTED_ARM_INFO,
    "llm_hosted_json_object": HOSTED_ARM_INFO,
    "tfidf": (
        "TF-IDF over word unigrams and bigrams, then one logistic regression per question, fitted "
        "on the train and validation rows that carry a gold label."
    ),
}

GENERIC_ARM_INFO = "No description of this arm is recorded in the generator."

#: Which engine each arm is a run of. Variants (a rubric ablation, a hosted
#: endpoint) share the engine of the arm they vary, so the page can say how many
#: engines produced the arms it lists without counting variants twice.
ARM_FAMILY: dict[str, str] = {
    "classifier_dev": "jev",
    "classifier_dev_no_rubric": "jev",
    "gliner": "gliner",
    "tfidf": "tfidf",
    "llm_local": "llm",
    "llm_hosted": "llm",
    "llm_hosted_json_object": "llm",
}

ENGINE_LABEL: dict[str, str] = {
    "jev": "Jev, served by classifier.dev",
    "gliner": "the GLiNER2.5-base zero-shot encoder",
    "tfidf": "TF-IDF with logistic regression",
    "llm": "Qwen2.5-1.5B-Instruct, served locally and over a hosted endpoint",
}

#: The literal command sequence the footer prints. {window} comes from the run.
REPRODUCE = (
    "uv sync",
    "uv run hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models",
    "llama-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf --jinja -c 16384 -np 4 "
    "--cache-ram 128 --port 8080 --host 127.0.0.1",
    "uv run python -m triage.run --arms all --split test --limit {limit} --seed {seed} --concurrency 4",
    "uv run python -m triage.evaluate --report",
    "uv run python web/build_site.py",
)

#: Why the cache-ram flag is in the command above: without it llama.cpp reserves
#: an 8 GiB prompt cache, which is what put this machine into swap during the run.
CACHE_NOTE = (
    "Keep <code>--cache-ram 128</code> in the server line: the default 8 GiB prompt cache is what "
    "filled memory and put this machine into swap during the run."
)

#: The posting excerpt the hero may show. The repo does not redistribute dataset
#: text beyond this window, so the cap is not negotiable.
EXCERPT_CHARS = 400


class BuildError(RuntimeError):
    """A missing or unusable input. The message names exactly what was wrong."""


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def share(value: float) -> str:
    return f"{value * 100:.1f}%"


def score(value: float) -> str:
    return f"{value:.3f}"


def millis(value: float) -> str:
    """Latency in milliseconds, with enough precision that a real measurement
    never renders as 0.0."""
    if value < 1:
        return f"{value:.3f} ms"
    if value < 100:
        return f"{value:.1f} ms"
    return f"{value:,.0f} ms"


def count(value: float) -> str:
    return f"{value:,.0f}"


def interval(bounds: Any) -> str:
    return f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"


def tokens(value: float) -> str:
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_metrics(results_dir: Path) -> dict:
    path = results_dir / "metrics.json"
    if not path.exists():
        raise BuildError(
            f"missing {path} (run `uv run python -m triage.evaluate --report` first)"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise BuildError(f"{path} must hold a JSON object, found {type(doc).__name__}")
    missing = [key for key in REQUIRED_KEYS if key not in doc]
    if missing:
        raise BuildError(
            f"{path} is missing the key(s) {', '.join(missing)}: the page cannot be rendered "
            "without them"
        )
    if FRAUD not in (doc.get("per_question") or {}):
        raise BuildError(f"{path} has no per_question.{FRAUD} block")
    return doc


def require_files(results_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (results_dir / name).exists()]
    if missing:
        listed = ", ".join(str(results_dir / name) for name in missing)
        raise BuildError(f"missing result file(s): {listed}")


class NAReasons:
    """Looks a null up in the ``n_a`` list, so an n/a cell always carries a why."""

    def __init__(self, entries: Any) -> None:
        self.entries = [e for e in entries or [] if isinstance(e, dict)]
        self.index: dict[tuple[Any, Any, str], str] = {}
        for entry in self.entries:
            reason = entry.get("reason")
            if isinstance(reason, str) and reason.strip():
                key = (entry.get("arm"), entry.get("question"), str(entry.get("metric")))
                self.index.setdefault(key, reason.strip())

    def __call__(self, arm: str | None, question: str | None, metric: str) -> str:
        # A nested metric such as per_class.fraudulent.recall is reported in n_a
        # under the name the evaluator uses for it, which may be the base name, so
        # every spelling is tried before giving up.
        names = [metric]
        if "." in metric:
            names.append(metric.split(".", 1)[0])
            names.append(metric.rsplit(".", 1)[-1])
        # The evaluator spells a per-class cell per_class_precision[<label>] while
        # the tables ask for per_class.<label>.precision, so without this the cell
        # renders the no-reason-found fallback instead of the evaluator's reason.
        parts = metric.split(".")
        if len(parts) == 3 and parts[0] == "per_class":
            names.append(f"per_class_{parts[2]}[{parts[1]}]")
        if arm:
            names.append(str(arm))
        for name in names:
            for key in ((arm, question, name), (arm, None, name), (None, question, name)):
                found = self.index.get(key)
                if found:
                    row_level = arm is not None and name == str(arm) and name != metric
                    return f"row-level reason: {found}" if row_level else found
        return (
            f"metrics.json carries no value for {metric} on {arm} and no reason for it in n_a"
        )


# ---------------------------------------------------------------------------
# table plumbing
# ---------------------------------------------------------------------------


def na_cell(reason: str) -> str:
    return f'<td class="num"><span class="na" title="{esc(reason)}">n/a</span></td>'


class Table:
    """Rows plus the n/a reasons they raised, rendered as a note list underneath."""

    def __init__(self, reasons: NAReasons, columns: list[tuple[str, str]]) -> None:
        self.reasons = reasons
        self.columns = columns
        self.rows: list[str] = []
        self.notes: list[tuple[str, str, str]] = []

    def cell(
        self,
        value: Any,
        fmt: Callable[[Any], str],
        *,
        arm: str | None,
        question: str | None,
        metric: str,
    ) -> str:
        if value is None:
            reason = self.reasons(arm, question, metric)
            self.notes.append((arm or "(baseline)", metric, reason))
            return na_cell(reason)
        return f'<td class="num">{fmt(value)}</td>'

    def ci_cell(
        self,
        value: Any,
        bounds: Any,
        fmt: Callable[[Any], str],
        *,
        arm: str | None,
        question: str | None,
        metric: str,
        ci_metric: str,
    ) -> str:
        """A point estimate with its 95% interval under it, in one column.

        The two can be missing separately, and each missing half carries its own
        reason rather than a zero.
        """
        if value is None:
            reason = self.reasons(arm, question, metric)
            self.notes.append((arm or "(baseline)", metric, reason))
            inner = f'<span class="na" title="{esc(reason)}">n/a</span>'
        else:
            inner = f'<span class="point">{fmt(value)}</span>'
        if bounds is None:
            reason = self.reasons(arm, question, ci_metric)
            self.notes.append((arm or "(baseline)", ci_metric, reason))
            inner += f'<span class="na ci" title="{esc(reason)}">n/a</span>'
        else:
            inner += f'<span class="ci">{esc(interval(bounds))}</span>'
        return f'<td class="num">{inner}</td>'

    def row(self, cells: list[str], kind: str = "plain") -> None:
        self.rows.append(f'<tr class="row-{kind}">{"".join(cells)}</tr>')

    def notes_html(self) -> str:
        if not self.notes:
            return ""
        items: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for arm, metric, reason in self.notes:
            key = (arm, metric, reason)
            if key in seen:
                continue
            seen.add(key)
            items.append(f"<li><strong>{esc(arm)}</strong>, {esc(metric)}: {esc(reason)}</li>")
        return (
            '<ul class="notes"><li class="notes-head">Why a cell reads n/a</li>'
            + "".join(items)
            + "</ul>"
        )

    def render(self, caption: str = "") -> str:
        head = "".join(
            f'<th scope="col" class="{cls}">{esc(title)}</th>' for title, cls in self.columns
        )
        return (
            '<div class="table-wrap"><table>'
            f"{caption}<thead><tr>{head}</tr></thead>"
            f'<tbody>{"".join(self.rows)}</tbody></table></div>'
            '<p class="scroll-hint">The table scrolls sideways on a narrow screen.</p>'
            f"{self.notes_html()}"
        )


def q_cell(label: str, chip: str = "", chip_class: str = "plain", kind: str = "plain") -> str:
    badge = f'<span class="chip chip-{chip_class}">{esc(chip)}</span>' if chip else ""
    return (
        f'<td class="arm"><span class="arm-name">{esc(label)}</span>{badge}</td>'
    )


def arm_order(doc: dict) -> list[str]:
    return [str(arm) for arm in doc.get("arms") or []]


def unavailable_map(doc: dict) -> dict:
    return doc.get("unavailable") if isinstance(doc.get("unavailable"), dict) else {}


# ---------------------------------------------------------------------------
# hero
# ---------------------------------------------------------------------------


def record_sort_key(record_id: str) -> tuple[str, int, str]:
    head, _, tail = str(record_id).rpartition("-")
    if tail.isdigit():
        return (head, int(tail), "")
    return (str(record_id), -1, str(record_id))


FRAME_KEYS = ("split", "limit", "seed", "sampling", "records_sent", "window", "window_note")


def load_frame_file(results_dir: Path) -> dict | None:
    """``results/_frame.json``: the frame the runner actually used, when present.

    It is written next to the results rather than inside metrics.json, so it is
    optional. When it is there it is preferred over anything derived from record
    indices, because it records the sampling method and the seed directly.
    """
    path = results_dir / "_frame.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def frame_facts(doc: dict, frame_file: dict | None) -> dict:
    """What is known about the frame, best source first.

    Precedence: ``metrics.json:frame_source`` (the evaluator's own record of how
    the frame was drawn), then ``_frame.json``, then the top-level ``window``
    keys of metrics.json.
    """
    facts: dict[str, Any] = {}
    source = doc.get("frame_source")
    if isinstance(source, dict):
        for key in FRAME_KEYS:
            if source.get(key) is not None:
                facts[key] = source[key]
    elif isinstance(source, str) and source:
        facts["source"] = source
    if frame_file:
        for key in FRAME_KEYS:
            if facts.get(key) is None and frame_file.get(key) is not None:
                facts[key] = frame_file[key]
        facts["source"] = facts.get("source") or "_frame.json"
    for key in ("window", "window_note"):
        if facts.get(key) is None and doc.get(key) is not None:
            facts[key] = doc[key]
    return facts


def sample_size(doc: dict, facts: dict | None = None) -> int | None:
    """How many postings each arm was asked about, from the best source available."""
    facts = facts or frame_facts(doc, None)
    for key in ("limit", "records_sent"):
        value = facts.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    frame = doc.get("frame")
    if isinstance(frame, dict):
        body = frame.get(FRAUD)
        if isinstance(body, dict) and isinstance(body.get("majority_set_size"), int):
            return body["majority_set_size"]
    counts = [
        entry.get("scored_records")
        for entry in (doc.get("per_question") or {}).values()
        if isinstance(entry, dict) and isinstance(entry.get("scored_records"), int)
    ]
    if counts:
        return max(counts)
    window = facts.get("window", doc.get("window"))
    return int(window) if isinstance(window, (int, float)) else None


def sampling_prose(facts: dict) -> str:
    """How the frame was drawn, in words, or an empty string when unrecorded."""
    sampling = facts.get("sampling")
    seed = facts.get("seed")
    if isinstance(sampling, str) and sampling:
        if isinstance(seed, (int, float)):
            return f"drawn with seed {int(seed)} as a {sampling}"
        return f"drawn as a {sampling}"
    if isinstance(seed, (int, float)):
        return f"drawn with seed {int(seed)} as a random sample"
    return "drawn as a sample of the split"


def hero_from_metrics(doc: dict) -> dict | None:
    hero = doc.get("hero")
    if isinstance(hero, dict) and hero.get("engines"):
        return hero
    return None


def load_result_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        if path.name == "cache.jsonl":
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "record_id" in row and "question" in row:
                rows.append(row)
    return rows


def hero_from_results(results_dir: Path, doc: dict) -> dict:
    """Derive a hero from the arm result rows when metrics.json carries none.

    Rule: among the records whose gold is the fraud positive where at least two
    arms returned a label and at least two of those labels disagree, the lowest
    record id wins. The posting text comes from the dataset loader and is capped
    at EXCERPT_CHARS characters.
    """
    rows = load_result_rows(results_dir)
    if not rows:
        raise BuildError(
            f"no hero block in metrics.json and no arm result rows under {results_dir} to "
            "derive one from"
        )
    by_record: dict[str, dict[str, dict]] = {}
    golds: dict[str, str] = {}
    for row in rows:
        if row.get("question") != FRAUD or not row.get("value"):
            continue
        record_id = str(row["record_id"])
        arm = str(row.get("arm") or "")
        if not arm:
            continue
        by_record.setdefault(record_id, {})[arm] = row
        if row.get("gold"):
            golds.setdefault(record_id, str(row["gold"]))
    candidates = [
        record_id
        for record_id, arms in by_record.items()
        if golds.get(record_id) == "fraudulent"
        and len(arms) >= 2
        and len({str(r.get("value")) for r in arms.values()}) >= 2
    ]
    candidates.sort(key=record_sort_key)
    if not candidates:
        raise BuildError(
            f"no record under {results_dir} has a fraudulent gold label with two disagreeing "
            "arms, so no hero could be derived"
        )
    record_id = candidates[0]
    arms = by_record[record_id]
    order = [arm for arm in arm_order(doc) if arm in arms] + sorted(set(arms) - set(arm_order(doc)))
    title, excerpt = posting_excerpt(record_id)
    return {
        "record_id": record_id,
        "title": title,
        "state_excerpt": excerpt,
        "gold": golds[record_id],
        "engines": [
            {
                "arm": arm,
                "value": arms[arm].get("value"),
                "probabilities": arms[arm].get("probabilities"),
                "confidence": arms[arm].get("confidence"),
                "correct": arms[arm].get("correct", arms[arm].get("value") == golds[record_id]),
            }
            for arm in order
        ],
    }


def import_triage_data():
    """Import ``triage.data`` with the repo root on the path.

    The generator is run as ``uv run python web/build_site.py``, so the script's
    own directory is what Python puts first on ``sys.path`` and the package next
    to it is not importable without this.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from triage import data

    return data


def posting_excerpt(record_id: str) -> tuple[str, str]:
    try:
        data = import_triage_data()
    except Exception as exc:  # pragma: no cover - depends on the caller's environment
        raise BuildError(
            f"hero {record_id} needs the posting text from triage.data, which could not be "
            f"imported: {type(exc).__name__}: {exc}"
        ) from exc
    split = record_id.rpartition("-")[0] or record_id
    try:
        records = {row["record_id"]: row["state"] for row in data.records(split)}
    except Exception as exc:
        raise BuildError(
            f"hero {record_id}: triage.data could not load split {split!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    state = records.get(record_id)
    if not state:
        raise BuildError(f"hero {record_id}: split {split!r} has no such record")
    head, _, tail = state.partition("\n\n")
    title = head.strip().splitlines()[0].strip() if head.strip() else record_id
    body = tail.strip() or head.strip()
    return title, body[:EXCERPT_CHARS]


def decision_probability(engine: dict) -> float | None:
    for key in ("p_decision", "confidence"):
        value = engine.get(key)
        if value is None:
            continue
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return None
    probabilities = engine.get("probabilities")
    if isinstance(probabilities, dict):
        value = engine.get("value")
        if value in probabilities:
            try:
                return min(1.0, max(0.0, float(probabilities[value])))
            except (TypeError, ValueError):
                return None
    return None


def hero_parts(hero: dict) -> tuple[str, str]:
    """Title and body of the hero posting.

    metrics.json may carry a title next to the excerpt, or the excerpt may open
    with the title line the state itself starts with. Both shapes are handled
    without dropping a character of the excerpt.
    """
    excerpt = str(hero.get("state_excerpt") or hero.get("state") or "")
    title = hero.get("title")
    if title:
        return str(title), excerpt
    head, sep, tail = excerpt.partition("\n\n")
    if sep and head.strip():
        return head.strip(), tail.strip()
    first, _, rest = excerpt.partition("\n")
    if rest.strip():
        return first.strip(), rest.strip()
    return str(hero.get("record_id") or "(unnamed posting)"), excerpt


def render_hero(hero: dict, known_arms: list[str]) -> str:
    title, excerpt = hero_parts(hero)
    gold = hero.get("gold")
    engines = [e for e in hero.get("engines") or [] if isinstance(e, dict)]
    known = [e for e in engines if str(e.get("arm")) in known_arms] or engines
    unknown = [e for e in engines if e not in known]
    wrong = [e for e in known if e.get("correct") is False]

    rows: list[str] = []
    for engine in known:
        arm = str(engine.get("arm") or "unknown_arm")
        value = str(engine.get("value") or "")
        correct = engine.get("correct")
        if correct is True:
            kind, mark = "right", "right"
        elif correct is False:
            kind, mark = "wrong", "wrong"
        else:
            kind, mark = "plain", "not scored"
        if value == "fraudulent":
            chip_class = "flag"
        elif value == "legitimate":
            chip_class = "clear"
        else:
            chip_class = "plain"
        chip = (
            f'<span class="chip chip-{chip_class}">{esc(value)}</span>'
            if value
            else '<span class="chip chip-plain">no answer</span>'
        )
        probability = decision_probability(engine)
        if probability is None:
            figure = (
                '<span class="na" title="the arm reported no probability for this posting">'
                "n/a</span>"
            )
            bar = '<div class="bar"></div>'
        else:
            figure = f'<span class="figure">{probability:.2f}</span>'
            bar = (
                f'<div class="bar"><span class="fill fill-{kind}" '
                f'style="width:{probability * 100:.1f}%"></span></div>'
            )
        rows.append(
            '<li class="verdict"><div class="verdict-head">'
            f'<span class="engine">{esc(arm)}</span>{chip}{figure}'
            f'<span class="mark mark-{kind}">{esc(mark)}</span></div>{bar}</li>'
        )
    if wrong:
        gold_line = (
            f'<p class="gold">Gold label: <strong>{esc(gold)}</strong>. '
            f"{len(wrong)} of {len(known)} engines disagreed with it.</p>"
        )
    else:
        gold_line = (
            f'<p class="gold">Gold label: <strong>{esc(gold)}</strong>. '
            "Every engine agreed with it.</p>"
        )
    excluded = list(hero.get("engines_excluded") or [])
    if unknown or excluded:
        names = ", ".join(
            [str(e.get("arm")) for e in unknown] + [str(name) for name in excluded]
        )
        extra = f'<p class="caption">No answer on this posting from: {esc(names)}.</p>'
    else:
        extra = ""
    record_id = hero.get("record_id")
    shown = hero.get("state_excerpt_chars")
    shown_text = (
        f"first {count(shown)} characters of the text the engines read"
        if isinstance(shown, (int, float))
        else "opening of the text the engines read"
    )
    if record_id:
        source = (
            f'<p class="caption">Posting {esc(record_id)} from the public dataset, showing the '
            f"{esc(shown_text)}. Their input is the title plus the description, cut at 6,000 "
            "characters. The text is verbatim, so a link in the source appears as a #URL_ "
            "placeholder and runs of whitespace stay as they were.</p>"
        )
    else:
        source = ""
    return f"""
      <div class="hero-grid">
        <article class="posting">
          <h2 class="posting-title">{esc(title)}</h2>
          <p class="posting-body">{esc(excerpt)}</p>
          {source}
        </article>
        <div class="verdicts">
          <h2 class="sub">What each engine said about it</h2>
          <p class="legend">Bar length is the probability the engine gave its own answer. Teal
          marks a call that matches the gold label, crimson marks a call that does not.</p>
          <ol class="verdict-list">{"".join(rows)}</ol>
          {gold_line}
          {extra}
        </div>
      </div>"""


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def render_engines(doc: dict) -> str:
    reasons = NAReasons(doc.get("n_a"))
    listed = arm_order(doc)
    unavailable = unavailable_map(doc)
    extra = [str(arm) for arm in unavailable if str(arm) not in listed]
    entries = []
    for arm in listed:
        status = unavailable.get(arm) or {}
        note = ""
        if status and str(status.get("status") or "ok") != "ok":
            label = str(status.get("status"))
            reason = status.get("reason") or reasons(arm, None, "rows")
            note = (
                f'<span class="chip chip-plain">{esc(label)}</span>'
                f' <span class="why">{esc(reason)}</span>'
            )
        entries.append(
            '<div class="engine-entry">'
            f'<dt><code class="arm-id">{esc(arm)}</code>{note}</dt>'
            f"<dd>{esc(ARM_INFO.get(arm, GENERIC_ARM_INFO))}</dd>"
            "</div>"
        )
    for arm in extra:
        status = unavailable.get(arm) or {}
        reason = status.get("reason") or reasons(arm, None, "rows")
        entries.append(
            '<div class="engine-entry">'
            f'<dt><code class="arm-id">{esc(arm)}</code>'
            '<span class="chip chip-plain">not run</span></dt>'
            f"<dd>{esc(ARM_INFO.get(arm, GENERIC_ARM_INFO))} "
            f'<span class="why">{esc(reason)}</span></dd>'
            "</div>"
        )
    families: list[str] = []
    for arm in listed:
        family = ARM_FAMILY.get(arm, arm)
        if family not in families:
            families.append(family)
    lead = ""
    if listed:
        names = ", ".join(ENGINE_LABEL.get(family, family) for family in families)
        extra = len(listed) - len(families)
        tail = ""
        if extra:
            noun = "arm" if extra == 1 else "arms"
            verb = "varies" if extra == 1 else "vary"
            tail = f", and the other {extra} {noun} {verb} one of them rather than adding an engine"
        engine_word = "engine" if len(families) == 1 else "engines"
        lead = (
            f'<p class="legend">{len(listed)} arms ran on this sample, and they come from '
            f"{len(families)} {engine_word}: {esc(names)}{esc(tail)}.</p>"
        )
    return f'{lead}<dl class="engine-list">{"".join(entries)}</dl>'


def fraud_block(doc: dict) -> dict:
    return doc.get("per_question", {}).get(FRAUD) or {}


def fraud_metrics(doc: dict, arm: str) -> dict | None:
    block = fraud_block(doc)
    for group in ("rows", "baselines"):
        entry = (block.get(group) or {}).get(arm)
        if isinstance(entry, dict):
            return entry
    return None


def positive_pair(entry: dict) -> tuple[Any, Any]:
    precision = entry.get("precision_positive")
    recall = entry.get("recall_positive")
    positive = entry.get("positive_label")
    per_class = entry.get("per_class") or {}
    if positive in per_class:
        precision = precision if precision is not None else per_class[positive].get("precision")
        recall = recall if recall is not None else per_class[positive].get("recall")
    return precision, recall


def not_in_table(doc: dict, rendered: list[str]) -> str:
    reasons = NAReasons(doc.get("n_a"))
    unavailable = unavailable_map(doc)
    listed = arm_order(doc)
    missing = sorted(
        {str(arm) for arm in unavailable if str(arm) not in rendered}
        | {arm for arm in listed if arm not in rendered}
    )
    if not missing:
        return ""
    items = "".join(
        f'<li><code class="arm-id">{esc(arm)}</code>: '
        f"{esc((unavailable.get(arm) or {}).get('reason') or f'no row was written for it on the {FRAUD} question')}</li>"
        for arm in missing
    )
    return f'<ul class="notes"><li class="notes-head">Not in this table</li>{items}</ul>'


def render_per_class(doc: dict) -> str:
    """The disclosure body: per-class precision and recall behind the fraud table."""
    reasons = NAReasons(doc.get("n_a"))
    block = fraud_block(doc)
    labels = [str(label) for label in block.get("labels") or []]
    if not labels:
        return ""
    rows: list[str] = []
    notes: list[tuple[str, str, str]] = []
    for group in ("rows", "baselines"):
        for arm, entry in (block.get(group) or {}).items():
            if not isinstance(entry, dict):
                continue
            per_class = entry.get("per_class")
            kind = "floor" if group == "baselines" else "plain"
            cells = []
            for label in labels:
                stats = (per_class or {}).get(label) or {}
                for metric in ("precision", "recall"):
                    value = stats.get(metric)
                    if value is None:
                        reason = reasons(str(arm), FRAUD, f"per_class.{label}.{metric}")
                        notes.append((f"{arm} / {label}", metric, reason))
                        cells.append(na_cell(reason))
                    else:
                        cells.append(f'<td class="num">{share(value)}</td>')
            chip = "floor" if group == "baselines" else ""
            rows.append(
                f'<tr class="row-{kind}">'
                + q_cell(str(arm), chip, "floor" if chip else "plain", kind)
                + f'<td class="num">{count(entry.get("n")) if entry.get("n") is not None else "n/a"}</td>'
                + "".join(cells)
                + "</tr>"
            )
    head = "".join(
        f'<th scope="col">{esc(f"{label} {metric}")}</th>' for label in labels for metric in ("precision", "recall")
    )
    notes_html = ""
    if notes:
        seen: set[tuple[str, str, str]] = set()
        items = []
        for arm, metric, reason in notes:
            if (arm, metric, reason) in seen:
                continue
            seen.add((arm, metric, reason))
            items.append(f"<li><strong>{esc(arm)}</strong>, {esc(metric)}: {esc(reason)}</li>")
        notes_html = (
            '<ul class="notes"><li class="notes-head">Why a cell reads n/a</li>'
            + "".join(items)
            + "</ul>"
        )
    return (
        f'<div class="table-wrap"><table><thead><tr><th scope="col" class="text">arm</th>'
        f'<th scope="col">n</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f'<p class="scroll-hint">The table scrolls sideways on a narrow screen.</p>{notes_html}'
    )


def render_fraud(doc: dict) -> str:
    reasons = NAReasons(doc.get("n_a"))
    block = fraud_block(doc)
    rows_by_arm = block.get("rows") or {}
    baselines = block.get("baselines") or {}
    floor_entry = baselines.get("majority_class") or {}
    floor = floor_entry.get("accuracy")

    table = Table(
        reasons,
        [
            ("arm", "text"),
            ("n", "num"),
            ("accuracy", "num"),
            ("macro F1", "num"),
            ("precision", "num"),
            ("recall", "num"),
            ("F1", "num"),
            ("PR-AUC", "num"),
            ("invalid JSON", "num"),
            ("p50 latency", "num"),
        ],
    )
    rendered: list[str] = []
    degenerate_names = {arm for arm, _ in degenerate_arms(doc)}
    specs = [(arm, rows_by_arm, "rows") for arm in arm_order(doc)]
    specs += [(name, baselines, "baselines") for name in ("majority_class", "dummy_stratified")]
    for name, group, group_name in specs:
        entry = group.get(name)
        if not isinstance(entry, dict):
            continue
        rendered.append(name)
        precision, recall = positive_pair(entry)
        accuracy = entry.get("accuracy")
        is_floor = group_name == "baselines"
        if is_floor:
            kind = "floor"
        elif accuracy is not None and isinstance(floor, (int, float)):
            kind = "right" if accuracy >= floor else "wrong"
        else:
            kind = "plain"
        chip_text = (
            "floor"
            if is_floor
            else "no spread"
            if name in degenerate_names
            else "fitted"
            if name in FITTED_ARMS
            else ""
        )
        chip_class = "floor" if is_floor else "plain"
        table.row(
            [
                q_cell(name, chip_text, chip_class, kind),
                table.cell(entry.get("n"), count, arm=name, question=FRAUD, metric="n"),
                table.cell(accuracy, share, arm=name, question=FRAUD, metric="accuracy"),
                table.cell(entry.get("macro_f1"), share, arm=name, question=FRAUD, metric="macro_f1"),
                table.ci_cell(
                    precision,
                    entry.get("precision_ci95"),
                    share,
                    arm=name,
                    question=FRAUD,
                    metric="precision_positive",
                    ci_metric="precision_ci95",
                ),
                table.ci_cell(
                    recall,
                    entry.get("recall_ci95"),
                    share,
                    arm=name,
                    question=FRAUD,
                    metric="recall_positive",
                    ci_metric="recall_ci95",
                ),
                table.ci_cell(
                    entry.get("f1_positive"),
                    entry.get("f1_ci95"),
                    share,
                    arm=name,
                    question=FRAUD,
                    metric="f1_positive",
                    ci_metric="f1_ci95",
                ),
                table.cell(entry.get("pr_auc"), score, arm=name, question=FRAUD, metric="pr_auc"),
                table.cell(
                    entry.get("invalid_json_rate"),
                    share,
                    arm=name,
                    question=FRAUD,
                    metric="invalid_json_rate",
                ),
                table.cell(
                    entry.get("latency_p50"), millis, arm=name, question=FRAUD, metric="latency_p50"
                ),
            ],
            kind,
        )

    floor_text = share(floor) if isinstance(floor, (int, float)) else "majority"
    ci_note = ""
    for entry in list(rows_by_arm.values()) + list(baselines.values()):
        if isinstance(entry, dict) and entry.get("ci_note"):
            ci_note = str(entry["ci_note"])
            break
    caption = (
        '<caption class="table-caption">The two floor rows carry a neutral rule and a floor chip. '
        f'A 3px rule in teal marks an arm that clears the {esc(floor_text)} majority floor on this '
        "question, crimson marks one that does not. Precision, recall and F1 belong to the "
        "fraudulent class, the rare one, and the number under each of them is its 95% interval"
        + (f" (the file records the interval frame as: {esc(ci_note)})" if ci_note else "")
        + ".</caption>"
    )
    scored = block.get("scored_records")
    gold = (block.get("gold_distribution") or {}).get(FRAUD)
    distribution = ""
    if isinstance(scored, (int, float)) and isinstance(gold, (int, float)):
        distribution = (
            f"{count(gold)} of {count(scored)} scored postings carry the fraudulent label, "
            f"which is {share(gold / scored)}."
        )
    detail = render_per_class(doc)
    disclosure = (
        '<details class="disclosure"><summary>Per-class precision and recall</summary>'
        f'<div class="details-body">{detail}</div></details>'
        if detail
        else ""
    )
    return f"""
      {table.render(caption)}
      {not_in_table(doc, rendered)}
      <p class="legend">Precision, recall, PR-AUC and the invalid-JSON rate all belong to the
      <strong>fraudulent</strong> class, the rare one. {esc(distribution)} p50 latency is the median
      answer time per posting for this question.</p>
      {interval_prose(doc)}
      {render_degenerate(doc)}
      {disclosure}"""


def frame_note(doc: dict) -> str:
    """The frame-consistency check, driven by metrics.json's ``frame`` block.

    ``mismatch`` true means one or more arms were scored on a different set of
    postings than the rest, so their rows are not directly comparable.
    """
    frame = doc.get("frame")
    if not isinstance(frame, dict) or not frame:
        return ""
    mismatched = [
        (question, body)
        for question, body in frame.items()
        if isinstance(body, dict) and body.get("mismatch")
    ]
    if mismatched:
        parts = []
        for question, body in mismatched:
            arms = ", ".join(str(arm) for arm in body.get("mismatched_arms") or []) or "an arm"
            against = body.get("majority_arm") or "the other arms"
            extra = body.get("only_in_arm") or {}
            missing = body.get("missing_from_arm") or {}
            detail = []
            for arm in body.get("mismatched_arms") or []:
                if isinstance(extra.get(arm), int) and isinstance(missing.get(arm), int):
                    detail.append(f"{arm} answered {count(extra[arm])} postings the others did not and skipped {count(missing[arm])} of theirs")
            detail_text = f" ({'; '.join(detail)})" if detail else ""
            parts.append(f"{question}: {arms} against {against}{detail_text}")
        return (
            '<p class="guard">Frame check: on '
            + esc("; ".join(parts))
            + ". Those arms were scored on a different set of postings, so their rows here are "
            "not directly comparable with the rest.</p>"
        )
    return (
        '<p class="guard">Frame check: every arm answered the same set of postings on every '
        "question, so the rows in the tables above are directly comparable.</p>"
    )


def degenerate_arms(doc: dict) -> list[tuple[str, list[str]]]:
    """Arms whose scores carry no spread: PR-AUC at the base rate, or ECE >= 0.5."""
    block = fraud_block(doc)
    flagged: list[tuple[str, list[str]]] = []
    for arm, entry in (block.get("rows") or {}).items():
        if not isinstance(entry, dict):
            continue
        reasons: list[str] = []
        n = entry.get("n")
        positives = entry.get("positives")
        pr_auc = entry.get("pr_auc")
        if (
            isinstance(pr_auc, (int, float))
            and isinstance(n, (int, float))
            and n
            and isinstance(positives, (int, float))
        ):
            base_rate = positives / n
            if abs(pr_auc - base_rate) <= 0.0005:
                reasons.append(
                    f"its PR-AUC equals the base rate ({pr_auc:.3f}): the scores never separate "
                    "the two classes"
                )
        ece = entry.get("ece")
        if isinstance(ece, (int, float)) and ece >= 0.5:
            reasons.append(f"its ECE is {ece:.3f}, so its confidence carries no information")
        if reasons:
            flagged.append((str(arm), reasons))
    return flagged


def render_degenerate(doc: dict) -> str:
    flagged = degenerate_arms(doc)
    if not flagged:
        return ""
    items = "".join(
        f"<li><strong>{esc(arm)}</strong>: {'; '.join(esc(reason) for reason in reasons)}.</li>"
        for arm, reasons in flagged
    )
    return (
        '<h3 class="sub-head">Arms with no spread</h3>'
        '<p class="legend">These arms answer nearly the same thing about every posting, so each '
        "row is a single operating point. The numbers come from the same rows as the table "
        "above.</p>"
        f'<ul class="notes">{items}</ul>'
    )


def interval_prose(doc: dict) -> str:
    """What the intervals do and do not separate, read from the file."""
    block = fraud_block(doc)
    rows = block.get("rows") or {}
    scored = [
        (str(arm), entry["f1_ci95"], entry.get("accuracy"))
        for arm, entry in rows.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("f1_ci95"), list)
        and len(entry["f1_ci95"]) == 2
        and all(isinstance(value, (int, float)) for value in entry["f1_ci95"])
    ]
    if len(scored) < 2:
        return ""
    scored.sort(key=lambda item: -item[1][0])
    top, top_ci, _ = scored[0]
    others = scored[1:]
    best_upper = max(bounds[1] for _, bounds, _ in others)
    if top_ci[0] > best_upper:
        first = (
            f"On F1, {top}'s interval {interval(top_ci)} starts above the highest upper bound of "
            f"every other arm ({best_upper:.2f}), so that ordering does not depend on where the "
            "threshold sits."
        )
    else:
        first = (
            f"On F1, the highest interval is {top}'s {interval(top_ci)} and it overlaps the others, "
            "so this sample does not separate them."
        )
    floor = ((block.get("baselines") or {}).get("majority_class") or {}).get("accuracy")
    names = ", ".join(arm for arm, _, _ in others)
    accuracies = [acc for _, _, acc in others if isinstance(acc, (int, float))]
    shared = min(bounds[1] for _, bounds, _ in others) >= max(bounds[0] for _, bounds, _ in others)
    if others and accuracies:
        overlap_text = (
            "all overlap one another" if shared else "overlap in part"
        )
        floor_text = (
            f", against a {share(floor)} majority-class floor on accuracy"
            if isinstance(floor, (int, float))
            else ""
        )
        second = (
            f"The other arms' intervals {overlap_text} ({esc(names)}), and their accuracy runs "
            f"from {share(min(accuracies))} to {share(max(accuracies))}{floor_text}."
        )
    else:
        second = ""
    ci_note = ""
    for entry in rows.values():
        if isinstance(entry, dict) and entry.get("ci_note"):
            ci_note = str(entry["ci_note"])
            break
    note = (
        f" The file records the interval frame as: {esc(ci_note)}." if ci_note else ""
    )
    return f'<p class="legend">{first} {second}{note}</p>'


def render_ablation(doc: dict) -> str:
    ablation = doc.get("ablation")
    if not isinstance(ablation, dict):
        return ""
    block = ablation.get(FRAUD)
    if not isinstance(block, dict):
        return ""
    reasons = NAReasons(doc.get("n_a"))
    table = Table(
        reasons,
        [
            ("request variant", "text"),
            ("accuracy", "num"),
            ("macro F1", "num"),
            ("precision (fraudulent)", "num"),
            ("recall (fraudulent)", "num"),
            ("PR-AUC", "num"),
        ],
    )
    rows = 0
    for arm, label in (("classifier_dev", "with the rubric"), ("classifier_dev_no_rubric", "without the rubric")):
        entry = block.get(arm)
        if not isinstance(entry, dict):
            continue
        rows += 1
        precision, recall = positive_pair(entry)
        table.row(
            [
                f'<td class="arm"><span class="arm-name">{esc(arm)}</span>'
                f'<span class="chip chip-plain">{esc(label)}</span></td>',
                table.cell(entry.get("accuracy"), share, arm=arm, question=FRAUD, metric="accuracy"),
                table.cell(entry.get("macro_f1"), share, arm=arm, question=FRAUD, metric="macro_f1"),
                table.cell(precision, share, arm=arm, question=FRAUD, metric="precision_positive"),
                table.cell(recall, share, arm=arm, question=FRAUD, metric="recall_positive"),
                table.cell(entry.get("pr_auc"), score, arm=arm, question=FRAUD, metric="pr_auc"),
            ]
        )
    if not rows:
        return ""
    return f"""
      <h3 class="sub-head">The rubric ablation</h3>
      <p class="legend">One field differs between these rows: the <code>instructions</code> text
      carrying the per-question rubric. The model, the postings and the labels are the same.</p>
      {table.render()}"""


def render_guard(doc: dict) -> str:
    guard = doc.get("leakage_guard")
    if not isinstance(guard, dict) or not guard:
        return ""
    failures: list[str] = []
    below: list[str] = []
    floor_text = ""
    for arm, entry in guard.items():
        if not isinstance(entry, dict):
            continue
        if floor_text == "" and isinstance(entry.get("floor"), (int, float)):
            floor_text = share(entry["floor"])
        delta = entry.get("delta_pp")
        delta_text = (
            f"{delta:+.1f} points" if isinstance(delta, (int, float)) else "no delta recorded"
        )
        if entry.get("passed") is False:
            failures.append(f"{arm} sits {delta_text} above the floor")
        elif str(entry.get("verdict") or "").upper() == "BELOW":
            below.append(f"{arm} sits {delta_text}")
    asymmetry = (
        "The guard is one-directional: only an accuracy above the floor plus 3 points fails, "
        "because only that direction can mean the gold label leaked into the posting text."
    )
    if failures:
        text = (
            f"Leakage guard: {len(failures)} arm(s) leave the band around the {esc(floor_text)} "
            f"majority floor on the salary control question ({esc('; '.join(failures))}), so their "
            f"fraud numbers should be read with that in mind. {asymmetry}"
        )
    else:
        text = (
            f"Leakage guard: no arm rises more than 3 points above the {esc(floor_text)} majority "
            "floor on the salary control question, whose label lives in the dataset and not in the "
            f"posting text. {asymmetry}"
        )
    if below:
        text += (
            f" Under the floor: {esc('; '.join(below))}, which reads as an arm that is not "
            "answering the question that was asked rather than as a leak."
        )
    return f'<p class="guard">{text}</p>'


def calibration_ranking(doc: dict) -> tuple[list[str], list[str]]:
    """The two lowest-ECE arms and the two highest-ECE arms on the fraud question."""
    block = fraud_block(doc)
    scored = [
        (str(arm), entry["ece"])
        for arm, entry in (block.get("rows") or {}).items()
        if isinstance(entry, dict) and isinstance(entry.get("ece"), (int, float))
    ]
    scored.sort(key=lambda pair: pair[1])
    best = [arm for arm, _ in scored[:2]]
    worst = [arm for arm, _ in reversed(scored[-2:])]
    return best, [arm for arm in worst if arm not in best]


def render_calibration(doc: dict) -> str:
    reasons = NAReasons(doc.get("n_a"))
    block = fraud_block(doc)
    rows_by_arm = block.get("rows") or {}
    table = Table(
        reasons,
        [("arm", "text"), ("n", "num"), ("Brier", "num"), ("ECE", "num"), ("excluded", "num")],
    )
    if not rows_by_arm:
        return '<p class="legend">No arm produced a stated probability on this question, so there is nothing to calibrate.</p>'
    for arm in arm_order(doc):
        entry = rows_by_arm.get(arm)
        if not isinstance(entry, dict):
            continue
        ece = entry.get("ece")
        kind = "plain"
        if isinstance(ece, (int, float)):
            kind = "right" if ece <= 0.05 else "wrong"
        table.row(
            [
                q_cell(arm, "", "plain", kind),
                table.cell(
                    entry.get("calibration_n"), count, arm=arm, question=FRAUD, metric="calibration_n"
                ),
                table.cell(entry.get("brier"), score, arm=arm, question=FRAUD, metric="brier"),
                table.cell(ece, score, arm=arm, question=FRAUD, metric="ece"),
                table.cell(
                    entry.get("calibration_excluded"),
                    count,
                    arm=arm,
                    question=FRAUD,
                    metric="calibration_excluded",
                ),
            ],
            kind,
        )
    best, worst = calibration_ranking(doc)
    picked = best + worst
    bins = doc.get("calibration_bins") if isinstance(doc.get("calibration_bins"), dict) else {}
    bin_table = Table(
        reasons,
        [
            ("bin of stated probability", "text"),
            ("n", "num"),
            ("mean stated probability", "num"),
            ("accuracy in the bin", "num"),
        ],
    )
    empty_bins = 0
    for arm in picked:
        entries = ((bins.get(arm) or {}).get(FRAUD)) or []
        if not entries:
            bin_table.notes.append((arm, "calibration bins", reasons(arm, FRAUD, "calibration_bin")))
            continue
        # An empty bin holds no answers, so there is nothing to report for it.
        filled = [entry for entry in entries if entry.get("n")]
        empty_bins += len(entries) - len(filled)
        if not filled:
            bin_table.notes.append((arm, "calibration bins", reasons(arm, FRAUD, "calibration_bin")))
            continue
        bin_table.rows.append(
            '<tr class="row-group"><th scope="rowgroup" colspan="4">'
            f'<code class="arm-id">{esc(arm)}</code></th></tr>'
        )
        for entry in filled:
            low, high = entry.get("bin") or (None, None)
            mean_p = entry.get("mean_p")
            accuracy = entry.get("accuracy")
            kind = "plain"
            if isinstance(mean_p, (int, float)) and isinstance(accuracy, (int, float)):
                kind = "right" if accuracy >= mean_p else "wrong"
            label = (
                f"{low:.1f} to {high:.1f}"
                if isinstance(low, (int, float)) and isinstance(high, (int, float))
                else "n/a"
            )
            bin_table.row(
                [
                    f'<td class="text">{esc(label)}</td>',
                    bin_table.cell(
                        entry.get("n"), count, arm=arm, question=FRAUD, metric="calibration_bin"
                    ),
                    bin_table.cell(
                        mean_p, score, arm=arm, question=FRAUD, metric="calibration_bin"
                    ),
                    bin_table.cell(
                        accuracy, score, arm=arm, question=FRAUD, metric="calibration_bin"
                    ),
                ],
                kind,
            )
    empty_note = (
        f" {count(empty_bins)} bins across these arms held no answer at all and are left out."
        if empty_bins
        else ""
    )
    if picked:
        picks = (
            f"<p class=\"legend\">The bins below cover {esc(', '.join(best) or 'no arm')}, "
            f"the two arms with the lowest ECE on this question, and "
            f"{esc(', '.join(worst) or 'no arm')}, the two with the highest. These are the ends of "
            "the ranking: the arms whose stated probabilities land closest to what they get "
            f"right, and furthest from it.{esc(empty_note)}</p>"
        )
    else:
        picks = (
            '<p class="legend">No arm carries an ECE on this question, so there is no ranking to '
            f"pick from: {esc(reasons(None, FRAUD, 'ece'))}</p>"
        )
    figure = (
        '<figure class="figure-block">'
        '<img src="calibration.png" alt="Stated probability against accuracy per arm on the fraudulent question" '
        'width="540" height="288">'
        "<figcaption>Calibration on the fraud question, drawn by the evaluator from the same "
        "records these tables come from. A well calibrated arm sits on the diagonal.</figcaption>"
        "</figure>"
    )
    return f"""
      <p class="legend">ECE sorts every answered posting into ten bins of stated probability and
      averages the gap between the probability an arm claimed in a bin and the share of those
      answers it got right. 0.000 means the stated number can be read as a probability; 0.100
      means the arm is off by ten points on average.</p>
      {table.render(
        '<caption class="table-caption">A 3px rule in teal marks an ECE of 0.05 or less, where the '
        "stated probability is within five points of the outcome on average; crimson marks an arm "
        "above that. A Brier score is the mean squared error of the same probabilities, so lower "
        'is better in both columns.</caption>'
      )}
      {figure}
      {picks}
      {bin_table.render()}"""


def render_checks(doc: dict) -> str:
    """The checks that decide whether the tables above can be trusted.

    Each one answers a question a reader should ask before believing a number: what
    the rubric text bought, whether every answer was usable, whether every arm saw the
    same postings, and whether the control question leaked.
    """
    return f"""
      {render_ablation(doc)}
      {render_invalid(doc)}
      {frame_note(doc)}
      {render_guard(doc)}"""


def render_invalid(doc: dict) -> str:
    """How many answers were unusable per arm, and why, from metrics.json."""
    reasons = NAReasons(doc.get("n_a"))
    block = fraud_block(doc)
    rows = block.get("rows") or {}
    invalid_rows = doc.get("invalid_rows")
    invalid_reasons = doc.get("invalid_reasons")
    items: list[str] = []
    clean: list[str] = []
    for arm in arm_order(doc):
        entry = rows.get(arm)
        if not isinstance(entry, dict):
            continue
        rate = entry.get("invalid_json_rate")
        recorded = None
        if isinstance(invalid_rows, dict):
            recorded = invalid_rows.get(arm)
            if recorded is None and isinstance(invalid_rows.get(FRAUD), dict):
                recorded = invalid_rows[FRAUD].get(arm)
        if isinstance(invalid_reasons, dict) and invalid_reasons.get(arm):
            why = str(invalid_reasons[arm])
        elif isinstance(rate, (int, float)) and rate > 0:
            why = reasons(arm, FRAUD, "invalid_json_rate")
        else:
            why = reasons(arm, FRAUD, "invalid_reasons")
        count_text = (
            f"{count(recorded)} of {count(entry.get('n'))} answers"
            if isinstance(recorded, (int, float)) and isinstance(entry.get("n"), (int, float))
            else (
                f"{share(rate)} of answers"
                if isinstance(rate, (int, float))
                else "no rate recorded"
            )
        )
        if isinstance(rate, (int, float)) and rate == 0 and not (isinstance(recorded, (int, float)) and recorded):
            clean.append(arm)
            continue
        items.append(f"<li><strong>{esc(arm)}</strong>: {esc(count_text)} unusable. {esc(why)}</li>")
    if clean:
        listed = ", ".join(esc(arm) for arm in clean)
        items.append(
            f"<li><strong>{listed}</strong>: no unusable answer at all ({esc(reasons(clean[0], FRAUD, 'invalid_reasons'))}).</li>"
        )
    if not items:
        return ""
    return (
        '<h3 class="sub-head">Unusable answers</h3>'
        '<p class="legend">An answer is unusable when the arm failed its own output validation. '
        "Those rows stay in the run and are scored wrong, so they cost accuracy rather than "
        "disappearing from it.</p>"
        f'<ul class="notes">{"".join(items)}</ul>'
    )


#: A service-backed arm can return slightly different probabilities for the same
#: posting on a later fetch, so a threshold cell can move by a record between
#: reruns. The page says so, using the acted counts the sweep records.
JITTER_SHARE = 0.01


def routing_jitter(doc: dict) -> str:
    routing = doc.get("routing") if isinstance(doc.get("routing"), dict) else {}
    taus = [
        entry.get("tau")
        for sweep in routing.values()
        for entry in sweep or []
        if isinstance(entry, dict) and isinstance(entry.get("tau"), (int, float))
    ]
    if not taus:
        return ""
    top = max(taus)
    few: list[tuple[str, int]] = []
    scored: int | None = None
    for arm, sweep in routing.items():
        for entry in sweep or []:
            if not isinstance(entry, dict) or entry.get("tau") != top:
                continue
            acted = entry.get("acted")
            n = entry.get("scored")
            if isinstance(n, (int, float)) and n:
                scored = int(n) if scored is None else scored
            if not isinstance(acted, (int, float)) or not isinstance(n, (int, float)) or not n:
                continue
            if isinstance(acted, (int, float)) and 0 < acted < JITTER_SHARE * n:
                few.append((str(arm), int(acted)))
    few.sort(key=lambda pair: (pair[1], pair[0]))
    scored_text = count(scored) if scored else "the scored"
    tau_text = f"{top:.2f}"
    if few:
        listed = ", ".join(f"{esc(arm)} {count(acted)}" for arm, acted in few)
        arms_word = "arm" if len(few) == 1 else "arms"
        counts = (
            f"where few postings carry the decision: at tau {tau_text}, {len(few)} {arms_word} act "
            f"on fewer than 1% of the {esc(scored_text)} scored postings ({listed})"
        )
    else:
        counts = (
            f"where few postings carry the decision: no arm acts on fewer than 1% of the "
            f"{esc(scored_text)} scored postings at tau {tau_text}"
        )
    return (
        '<p class="guard">classifier.dev is a shared service and its probabilities shift slightly '
        "between fetches of the same posting, so a threshold-dependent cell in this table can move "
        "by one record between reruns while the label-based tables in Fraud triage do not. Treat "
        f"the routing cells as indicative at high thresholds, {counts}.</p>"
    )


def render_routing(doc: dict) -> str:
    reasons = NAReasons(doc.get("n_a"))
    routing = doc.get("routing") if isinstance(doc.get("routing"), dict) else {}
    # Arms in the order the rest of the page uses, then any baseline or extra sweep.
    order = [arm for arm in arm_order(doc) if arm in routing]
    order += [str(arm) for arm in routing if str(arm) not in order]
    rows: list[str] = []
    notes: list[tuple[str, str, str]] = []
    unreached = 0
    for arm in order:
        sweep = routing.get(arm)
        entries = [e for e in sweep or [] if isinstance(e, dict)]
        if not entries:
            continue
        entries.sort(
            key=lambda entry: entry.get("tau") if isinstance(entry.get("tau"), (int, float)) else 0
        )
        rows.append(
            '<tr class="row-group"><th scope="rowgroup" colspan="6">'
            f'<code class="arm-id">{esc(arm)}</code></th></tr>'
        )
        for entry in entries:
            tau = entry.get("tau")
            precision = entry.get("precision")
            row_note = entry.get("note")
            # A tau no score reaches is its own state: nothing is acted on, so
            # precision has no value. It is not a failed threshold.
            if precision is None:
                kind = "plain"
                unreached += 1
            else:
                kind = "right" if precision >= 0.9 else "wrong"
            cells = [
                f'<td class="num">{esc(f"{tau:.2f}") if isinstance(tau, (int, float)) else "n/a"}</td>'
            ]
            for metric, fmt in (
                ("coverage", share),
                ("precision", share),
                ("recall", share),
                ("false_negatives", count),
            ):
                value = entry.get(metric)
                if value is None:
                    reason = (
                        str(row_note)
                        if metric == "precision" and row_note
                        else reasons(str(arm), FRAUD, metric)
                    )
                    notes.append((str(arm), metric, reason))
                    cells.append(na_cell(reason))
                else:
                    cells.append(f'<td class="num">{fmt(value)}</td>')
            acted_on = entry.get("positives_acted_on")
            total = entry.get("positives_total")
            if acted_on is None or total is None:
                reason = reasons(str(arm), FRAUD, "positives_acted_on")
                notes.append((str(arm), "positives acted on", reason))
                cells.append(na_cell(reason))
            else:
                cells.append(f'<td class="num">{count(acted_on)} of {count(total)}</td>')
            rows.append(f'<tr class="row-{kind}">{"".join(cells)}</tr>')
    if rows:
        head = "".join(
            f'<th scope="col">{esc(title)}</th>'
            for title in (
                "tau",
                "coverage",
                "precision",
                "recall",
                "false negatives among acted on",
                "positives acted on",
            )
        )
        notes_html = ""
        if notes:
            seen: set[tuple[str, str, str]] = set()
            items = []
            for arm, metric, reason in notes:
                if (arm, metric, reason) in seen:
                    continue
                seen.add((arm, metric, reason))
                items.append(f"<li><strong>{esc(arm)}</strong>, {esc(metric)}: {esc(reason)}</li>")
            notes_html = (
                '<ul class="notes"><li class="notes-head">Why a cell reads n/a</li>'
                + "".join(items)
                + "</ul>"
            )
        sweep_html = (
            '<div class="table-wrap"><table><caption class="table-caption">Read a row as: at '
            "stated probability tau or above, this share of postings is acted on, and this is how "
            "the acted-on set scores. The 3px rule marks a tau whose precision clears 0.90 in "
            "teal and one that does not in crimson. tau is a sequence, so the rows are ordered "
            "rather than numbered.</caption>"
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="scroll-hint">The table scrolls sideways on a narrow screen.</p>'
            f"{notes_html}"
        )
    else:
        sweep_html = '<p class="legend">No routing sweep was written into metrics.json.</p>'
    unreached_note = (
        f"{count(unreached)} of these rows sit at a tau that no score in the arm reaches, so "
        "nothing is acted on: coverage is 0.0% and precision has no value."
        if unreached
        else ""
    )

    cost = doc.get("cost") if isinstance(doc.get("cost"), dict) else {}
    cost_rows: list[str] = []
    for arm in arm_order(doc):
        body = cost.get(arm)
        if not isinstance(body, dict):
            continue
        entry = fraud_metrics(doc, arm) or {}
        cost_rows.append(
            '<tr class="row-plain">'
            f'<td class="arm"><span class="arm-name">{esc(arm)}</span></td>'
            + _num_cell(entry.get("latency_p50"), millis, arm, reasons, "latency_p50")
            + _num_cell(entry.get("latency_p95"), millis, arm, reasons, "latency_p95")
            + _num_cell(entry.get("tokens_in_mean"), tokens, arm, reasons, "tokens_in_mean")
            + _num_cell(entry.get("tokens_out_mean"), tokens, arm, reasons, "tokens_out_mean")
            + _num_cell(body.get("classifications"), count, arm, reasons, "classifications")
            # One decimal, so a sub-second wall clock reads as 0.1 s rather than 0 s.
            + _num_cell(body.get("seconds"), lambda value: f"{value:,.1f} s", arm, reasons, "seconds")
            + "</tr>"
        )
    if cost_rows:
        cost_head = "".join(
            f'<th scope="col">{esc(title)}</th>'
            for title in (
                "arm",
                "p50 per posting",
                "p95 per posting",
                "mean tokens in",
                "mean tokens out",
                "classifications spent",
                "wall seconds",
            )
        )
        cost_html = (
            '<div class="table-wrap"><table><caption class="table-caption">Latency and tokens are '
            "measured on the fraud question. A classification is counted per answer, so one "
            "posting on one question is one classification.</caption>"
            f'<thead><tr>{cost_head}</tr></thead><tbody>{"".join(cost_rows)}</tbody></table></div>'
            '<p class="scroll-hint">The table scrolls sideways on a narrow screen.</p>'
        )
        flat = []
        for arm in arm_order(doc):
            entry = fraud_metrics(doc, arm) or {}
            p50, p95 = entry.get("latency_p50"), entry.get("latency_p95")
            if isinstance(p50, (int, float)) and p50 == p95:
                flat.append(arm)
        if flat:
            names = ", ".join(esc(arm) for arm in flat)
            cost_html += (
                f'<p class="legend">For {names}, p50 equals p95 in this run, so those two columns '
                "describe one measured value rather than a spread.</p>"
            )
    else:
        cost_html = ""
    figure = (
        '<figure class="figure-block">'
        '<img src="routing.png" alt="Precision and recall of the acted-on set as tau rises" '
        'width="540" height="288">'
        "<figcaption>Precision and recall of the acted-on set as tau rises, drawn by the "
        "evaluator. Each point is one tau from the table above.</figcaption></figure>"
    )
    return f"""
      {sweep_html}
      {routing_jitter(doc)}
      <p class="legend">False negatives are the fraudulent postings a tau leaves in the unreviewed
      pile, so they are what higher precision costs.
      {esc(unreached_note)}</p>
      {cost_html}
      {render_cost_note(doc)}
      {render_projection(doc)}
      {figure}"""


def _num_cell(
    value: Any, fmt: Callable[[Any], str], arm: str, reasons: NAReasons, metric: str
) -> str:
    if value is None:
        return na_cell(reasons(arm, FRAUD, metric))
    return f'<td class="num">{fmt(value)}</td>'


#: Used when metrics.json carries no cost_note. The latencies in the cost table
#: were measured while several arms ran at once, so they describe the machine
#: under load rather than the models on their own.
COST_NOTE_FALLBACK = (
    "These latencies were measured while several arms ran at once, so they describe this loaded "
    "machine rather than the models on their own: treat the projection below as an upper bound, "
    "not as a benchmark result."
)


def render_cost_note(doc: dict) -> str:
    note = doc.get("cost_note")
    text = note.strip() if isinstance(note, str) and note.strip() else COST_NOTE_FALLBACK
    return f'<p class="guard">{esc(text)}</p>'


def render_projection(doc: dict) -> str:
    cost = doc.get("cost") if isinstance(doc.get("cost"), dict) else {}
    items = []
    for arm, body in cost.items():
        if not isinstance(body, dict):
            continue
        classifications = body.get("classifications")
        records = body.get("records")
        if not isinstance(classifications, (int, float)) or not isinstance(records, (int, float)):
            continue
        if not records:
            continue
        per_posting = classifications / records
        items.append(
            f"<li><strong>{esc(arm)}</strong> spent {count(classifications)} classifications on "
            f"{count(records)} postings, which is {per_posting:.2f} per posting; at 10,000 postings "
            f"a day that is {count(per_posting * 10_000)} classifications a day.</li>"
        )
    if not items:
        return ""
    return (
        '<h3 class="sub-head">Projection at 10,000 postings a day</h3>'
        f'<ul class="notes projection">{"".join(items)}</ul>'
    )


def render_limits(doc: dict, facts: dict) -> str:
    block = fraud_block(doc)
    split = facts.get("split") or doc.get("split")
    sample = sample_size(doc, facts)
    per_question = doc.get("per_question") or {}
    counts = []
    for question in ("fraudulent", "required_experience", "required_education", "salary_range_stated"):
        entry = per_question.get(question)
        if not isinstance(entry, dict):
            continue
        n = entry.get("scored_records")
        if isinstance(n, (int, float)) and n:
            counts.append(f"{count(n)} for {question}")
    counts_text = ", ".join(counts) if counts else "no per-question counts recorded"
    positives = (block.get("gold_distribution") or {}).get(FRAUD)
    positive_text = count(positives) if isinstance(positives, (int, float)) else "an unrecorded number of"
    sample_text = count(sample) if sample else "an unrecorded number of"
    frame = doc.get("frame") if isinstance(doc.get("frame"), dict) else {}
    mismatched = [
        question
        for question, body in frame.items()
        if isinstance(body, dict) and body.get("mismatch")
    ]
    if mismatched:
        frame_text = (
            " One or more arms were scored on a different set of postings on "
            f"{esc(', '.join(sorted(mismatched)))}, so those rows are not directly comparable and "
            "the frame check under the fraud table names them."
        )
    else:
        frame_text = ""
    items = (
        "<li>The data is the public Kaggle fake job postings mirror, which dates from 2014 to "
        "2018. Both language models may have seen it while they were pretrained, so their numbers "
        "here are an upper bound rather than a clean measurement.</li>"
        f"<li>This run scores {esc(sample_text)} postings of the whole {esc(split)} split, "
        f"{esc(sampling_prose(facts))}, so it is not a prefix of the split. Each question is scored on the rows "
        f"that carry a gold answer ({esc(counts_text)}), so the rare-class numbers rest on the "
        f"{esc(positive_text)} fraudulent postings that a draw of this size holds, which is what "
        f"makes the intervals in the fraud table as wide as they are.{frame_text}</li>"
        "<li><code>required_experience</code> and <code>required_education</code> are platform "
        "tags a poster filled in, so agreeing with them measures tag recovery.</li>"
        "<li>classifier.dev is a shared free service with a daily quota, so rerunning the Jev arm "
        "can shift its numbers.</li>"
        "<li>The LinkedIn demo set is unlabelled, so its numbers are flag rates and agreement "
        "between arms, never accuracy.</li>"
        "<li>Precision and recall here count a flagged posting as a hit against a platform tag. "
        "Nothing in this run says whether a human reviewer would agree with the flag.</li>"
    )
    return f'<ul class="limits">{items}</ul>'


def render_footer(doc: dict, stamp: str, facts: dict) -> str:
    # The command is meant to be pasted into a shell, so the limit keeps its
    # digits and takes no thousands separator. Both it and the seed come from the
    # frame the runner recorded, so the line reproduces the run that produced
    # these results rather than a guess.
    sample = sample_size(doc, facts)
    limit = sample if sample else 1000
    seed = facts.get("seed")
    seed_text = str(int(seed)) if isinstance(seed, (int, float)) else "<seed>"
    commands = [line.format(limit=limit, seed=seed_text) for line in REPRODUCE]
    seed_note = (
        ""
        if isinstance(seed, (int, float))
        else (
            '<p class="legend">These results do not record the seed, so the run line leaves it as '
            "a placeholder rather than stating a number that is not in the file.</p>"
        )
    )
    stamp_text = f", {esc(stamp)}" if stamp else ""
    return f"""
      <h2 class="sub">Reproduce it</h2>
      <pre class="commands"><code>{esc(chr(10).join(commands))}</code></pre>
      <p class="legend">{CACHE_NOTE}</p>
      {seed_note}
      <p class="legend">This page is generated by <code>web/build_site.py</code> from
      <code>results/metrics.json</code>{stamp_text}. The full tables, including every n/a with its
      reason, are in <a href="report.md">report.md</a>.</p>"""

# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def render_rail() -> str:
    items = "".join(
        f'<li><a href="#{anchor}" data-rail="{anchor}">{esc(name)}</a></li>'
        for anchor, name in SECTIONS
    )
    return f'<nav class="rail" aria-label="Sections"><ol>{items}</ol></nav>'


def render_measurement(doc: dict, facts: dict) -> str:
    per_question = doc.get("per_question") or {}
    block = per_question.get(FRAUD) or {}
    counts = []
    for question in ("fraudulent", "required_experience", "required_education", "salary_range_stated"):
        entry = per_question.get(question)
        if not isinstance(entry, dict):
            continue
        n = entry.get("scored_records")
        if isinstance(n, (int, float)) and n:
            counts.append(f"{count(n)} for {question}")
    counts_text = ", ".join(counts) if counts else "no per-question counts were recorded"
    gold = block.get("gold_distribution") or {}
    positives = gold.get(FRAUD)
    positive_text = (
        f"The fraud question holds {count(positives)} positive postings in this sample."
        if isinstance(positives, (int, float))
        else "The fraud question carries no positive count in this window."
    )
    frame = doc.get("frame") if isinstance(doc.get("frame"), dict) else {}
    fraud_frame = frame.get(FRAUD) if isinstance(frame.get(FRAUD), dict) else {}
    sample = sample_size(doc, facts)
    sample_text = count(sample) if sample else "an unrecorded number of"
    split = facts.get("split") or doc.get("split")
    mismatch_text = ""
    if fraud_frame.get("mismatch"):
        arms = ", ".join(str(arm) for arm in fraud_frame.get("mismatched_arms") or [])
        mismatch_text = (
            f" The arms together cover {count(doc.get('records_sent'))} record ids because {esc(arms)} "
            "was scored on a different set of postings; the frame check under Method checks names it."
        )
    rows = block.get("rows") or {}
    baselines = block.get("baselines") or {}
    floor = (baselines.get("majority_class") or {}).get("accuracy")
    named = [
        (name, entry)
        for name, entry in list(rows.items()) + list(baselines.items())
        if isinstance(entry, dict)
    ]

    def top(key: str) -> tuple[float | None, str | None]:
        pairs = [(e[key], name) for name, e in named if isinstance(e.get(key), (int, float))]
        return max(pairs, default=(None, None))

    best_f1, best_f1_arm = top("f1_positive")
    best_pr, best_pr_arm = top("pr_auc")

    def who(arm: str | None) -> str:
        return f"{arm} (fitted)" if arm in FITTED_ARMS else str(arm)

    scored = block.get("scored_records")
    base = (
        positives / scored
        if isinstance(scored, (int, float)) and scored and isinstance(positives, (int, float))
        else None
    )
    flat = degenerate_arms(doc)
    spread_label = "arm with no spread" if len(flat) == 1 else "arms with no spread"
    stats = [
        (share(floor) if isinstance(floor, (int, float)) else "n/a", "majority-class floor"),
        (score(best_f1) if best_f1 is not None else "n/a", f"best F1, {who(best_f1_arm)}"),
        (
            score(best_pr) if best_pr is not None else "n/a",
            f"best PR-AUC, {who(best_pr_arm)}"
            + (f", base rate {score(base)}" if base is not None else ""),
        ),
        (count(len(flat)), spread_label),
    ]
    stats_html = "".join(
        f"<li><b>{esc(value)}</b><span>{esc(label)}</span></li>" for value, label in stats
    )
    return f"""
      <h1>Job posting triage: four engines, one labelled test split</h1>
      <p class="lede">Four engines answered the same four questions about the same postings: each
      saw {esc(sample_text)} postings from the {esc(split)} split of the public fake job postings
      dataset, {esc(sampling_prose(facts))}. A question is scored on the rows that carry a gold
      answer: {esc(counts_text)}. Every answer, its latency, its confidence and its cost were
      recorded. {esc(positive_text)} {esc(mismatch_text.strip())}</p>
      <p class="links">
        <a href="{esc(REPO_URL)}">The repository</a>
        <a href="report.md">The full report</a>
      </p>
      <ul class="stats">{stats_html}</ul>"""


def build_page(doc: dict, hero: dict | None, stamp: str, facts: dict) -> str:
    listed = tuple(anchor for anchor, _ in SECTIONS)
    if listed != RENDERED_SECTIONS:
        raise BuildError(
            "SECTIONS and the section markup disagree: the rail lists "
            f"{listed}, the page writes {RENDERED_SECTIONS}"
        )
    hero_html = render_hero(hero, arm_order(doc)) if hero else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job posting triage, four engines, one labelled test split</title>
<meta name="description" content="Measured accuracy, calibration and cost for four engines triaging job postings for fraud on a labelled test split.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700&amp;family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&amp;display=swap">
<link rel="stylesheet" href="styles.css">
</head>
<body>
{render_rail()}
<header class="strip" id="measurement">
  <div class="wrap">
{render_measurement(doc, facts)}
  </div>
</header>
<section class="strip" id="fraud">
  <div class="wrap">
    <h2>Fraud triage</h2>
{render_fraud(doc)}
  </div>
</section>
<section class="strip hero" id="example" aria-label="One posting, four verdicts">
  <div class="wrap">
    <h2>One posting, four verdicts</h2>
{hero_html}
  </div>
</section>
<section class="strip" id="engines">
  <div class="wrap">
    <h2>The four engines</h2>
{render_engines(doc)}
  </div>
</section>
<section class="strip" id="calibration">
  <div class="wrap">
    <h2>Calibration</h2>
{render_calibration(doc)}
  </div>
</section>
<section class="strip" id="routing">
  <div class="wrap">
    <h2>Routing and cost</h2>
{render_routing(doc)}
  </div>
</section>
<section class="strip" id="checks">
  <div class="wrap">
    <h2>Method checks</h2>
{render_checks(doc)}
  </div>
</section>
<section class="strip" id="limits">
  <div class="wrap">
    <h2>What this does not prove</h2>
{render_limits(doc, facts)}
  </div>
</section>
<footer class="strip" id="reproduce">
  <div class="wrap">
{render_footer(doc, stamp, facts)}
  </div>
</footer>
<script>
(function () {{
  var rail = document.querySelector(".rail");
  if (!rail || !("IntersectionObserver" in window)) return;
  var links = Array.prototype.slice.call(rail.querySelectorAll("a"));
  var sections = links.map(function (link) {{
    return document.querySelector(link.getAttribute("href"));
  }}).filter(Boolean);
  if (!sections.length) return;
  var mark = function (id) {{
    links.forEach(function (link) {{
      if (link.getAttribute("href") === "#" + id) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }});
  }};
  var observer = new IntersectionObserver(function (entries) {{
    entries.forEach(function (entry) {{ if (entry.isIntersecting) mark(entry.target.id); }});
  }}, {{ rootMargin: "-20% 0px -65% 0px" }});
  sections.forEach(function (section) {{ observer.observe(section); }});
}})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the job-posting-triage showcase site")
    parser.add_argument("--results", default="results", help="directory holding metrics.json and the assets")
    parser.add_argument("--out", default="docs", help="directory to write the site into")
    parser.add_argument("--stamp", default="", help="free-text note appended to the footer, for example a date")
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    try:
        doc = load_metrics(results_dir)
        require_files(results_dir)
        hero = hero_from_metrics(doc)
        if hero is None:
            hero = hero_from_results(results_dir, doc)
        if not STYLESHEET.exists():
            raise BuildError(f"missing stylesheet {STYLESHEET}")
        facts = frame_facts(doc, load_frame_file(results_dir))
    except BuildError as exc:
        print(f"build_site: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print(f"build_site: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(
        build_page(doc, hero, args.stamp, facts), encoding="utf-8"
    )
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(STYLESHEET, out_dir / "styles.css")
    for name in ("calibration.png", "routing.png", "report.md"):
        shutil.copyfile(results_dir / name, out_dir / name)
    print(
        f"build_site: wrote {out_dir}/index.html and {out_dir}/styles.css; "
        f"hero {hero.get('record_id')} from {'metrics.json' if 'hero' in doc and hero_from_metrics(doc) else 'result rows'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
