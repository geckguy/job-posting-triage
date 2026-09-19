"""Zero-shot encoder arm: GLiNER2.5-base as a schema-conditioned classifier.

This is the "no training, no prompt, one pass per question" arm. Each question is
compiled into its own schema; the encoder reads ``[schema tokens] [SEP_TEXT] [text
tokens]`` in a single forward pass per record and question, and the classification
head scores every ``[L] label`` marker against the pooled document embedding.
Nothing is generated, so there is no parser and no invalid-JSON failure mode: the
cost model is one encoder pass per record per question and the output is a
probability per label.

Three properties of this arm are stated where the code lives, because all three
are choices rather than accidents:

* **The rubric text is deliberately NOT sent to this arm.** The other arms get
  ``question.instructions``; this one gets only the label set, which is what
  conditions a zero-shot encoder. Measured on the three handcrafted fixtures in
  ``scripts/llm_mode_probe.py``, P(fraudulent) for the obvious scam, the ordinary
  engineering posting and a recipe that is not a job ad at all is 0.9655 / 0.0264
  / 0.0007 with labels only and one task per schema. That same single-task schema
  tolerates the rubric (0.9999 / 0.0069 / 0.0056), but in a four-task schema the
  rubric scores the recipe ``fraudulent`` at 0.998, and the rubric is only ever
  one more block of prompt text competing with the posting for the encoder's
  attention. Labels only, one task per schema, gets all three right.
* **One task per schema.** The interference is measurable and monotone in the
  number of pooled label sets, and it is reproducible with
  ``scripts/gliner_pooling_probe.py``: P(fraudulent) for the scam / the ordinary
  engineering posting / the recipe is 0.9655 / 0.0264 / 0.0007 with one task,
  0.6914 / 0.3369 / 0.0132 with two, 0.1900 / 0.7797 / 0.0183 with three and
  0.1114 / 0.7702 / 0.0466 with all four. An independent run of the same probe
  on this checkpoint reproduced the shape (0.9977 / 0.0679, 0.9351 / 0.7502,
  0.2542 / 0.9650, 0.1409 / 0.9398). From three tasks up the ranking inverts:
  the honest engineering posting looks more fraudulent than the scam. Batch size
  is not the variable (batches of 1 and 3 agree to four decimals), so the arm
  answers each question from its own schema: one encoder pass per question per
  record, ``meta["passes_per_record"]`` = 1 with ``meta["schema_tasks"]`` = 1,
  four passes for a posting that needs all four questions. That is the price of a
  fraud score that reads the posting, and the report's cost section states it
  rather than claiming the single pooled pass the brief assumed.
* **The per-label score map comes from ``gliner2.classification``.** The
  ``extract`` / ``classify_text`` surface computes the full softmax and then
  keeps only the argmax probability, so calibration is not measurable through
  it. The ``Classifier`` facade used here returns the whole
  ``{label: probability}`` map from the same single pass. Those numbers are
  already normalized (softmax, because every question here is single-label);
  normalizing them again would be a bug.

The 512-subword encoder window is a hard, silent limit. The checkpoint reports
``max_position_embeddings: 512`` and ``position_biased_input: false``, so an
over-long input does not raise, it degrades. The arm therefore measures each
schema's overhead in subwords at load time, truncates each posting at a word
boundary so that ``overhead + text <= 512``, and records ``meta["truncated"]``,
``meta["budget_subwords"]`` and ``meta["n_subwords"]`` per row, so the report can
quantify how much of the benchmark ran on a partial posting. With one task per
schema the measured overheads are 11 subwords for ``fraudulent``, 29 for
``required_experience``, 25 for ``required_education`` and 17 for
``salary_range_stated``, which leaves 483 to 501 subwords of posting text, half
again more than the pooled four-task schema allowed.

The checkpoint is loaded through ``gliner2.models.base.load_extractor_tokenizer``
(see :func:`GlinerArm._load_tokenizer`): no dependency pin and no edit to
``models/`` is needed for the tokenizer-config incompatibility that a plain
``AutoTokenizer.from_pretrained`` hits.
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from triage.arms.base import Answer, Arm, ArmUnavailable, Record
from triage.schema import Question

#: The checkpoint this benchmark pins: GLiNER2.5-base, 194M params, boundary
#: architecture, DeBERTa-v2-base encoder. Downloaded into ``models/`` (gitignored).
CHECKPOINT = Path("models/gliner2.5-base-v1")

#: The encoder's positional window, in subwords, *including* the schema segment
#: the library packs in front of the posting text.
ENCODER_WINDOW = 512

#: Markers the library relies on for logit-to-label alignment. Each must be a
#: single added token in the checkpoint's tokenizer: if one of them tokenizes
#: into subwords, the label scores are silently misaligned, so the arm refuses to
#: run rather than produce plausible-looking numbers.
MARKERS = ("[P]", "[L]", "[C]", "[E]", "[R]", "[SEP_STRUCT]", "[SEP_TEXT]")

#: Records per encoder pass. Measured on this machine over 64 sampled postings
#: (one fraud pass each), ms per record-question: batch 8 = 637 on MPS and 838 on
#: CPU, batch 16 = 855 on MPS, batch 32 = 917 on MPS and 977 on CPU, batch 64 =
#: 981 on MPS. Smaller is faster, because the library's decode syncs the device
#: once per label, so a bigger batch only makes each wait longer; batch 8 also
#: held peak RSS near 2.9 GB against 5.8 GB at batch 32. Do not raise it without
#: re-measuring.
DEFAULT_BATCH_SIZE = 8

#: Device preference, best first, for a clonee's machine.
DEVICE_PREFERENCE = ("cuda", "mps", "cpu")

#: Set to ``cpu`` or ``cuda`` to override :data:`DEVICE_PREFERENCE` without
#: touching code. Used to keep this arm off a GPU that the LLM arm needs: CPU is
#: only 1.3x slower per record than MPS here (838 against 637 ms), so running the
#: two arms on different compute units beats co-tenanting one Metal GPU.
DEVICE_ENV = "TRIAGE_GLINER_DEVICE"


@dataclass(frozen=True)
class _View:
    """One schema, its measured subword overhead, and the text budget left."""

    questions: tuple[Question, ...]
    schema: Any
    model_schema: dict
    overhead_subwords: int
    budget_subwords: int


class GlinerArm(Arm):
    name = "gliner"

    def __init__(
        self,
        checkpoint: str | Path = CHECKPOINT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.batch_size = batch_size
        self.device = device
        self.usage: dict[str, Any] = {}
        self._clf: Any = None
        self._tok: Any = None
        self._tok_fn: Any = None
        self._splitter: Any = None
        self._processor: Any = None
        self._marker_ids: dict[str, int] = {}
        self._load_note = ""
        self._views: dict[tuple[str, ...], _View] = {}

    # -- availability ------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        """Cheap pre-flight: is the library importable and the checkpoint here?

        The tokenizer and the weights are only read by :meth:`run`, which raises
        :class:`ArmUnavailable` carrying the exact exception text if the load
        fails.
        """
        try:
            import gliner2  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - any import failure is an answer
            return False, f"gliner2 unavailable: {type(exc).__name__}: {exc}"
        missing = [
            name
            for name in ("config.json", "model.safetensors")
            if not (self.checkpoint / name).is_file()
        ]
        if missing:
            return False, (
                f"checkpoint incomplete at {self.checkpoint}: missing {', '.join(missing)}"
            )
        return True, ""

    # -- loading -----------------------------------------------------------

    @staticmethod
    def _load_tokenizer(path: Path) -> tuple[Any, str]:
        """Load the checkpoint tokenizer despite its legacy marker metadata.

        ``transformers`` reads ``tokenizer_config.json:extra_special_tokens`` as a
        name to token mapping and expands it into the tokenizer's
        ``SPECIAL_TOKENS_ATTRIBUTES``. This checkpoint serialized the ten GLiNER2
        markers as a plain *list*, so the plain load dies with
        ``AttributeError: 'list' object has no attribute 'keys'`` before the
        markers are registered.

        ``gliner2`` 2.0.0 ships ``load_extractor_tokenizer`` for exactly this
        case: it retries with ``extra_special_tokens={}``. The retry preserves
        marker semantics here, because the markers are already *added tokens* of
        the checkpoint's own ``tokenizer.json`` (ids 128001 to 128010), so nothing
        is added, merged or renumbered by dropping that field. It only ever fed a
        saving path. Using the library's own normalization means no dependency
        pin, no rewrite of the downloaded checkpoint, and the same behaviour for
        anyone who downloads it fresh. Older ``gliner2`` builds without that
        helper get the identical retry inline.

        Returns the tokenizer and any warning text the library emitted.
        """
        from transformers import AutoTokenizer

        try:
            from gliner2.models.base import load_extractor_tokenizer
        except ImportError:  # pragma: no cover - gliner2 2.0.0 always has it
            load_extractor_tokenizer = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if load_extractor_tokenizer is not None:
                tok = load_extractor_tokenizer(str(path))
            else:
                try:
                    tok = AutoTokenizer.from_pretrained(str(path))
                except AttributeError as exc:
                    if "'list' object has no attribute 'keys'" not in str(exc):
                        raise
                    tok = AutoTokenizer.from_pretrained(
                        str(path), extra_special_tokens={}
                    )
        return tok, "; ".join(str(w.message) for w in caught)

    def _verify_markers(self, tokenizer: Any) -> dict[str, int]:
        """Every marker must be exactly one token, matching the checkpoint's own."""
        ids: dict[str, int] = {}
        for marker in MARKERS:
            encoded = tokenizer.encode(marker, add_special_tokens=False)
            if len(encoded) != 1:
                raise ArmUnavailable(
                    f"marker {marker!r} in {self.checkpoint} encodes to "
                    f"{len(encoded)} subwords {encoded} instead of one token, so the "
                    "marker layout is not intact and label scores cannot be trusted"
                )
            ids[marker] = int(encoded[0])

        added = {str(key): int(value) for key, value in tokenizer.get_added_vocab().items()}
        mismatched = {
            marker: (ids[marker], added.get(marker))
            for marker in MARKERS
            if str(added.get(marker)) != str(ids[marker])
        }
        if mismatched:
            raise ArmUnavailable(
                "marker ids do not match the checkpoint's added-token table "
                f"(marker: encoded vs added-token): {mismatched}"
            )
        return ids

    def _pick_device(self) -> str:
        """The device the encoder runs on.

        Preference order is ``cuda``, then ``mps``, then ``cpu``, so a clonee gets
        whatever the machine has. ``TRIAGE_GLINER_DEVICE`` overrides it: this arm
        shares a Metal GPU with the benchmark's LLM arm, where two clients on one
        GPU cost both of them about 3x, and different compute units are worth more
        than co-tenanting. The choice lands in ``usage["device"]`` and in every
        row's ``meta["device"]``, so a run can never be mistaken for one measured
        on another device.
        """
        import torch

        if self.device:
            return self.device
        override = os.environ.get(DEVICE_ENV, "").strip()
        if override:
            return override
        for candidate in DEVICE_PREFERENCE:
            if candidate == "cuda" and torch.cuda.is_available():
                return "cuda"
            if candidate == "mps" and torch.backends.mps.is_available():
                return "mps"
            if candidate == "cpu":
                return "cpu"
        return "cpu"

    def _ensure_loaded(self) -> None:
        """Load the tokenizer and the weights once, and move them to the device."""
        if self._clf is not None:
            return
        if not self.checkpoint.is_dir():
            raise ArmUnavailable(f"checkpoint not found at {self.checkpoint}")
        try:
            from gliner2.classification import Classifier
        except Exception as exc:  # noqa: BLE001
            raise ArmUnavailable(
                f"gliner2.classification unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        tokenizer, note = self._load_tokenizer(self.checkpoint)
        self._marker_ids = self._verify_markers(tokenizer)

        device = self._pick_device()
        try:
            # ``Classifier.from_pretrained(device=...)`` only records the device,
            # the weights stay where they were loaded, so the move is explicit.
            clf = Classifier.from_pretrained(str(self.checkpoint))
            clf.to(device=device)
        except Exception as exc:  # noqa: BLE001 - missing head, bad weights, no MPS
            raise ArmUnavailable(
                f"GLiNER2 classification head could not be loaded from "
                f"{self.checkpoint} ({type(exc).__name__}): {exc}"
            ) from exc

        self._clf = clf
        self._tok = tokenizer
        self._processor = clf.model.processor
        self._splitter = getattr(self._processor, "word_splitter", None)
        if self._splitter is None:  # pragma: no cover - processor always has one
            from gliner2.processing.word_splitter import resolve_word_splitter

            self._splitter = resolve_word_splitter("whitespace")
        # Share the processor's tokenize LRU, so the subword accounting below is
        # not paid for twice (the collator re-tokenizes every word it encodes).
        self._tok_fn = getattr(self._processor, "_tokenize_cached", tokenizer.tokenize)
        self._load_note = note
        self.usage["device"] = device
        self.usage["batch_size"] = self.batch_size
        self.usage["marker_ids"] = dict(self._marker_ids)
        self.usage["load_note"] = note

    # -- schema and budget -------------------------------------------------

    def _prepare(self, questions: tuple[Question, ...]) -> _View:
        """Build (and cache) the schema, then measure its subword overhead."""
        key = tuple(question.id for question in questions)
        cached = self._views.get(key)
        if cached is not None:
            return cached

        from gliner2.classification import ClassificationSchema

        self._check_prefix_collisions(key)
        schema = ClassificationSchema()
        for question in questions:
            # Labels only: no ``instruction=``. See the module docstring for the
            # measured effect of the rubric on this checkpoint.
            schema.single(question.id, list(question.labels))
        compiled = self._clf.compile_schema(schema)
        model_schema = copy.deepcopy(compiled.build())

        # Overhead is measured, not assumed: encode a probe whose own subword
        # count is known and subtract it.
        overhead = self._encoded_len(model_schema, "") - self._text_subwords("")
        if overhead < 0:
            raise ArmUnavailable(
                f"schema overhead measured as {overhead} subwords; the collator and the "
                "subword accounting disagree, so truncating on that basis is unsafe"
            )
        budget = ENCODER_WINDOW - overhead
        if budget < 1:
            raise ArmUnavailable(
                f"the schema for {key} costs {overhead} subwords of the "
                f"{ENCODER_WINDOW}-subword window, leaving no room for posting text"
            )
        missing = [question.id for question in questions if question.id not in compiled.task_order]
        if missing:
            raise ArmUnavailable(
                f"compiled schema is missing task(s) {missing}; it has "
                f"{list(compiled.task_order)}"
            )

        view = _View(
            questions=questions,
            schema=schema,
            model_schema=model_schema,
            overhead_subwords=overhead,
            budget_subwords=budget,
        )
        self._views[key] = view
        self.usage.setdefault("overheads", {})["+".join(key)] = {
            "overhead_subwords": overhead,
            "budget_subwords": budget,
        }
        return view

    @staticmethod
    def _check_prefix_collisions(task_ids: Iterable[str]) -> None:
        """Task ids must not be boundary-prefixes of one another.

        The library reads the encoded prompt back with a boundary-aware longest
        match to know which task an ``[L]`` block belongs to, so a task named
        ``fraud`` would steal the scores of ``fraudulent``. ``compile_schema``
        rejects the same shape; stating the rule here keeps it next to the ids it
        constrains.
        """
        ids = list(task_ids)
        for x in ids:
            for y in ids:
                if x != y and y.startswith(x) and len(y) > len(x) and y[len(x)] in (":", " "):
                    raise ValueError(
                        f"task id {x!r} is a boundary-prefix of {y!r}: the prompt "
                        "resolver could confuse them, so one of them must be renamed"
                    )

    # -- subword accounting ------------------------------------------------

    def _text_subwords(self, text: str) -> int:
        """Subwords the processor will encode for ``text``, exactly.

        The processor appends ``.`` to a posting that does not end in
        punctuation, splits on whitespace, lower-cases each word and tokenizes
        the words one by one into a flat subword list; this reproduces that
        count. Checked against the real ``input_ids`` length (minus the measured
        schema overhead) for all 3,182 test postings, including the longest one.
        """
        if text and not text.endswith((".", "!", "?")):
            text = text + "."
        elif not text:
            text = "."
        return sum(
            len(self._tok_fn(word)) for word, _start, _end in self._splitter(text, lower=True)
        )

    def _encoded_len(self, model_schema: dict, text: str) -> int:
        """The real ``input_ids`` length, measured through the library's collator."""
        batch = self._processor.collate_fn_inference(
            [(text, copy.deepcopy(model_schema))], max_len=None
        )
        return int(batch.input_ids.shape[1])

    def _fit(self, text: str, budget: int) -> tuple[str, int, bool]:
        """Cut ``text`` at a word boundary so its subwords fit ``budget``.

        Cutting the original string at a word boundary is lossless: no subword is
        split, and no decode round-trip can introduce characters the posting did
        not contain.
        """
        text = text.strip()
        whole = self._text_subwords(text)
        if whole <= budget:
            return text, whole, False

        kept_end = 0
        used = 0
        for word, _start, end in self._splitter(text, lower=True):
            cost = len(self._tok_fn(word))
            if cost == 0:  # pragma: no cover - the splitter yields non-empty words
                continue
            # The processor appends "." unless the kept text already ends in
            # punctuation, so leave room for it.
            penalty = 0 if text[end - 1] in ".!?" else 1
            if used + cost + penalty > budget:
                break
            used += cost
            kept_end = end

        if kept_end == 0:
            # A single leading word is longer than the whole budget (a pasted URL,
            # say). Fall back to subword truncation of that word, decoded back to
            # text, and shrink until the re-encoded count fits.
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
            keep = max(budget - 1, 1)  # leave one subword for the appended "."
            cut = self._tok.decode(ids[:keep]).strip()
            while keep > 1 and self._text_subwords(cut) > budget:
                keep -= 1
                cut = self._tok.decode(ids[:keep]).strip()
            return cut, self._text_subwords(cut), True

        cut = text[:kept_end]
        return cut, self._text_subwords(cut), True

    # -- inference ---------------------------------------------------------

    def _config(self, batch_size: int) -> Any:
        from gliner2.classification import ClassificationConfig

        return ClassificationConfig(
            batch_size=batch_size, max_len=None, include_confidence=True
        )

    def _answer(
        self,
        question: Question,
        result: Any,
        meta: dict,
        view: _View,
        n_subwords: int,
        latency_ms: float,
        was_truncated: bool,
    ) -> Answer:
        if question.id not in result.tasks:
            raise ArmUnavailable(
                f"GLiNER2 returned no answer for task {question.id!r}; it answered "
                f"{sorted(result.tasks)}"
            )
        reported = result.probabilities(question.id)
        # Keyed by the question's canonical labels, and zero-filled for any label
        # the library did not report, so every arm's dict shape matches.
        scores = {label: float(reported.get(label, 0.0)) for label in question.labels}
        library_value = result.value(question.id)
        value = library_value if library_value in scores else max(
            scores, key=lambda label: (scores[label], label)
        )
        return Answer(
            value=value,
            probabilities=scores,
            confidence=scores[value],
            raw={"value": library_value, "confidence": result.confidence(question.id)},
            invalid=library_value != value,
            # The subwords this record actually cost the encoder: the schema
            # segment the library prepends plus the (already truncated) text.
            tokens_in=view.overhead_subwords + n_subwords,
            tokens_out=None,
            meta={
                "truncated": was_truncated,
                "budget_subwords": view.budget_subwords,
                "n_subwords": n_subwords,
                "overhead_subwords": view.overhead_subwords,
                "window": ENCODER_WINDOW,
                # The cost model, per row: one encoder pass per question, and the
                # schema this row was scored with held exactly one task. The
                # report's cost section reads these two keys instead of assuming
                # the one pooled pass the brief expected.
                "passes_per_record": 1,
                "schema_tasks": len(view.questions),
                "device": self.usage.get("device"),
                "latency_ms": latency_ms,
                "decoder": meta.get("decoder"),
                "feasible": meta.get("feasible"),
                "checkpoint": str(self.checkpoint),
            },
        )

    def run(
        self,
        records: list[Record],
        questions: Iterable[Question],
        targets: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Answer]]:
        questions = tuple(questions)
        self._ensure_loaded()

        answers: dict[str, dict[str, Answer]] = {question.id: {} for question in questions}
        truncated_ids: set[str] = set()
        started = time.perf_counter()

        for question in questions:
            # One task per schema, one schema per question: see the module
            # docstring for the measured reason. ``targets`` narrows the records
            # to those whose gold exists for this question.
            view = self._prepare((question,))
            needed = targets.get(question.id) if targets else None
            rows = [
                record
                for record in records
                if needed is None or record["record_id"] in needed
            ]
            for offset in range(0, len(rows), self.batch_size):
                chunk = rows[offset : offset + self.batch_size]
                fitted = [
                    self._fit(record["state"], view.budget_subwords) for record in chunk
                ]
                before = time.perf_counter()
                results = self._clf.batch_classify(
                    [text for text, _n, _t in fitted],
                    view.schema,
                    config=self._config(len(chunk)),
                )
                elapsed_ms = (time.perf_counter() - before) * 1000.0
                self.usage["encoder_passes"] = self.usage.get("encoder_passes", 0) + 1
                per_record_ms = elapsed_ms / max(len(chunk), 1)

                for record, fitted_row, result in zip(chunk, fitted, results):
                    _text, n_subwords, was_truncated = fitted_row
                    if was_truncated:
                        truncated_ids.add(record["record_id"])
                    meta = result.to_dict().get("_meta", {})
                    answers[question.id][record["record_id"]] = self._answer(
                        question, result, meta, view, n_subwords, per_record_ms,
                        was_truncated,
                    )

        answered = sum(len(per_question) for per_question in answers.values())
        self.usage["records"] = len(
            {record_id for per_question in answers.values() for record_id in per_question}
        )
        self.usage["answers"] = answered
        self.usage["truncated_rows"] = sum(
            1
            for per_question in answers.values()
            for answer in per_question.values()
            if answer.meta["truncated"]
        )
        self.usage["truncated_records"] = len(truncated_ids)
        self.usage["truncated_rate"] = (
            round(len(truncated_ids) / self.usage["records"], 4) if self.usage["records"] else 0.0
        )
        self.usage["seconds"] = round(time.perf_counter() - started, 1)
        return answers

    def close(self) -> None:
        self._clf = None
        self._processor = None
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - releasing is best effort
            pass
