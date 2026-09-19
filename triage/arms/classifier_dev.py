"""Jev arm: TypeSafe's decision model, served free and keyless by classifier.dev.

One POST classifies up to 1,000 texts (200 on the smart tier) against one label
set, returning a calibrated ``confidence`` and a probability per label from a
single forward pass. Questions are therefore submitted one at a time, in
question-major chunks, and only for the records whose gold exists.

Every answer is cached in ``results/cache.jsonl`` keyed by the exact request
material plus the posting text, so a rerun, a ``--limit`` top-up or a retry after
a quota stop costs no additional classifications.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import httpx

from triage.arms.base import (
    Answer,
    Arm,
    ArmPartial,
    ArmUnavailable,
    Record,
)
from triage.schema import Question

API_URL = "https://classifier.dev/v1/classify"
HEALTH_URL = "https://classifier.dev/v1/health"

#: The public service caps a single request at 1,000 inputs on the fast tier and
#: at 200 on the smart tier, so a batch fits its per-minute quota.
CHUNK_FAST = 1_000
CHUNK_SMART = 200

MAX_ATTEMPTS = 5
#: Never sleep longer than this for a ``Retry-After``; a longer one means "come
#: back later", which is a stop, not a wait.
MAX_RETRY_AFTER = 60.0
BACKOFF_BASE = 2.0


def _key(
    question_id: str,
    labels: Iterable[str],
    instructions: str | None,
    tier: str,
    state: str,
) -> str:
    material = json.dumps(
        {
            "question": question_id,
            "labels": list(labels),
            "instructions": instructions,
            "tier": tier,
            "state": state,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ClassifierDevArm(Arm):
    """Wraps the classifier.dev HTTP API."""

    name = "classifier_dev"

    def __init__(
        self,
        tier: str = "fast",
        use_instructions: bool = True,
        cache_path: Path | str = "results/cache.jsonl",
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.tier = tier
        self.use_instructions = use_instructions
        self.cache_path = Path(cache_path)
        self.chunk = CHUNK_SMART if tier == "smart" else CHUNK_FAST
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._cache: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._pending: list[dict[str, Any]] = []
        self.usage = {"classifications": 0, "requests": 0, "escalated": 0, "cached": 0}

    # -- cache ------------------------------------------------------------

    def _load_cache(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.cache_path.exists():
            return
        with self.cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "key" in entry:
                    self._cache[entry["key"]] = entry

    def _flush(self) -> None:
        if not self._pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            for entry in self._pending:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
        self._pending.clear()

    # -- transport --------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        try:
            response = self._client.get(HEALTH_URL)
        except httpx.HTTPError as exc:
            return False, f"classifier.dev unreachable: {type(exc).__name__}: {exc}"
        if response.status_code != 200:
            return False, f"classifier.dev health returned HTTP {response.status_code}"
        return True, ""

    def _post(self, payload: dict[str, Any], question: Question, target_ids: list[str]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.post(API_URL, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == MAX_ATTEMPTS:
                    break
                time.sleep(BACKOFF_BASE ** attempt)
                continue

            if response.status_code == 200:
                return response.json()

            body = response.text[:300]
            if response.status_code == 429:
                last_error = f"HTTP 429 {body}"
                retry_after = _retry_after(response)
                if retry_after is not None and retry_after > MAX_RETRY_AFTER:
                    raise ArmPartial(
                        f"quota exhausted on tier {self.tier}: {last_error}; "
                        f"Retry-After {retry_after:.0f}s (resume with the same command)"
                    )
                if attempt == MAX_ATTEMPTS:
                    break
                time.sleep(retry_after if retry_after is not None else BACKOFF_BASE ** attempt)
                continue

            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code} {body}"
                if attempt == MAX_ATTEMPTS:
                    break
                time.sleep(BACKOFF_BASE ** attempt)
                continue

            raise ArmUnavailable(
                f"classifier.dev rejected {question.id} with HTTP "
                f"{response.status_code}: {body}"
            )
        raise ArmPartial(
            f"gave up on {question.id} after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    # -- Arm --------------------------------------------------------------

    def run(
        self,
        records: list[Record],
        questions: Iterable[Question],
        targets: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Answer]]:
        self._load_cache()
        states = {r["record_id"]: r["state"] for r in records}
        out: dict[str, dict[str, Answer]] = {}

        for question in questions:
            wanted = (
                set(states) if targets is None else set(targets.get(question.id, set()))
            )
            wanted &= set(states)
            record_ids = [r["record_id"] for r in records if r["record_id"] in wanted]

            instructions = question.instructions if self.use_instructions else None
            per_question: dict[str, Answer] = {}
            out[question.id] = per_question

            fresh: list[str] = []
            for record_id in record_ids:
                key = _key(question.id, question.labels, instructions, self.tier, states[record_id])
                cached = self._cache.get(key)
                if cached is None:
                    fresh.append(record_id)
                    continue
                self.usage["cached"] += 1
                per_question[record_id] = _answer_from(cached, cached=True)

            for start in range(0, len(fresh), self.chunk):
                batch_ids = fresh[start : start + self.chunk]
                payload: dict[str, Any] = {
                    "inputs": [states[record_id] for record_id in batch_ids],
                    "labels": list(question.labels),
                    "tier": self.tier,
                }
                if instructions is not None:
                    payload["instructions"] = instructions

                try:
                    started = time.perf_counter()
                    body = self._post(payload, question, batch_ids)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                except ArmPartial as exc:
                    self._flush()
                    exc.answers = out
                    raise
                # Client-observed time per answer, so this arm's latency is the
                # same quantity as every other arm's: wall clock from the client,
                # network and batching included. The service reports its own
                # per-answer figure, which is kept separately as `server_ms`
                # because the two differ by more than an order of magnitude and
                # only the wall clock answers "how long does this take me".
                per_row_ms = elapsed_ms / max(len(batch_ids), 1)

                results = body.get("results")
                if not isinstance(results, list) or len(results) != len(batch_ids):
                    # Never pair answers with the wrong postings.
                    self._flush()
                    raise ArmUnavailable(
                        f"classifier.dev returned {len(results) if isinstance(results, list) else 'no'} "
                        f"results for {len(batch_ids)} inputs on {question.id}"
                    )

                self.usage["requests"] += 1
                usage = body.get("usage") or {}
                self.usage["classifications"] += int(usage.get("classifications") or len(batch_ids))
                self.usage["escalated"] += int(usage.get("escalated") or 0)

                for record_id, result in zip(batch_ids, results):
                    entry = {
                        "key": _key(
                            question.id, question.labels, instructions, self.tier, states[record_id]
                        ),
                        "question": question.id,
                        "tier": self.tier,
                        "model": result.get("model", body.get("model")),
                        "value": result.get("label"),
                        "confidence": result.get("confidence"),
                        "probabilities": result.get("scores"),
                        "unscored": result.get("unscored"),
                        "ms": result.get("ms"),
                        "latency_ms": per_row_ms,
                        "batch_size": len(batch_ids),
                    }
                    self._cache[entry["key"]] = entry
                    self._pending.append(entry)
                    per_question[record_id] = _answer_from(entry)
                self._flush()

        return out

    def close(self) -> None:
        self._flush()
        self._client.close()


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _answer_from(entry: dict[str, Any], cached: bool = False) -> Answer:
    """One stored answer. ``cached`` means this run did not fetch it.

    A cache hit carries the server's timing from whichever run fetched it first,
    so reporting that as this row's latency would put a number nobody measured
    into the cost table. Cached rows therefore state that no latency was measured,
    and the runner leaves the field null so the statistics skip them.
    """
    probabilities = entry.get("probabilities")
    return Answer(
        value=str(entry.get("value")),
        probabilities=probabilities,
        confidence=entry.get("confidence"),
        raw=entry.get("unscored"),
        meta={
            "model": entry.get("model"),
            "tier": entry.get("tier"),
            "ms": None if cached else entry.get("ms"),
            "server_ms": entry.get("ms"),
            "latency_ms": None if cached else entry.get("latency_ms"),
            "batch_size": entry.get("batch_size"),
            "latency_measured": not cached,
            "cached": cached,
            "source": "classifier.dev",
        },
    )


def fraud_subsample(
    record_ids: list[str], positives: set[str], total: int, seed: int = 0
) -> list[str]:
    """All positives plus random negatives, capped at ``total``.

    The smart tier's 2,000/day ceiling cannot cover the 3,182 test postings, so
    it runs on a fraud-only subsample that at least contains every positive.
    """
    if total >= len(record_ids):
        return list(record_ids)
    rng = random.Random(seed)
    pos = [r for r in record_ids if r in positives]
    neg = [r for r in record_ids if r not in positives]
    keep = min(len(pos), total)
    negatives = min(total - keep, len(neg))
    return pos[:keep] + rng.sample(neg, negatives)
