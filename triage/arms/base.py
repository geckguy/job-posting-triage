"""Four engines behind one interface.

An arm answers questions about postings. It may or may not be able to express
uncertainty: when it cannot, ``probabilities`` is ``None`` and the evaluator
prints ``n/a`` for its calibration rows rather than a fabricated number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, TypedDict

from triage.schema import Question


@dataclass
class Answer:
    value: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    raw: Any = None
    invalid: bool = False
    tokens_in: int | None = None
    tokens_out: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "invalid": self.invalid,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "meta": self.meta,
        }


class Record(TypedDict):
    record_id: str
    state: str


class ArmUnavailable(RuntimeError):
    """The arm cannot run here at all (no service, no weights, no key)."""


class ArmPartial(RuntimeError):
    """The arm ran out of quota or the service kept failing mid-run.

    Carries whatever was answered before the stop so the runner can persist it
    and resume from cache, instead of silently truncating the results.
    """

    def __init__(self, message: str, answers: dict[str, dict[str, Answer]] | None = None):
        super().__init__(message)
        self.answers = answers or {}


class Arm:
    """Base class. Subclasses set ``name`` and implement ``available``/``run``."""

    name: str = "base"

    def available(self) -> tuple[bool, str]:
        """``(True, "")`` if this arm can run here, else ``(False, reason)``."""
        return True, ""

    def run(
        self,
        records: list[Record],
        questions: Iterable[Question],
        targets: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Answer]]:
        """Answer ``questions`` for ``records``.

        ``targets`` maps question id -> the record ids that actually need an
        answer for that question (the records whose gold exists). ``None`` means
        every record needs every question. Arms that can exploit it (the
        batched remote arm) narrow their requests; arms that cannot may answer
        everything and let the runner drop what it does not need.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass
