"""Question definitions: label sets, rubric text, and gold derivation.

This module is the single source of truth for every arm and for the evaluator:

* the label set each question is answered over,
* the rubric text handed to a model,
* how a gold label is derived from a dataset row.

Because all four engines read their questions from here, the differences in
``results/report.md`` are differences between engines, not between prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

# --------------------------------------------------------------------------
# Label sets
# --------------------------------------------------------------------------

FRAUD_LABELS: tuple[str, ...] = ("legitimate", "fraudulent")

EXPERIENCE_LABELS: tuple[str, ...] = (
    "internship",
    "entry_level",
    "associate",
    "mid_senior",
    "director",
    "executive",
    "not_applicable",
)

EDUCATION_LABELS: tuple[str, ...] = (
    "bachelors",
    "high_school",
    "unspecified",
    "masters",
    "associate",
    "certification",
    "other",
)

SALARY_LABELS: tuple[str, ...] = ("stated", "not_stated")

#: Ordinal position of the six ordered experience levels. ``not_applicable`` is
#: deliberately absent: it is scored in accuracy but excluded from MAE and from
#: the +/-1 accuracy, because it is not a position on the ladder.
ORDINAL: dict[str, int] = {
    "internship": 0,
    "entry_level": 1,
    "associate": 2,
    "mid_senior": 3,
    "director": 4,
    "executive": 5,
}

# --------------------------------------------------------------------------
# Gold mapping (dataset column value -> label)
# --------------------------------------------------------------------------

#: The six ``required_education`` values that carry >=30 examples in the test
#: split keep their own label; every other value (Some College Coursework
#: Completed, Professional, Doctorate, the two Vocational variants, Some High
#: School Coursework) collapses into ``other``.
EDUCATION_MAP: dict[str, str] = {
    "bachelor's degree": "bachelors",
    "high school or equivalent": "high_school",
    "unspecified": "unspecified",
    "master's degree": "masters",
    "associate degree": "associate",
    "certification": "certification",
}

EXPERIENCE_MAP: dict[str, str] = {
    "internship": "internship",
    "entry level": "entry_level",
    "associate": "associate",
    "mid-senior level": "mid_senior",
    "director": "director",
    "executive": "executive",
    "not applicable": "not_applicable",
}


def _text(value: Any) -> str | None:
    """Normalize a raw dataset cell to a stripped string, or ``None``.

    ``None``, NaN and the empty string all mean "the platform recorded nothing",
    and all three appear in the source data.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = str(value).strip()
    return text or None


def gold_fraudulent(row: Mapping[str, Any]) -> str | None:
    raw = row.get("fraudulent")
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:
        return None
    try:
        return "fraudulent" if int(raw) == 1 else "legitimate"
    except (TypeError, ValueError):
        return None


def gold_experience(row: Mapping[str, Any]) -> str | None:
    text = _text(row.get("required_experience"))
    if text is None:
        return None
    return EXPERIENCE_MAP.get(text.lower())


def gold_education(row: Mapping[str, Any]) -> str | None:
    text = _text(row.get("required_education"))
    if text is None:
        return None
    return EDUCATION_MAP.get(text.lower(), "other")


def gold_salary(row: Mapping[str, Any]) -> str | None:
    """Control question: the answer is platform metadata, not posting text."""
    return "not_stated" if _text(row.get("salary_range")) is None else "stated"


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    labels: tuple[str, ...]
    instructions: str
    kind: str
    gold: Callable[[Mapping[str, Any]], str | None]

    @property
    def positive(self) -> str | None:
        """The label that matters for the rare/positive class, if any."""
        return "fraudulent" if self.id == "fraudulent" else None

    @property
    def ordinal(self) -> bool:
        return self.id == "required_experience"


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="fraudulent",
        labels=FRAUD_LABELS,
        instructions=(
            "Judge whether the posting is a scam. Fraudulent postings typically "
            "ask for money, bank details or a fee, promise pay with no interview "
            "or no experience required, use a free mailbox for a corporate role, "
            "or describe an office with no address. Most postings are legitimate; "
            "flag only clear scam patterns."
        ),
        kind="binary, rare positive",
        gold=gold_fraudulent,
    ),
    Question(
        id="required_experience",
        labels=EXPERIENCE_LABELS,
        instructions=(
            "Judge the level of experience the posting asks for, from the "
            "responsibilities and requirements it states, not from the seniority "
            "word in the title alone."
        ),
        kind="7-way, 6 ordered levels + not_applicable",
        gold=gold_experience,
    ),
    Question(
        id="required_education",
        labels=EDUCATION_LABELS,
        instructions=(
            "Judge the highest formal education the posting requires or prefers. "
            "`unspecified` when the posting does not say."
        ),
        kind="7-way categorical",
        gold=gold_education,
    ),
    Question(
        id="salary_range_stated",
        labels=SALARY_LABELS,
        instructions="Does the posting state a numeric salary or pay range?",
        kind="control",
        gold=gold_salary,
    ),
)

QUESTION_IDS: tuple[str, ...] = tuple(q.id for q in QUESTIONS)


def by_id(question_id: str) -> Question:
    for question in QUESTIONS:
        if question.id == question_id:
            return question
    raise KeyError(f"unknown question id: {question_id!r}")


def select(ids: Iterable[str] | None) -> tuple[Question, ...]:
    """The questions named by ``ids``, in canonical order; ``None`` = all."""
    if ids is None:
        return QUESTIONS
    wanted = set(ids)
    unknown = wanted - set(QUESTION_IDS)
    if unknown:
        raise KeyError(f"unknown question id(s): {sorted(unknown)}")
    return tuple(q for q in QUESTIONS if q.id in wanted)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

STATE_CHAR_LIMIT = 6_000


def state(row: Mapping[str, Any], limit: int = STATE_CHAR_LIMIT) -> str:
    """The model-visible input: title + description, and nothing else.

    The answer columns (``fraudulent``, ``required_experience``,
    ``required_education``, ``salary_range``) are never part of the state.
    """
    title = _text(row.get("title")) or ""
    description = _text(row.get("description")) or ""
    text = f"{title}\n\n{description}".strip()
    return text[:limit]
