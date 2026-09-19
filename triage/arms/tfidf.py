"""Classical reference arm: TF-IDF + logistic regression.

Fits on ``train`` + ``validation`` (every row with a non-null gold for that
question) and predicts the test split. This is the arm that answers "how much
of this question was ever worth a model of this class?", and it is the floor
against which the three neural arms have to justify themselves.
"""

from __future__ import annotations

import time

from triage import data
from triage.arms.base import Answer, Arm, Record
from triage.schema import Question

FIT_SPLITS = ("train", "validation")
#: The split the decision threshold is tuned on. Never the split being scored.
TUNE_SPLIT = "validation"
#: Threshold grid for the rare positive class. The default 0.5 boundary of an
#: unbalanced logistic regression predicts the majority class almost always, so
#: the operating point is chosen on validation instead of shipped untuned.
THRESHOLD_GRID = [index / 100 for index in range(2, 96)]


def _other(question: Question, positive: str) -> str:
    """The one remaining label of a two-label question."""
    others = [label for label in question.labels if label != positive]
    if len(others) != 1:
        raise ValueError(f"{question.id} is not a two-label question")
    return others[0]


class TfidfArm(Arm):
    name = "tfidf"

    def __init__(self, split: str = "test", seed: int = 0) -> None:
        self.split = split
        self.seed = seed
        self._models: dict[str, tuple[object, object]] = {}
        self._fitted: dict[str, dict[str, Answer]] = {}
        self._thresholds: dict[str, float | None] = {}
        self._tune_scores: dict[str, float] = {}

    def available(self) -> tuple[bool, str]:
        try:
            import sklearn  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency is declared
            return False, f"scikit-learn unavailable: {exc}"
        return True, ""

    # -- fitting ----------------------------------------------------------

    def _fit(self, question: Question) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        states: list[str] = []
        labels: list[str] = []
        tune_states: list[str] = []
        tune_labels: list[str] = []
        for split in FIT_SPLITS:
            records = {r["record_id"]: r["state"] for r in data.records(split)}
            for record_id, gold in data.golds(split, (question,))[question.id].items():
                states.append(records[record_id])
                labels.append(gold)
                if split == TUNE_SPLIT:
                    tune_states.append(records[record_id])
                    tune_labels.append(gold)

        # A posting whose whole state is empty would make the vectorizer emit a
        # constant column; no split has one, but a sentinel keeps it safe.
        states = [s if s.strip() else "(empty posting)" for s in states]
        tune_states = [s if s.strip() else "(empty posting)" for s in tune_states]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(states)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(matrix, labels)
        self._models[question.id] = (vectorizer, clf)

        threshold = None
        if question.positive is not None and len(set(tune_labels)) > 1:
            threshold = self._tune_threshold(
                question, vectorizer, clf, tune_states, tune_labels
            )
        self._thresholds[question.id] = threshold

    def _tune_threshold(
        self,
        question: Question,
        vectorizer,
        clf,
        tune_states: list[str],
        tune_labels: list[str],
    ) -> float:
        """Pick the operating point on the validation split, for the rare class.

        A logistic regression fitted on a 4.3 percent positive class predicts the
        majority label almost always at its default 0.5 boundary, which makes the
        F1 of the positive class look like a model failure when the ranking
        (measured by PR-AUC) is in fact strong. The threshold is therefore chosen
        on validation by maximising the positive class F1, and recorded in the
        row metadata so the choice is visible. No part of the scored split is
        used to choose it.
        """
        from sklearn.metrics import f1_score

        positive = question.positive
        probabilities = clf.predict_proba(vectorizer.transform(tune_states))
        index = [str(c) for c in clf.classes_].index(positive)
        scores = [float(row[index]) for row in probabilities]
        truth = [1 if label == positive else 0 for label in tune_labels]

        best_threshold, best_score = 0.5, -1.0
        for threshold in THRESHOLD_GRID:
            predicted = [1 if score >= threshold else 0 for score in scores]
            score = f1_score(truth, predicted, zero_division=0)
            if score > best_score:
                best_threshold, best_score = threshold, score
        self._tune_scores[question.id] = best_score
        return best_threshold

    def _predict(self, question: Question, rows: list[Record]) -> None:
        vectorizer, clf = self._models[question.id]
        matrix = vectorizer.transform(
            [r["state"] if r["state"].strip() else "(empty posting)" for r in rows]
        )
        started = time.perf_counter()
        proba = clf.predict_proba(matrix)
        elapsed_ms = (time.perf_counter() - started) * 1000
        classes = [str(c) for c in clf.classes_]

        per_record: dict[str, Answer] = {}
        threshold = self._thresholds.get(question.id)
        for row, row_proba in zip(rows, proba):
            # Keyed by the question's canonical labels: a label with no training
            # examples can never be predicted, and its true probability under
            # this model is 0.0. Dict shape then matches every other arm.
            scores = {label: 0.0 for label in question.labels}
            for label, value in zip(classes, row_proba):
                scores[label] = float(value)
            best = max(scores, key=lambda label: scores[label])
            if threshold is not None:
                positive = question.positive
                decided = positive if scores[positive] >= threshold else _other(
                    question, positive
                )
            else:
                decided = best
            per_record[row["record_id"]] = Answer(
                value=decided,
                probabilities=scores,
                confidence=scores[decided],
                raw={"classes": classes},
                meta={
                    "n_features": int(len(vectorizer.vocabulary_)),
                    "argmax": best,
                    "decision_threshold": threshold,
                    "validation_f1_at_threshold": self._tune_scores.get(question.id),
                    "latency_ms": elapsed_ms / max(len(rows), 1),
                },
            )
        self._fitted[question.id] = per_record

    # -- Arm --------------------------------------------------------------

    def run(
        self,
        records: list[Record],
        questions,
        targets: dict[str, set[str]] | None = None,
    ) -> dict[str, dict[str, Answer]]:
        out: dict[str, dict[str, Answer]] = {}
        for question in questions:
            if question.id not in self._models:
                self._fit(question)
            wanted = targets.get(question.id) if targets else None
            rows = (
                records if wanted is None else [r for r in records if r["record_id"] in wanted]
            )
            if question.id not in self._fitted:
                self._predict(question, rows)
            else:
                # A second call for the same question (e.g. the ablation run)
                # reuses the fitted model rather than refitting.
                self._predict(question, rows)
            out[question.id] = self._fitted[question.id]
        return out
