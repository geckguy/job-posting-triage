"""Why this arm runs one classification task per schema, in one table.

    uv run python scripts/gliner_pooling_probe.py [--checkpoint PATH] [--device mps]

GLiNER2 scores every ``[L] label`` marker in the schema against the document
embedding, so label sets that share a schema share the encoder's attention. On
this checkpoint the cost of that sharing is not subtle: the fraud question stops
separating a scam from an honest posting as soon as other classification tasks
join the schema, and the ranking inverts from three tasks up.

The probe prints P(fraudulent) for the three fixtures in
``scripts/llm_mode_probe.py`` (an obvious scam, an ordinary engineering posting,
and a recipe that is not a job ad) with one, two, three and four tasks in the
schema. The fixtures are about 50 subwords each, far inside the 512-subword
window, so nothing here depends on truncation, and nothing here reads the
benchmark split: it needs only the checkpoint.

Two of the three fixtures are not scams. An engine that scores the recipe, or the
honest posting, above the scam is not reading the posting, whatever it scores on
the real corpus, which is why the arm answers each question from its own schema.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for entry in (str(ROOT), str(ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from llm_mode_probe import POSTINGS  # the one set of fixtures, shared with the LLM probe

from triage.schema import QUESTIONS

DEFAULT_CHECKPOINT = str(ROOT / "models" / "gliner2.5-base-v1")
QUESTION_ORDER = [question.id for question in QUESTIONS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gliner_pooling_probe")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="checkpoint directory or hub id")
    parser.add_argument("--device", default=None, help="torch device; default mps if available")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).exists() and "/" not in args.checkpoint:
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    warnings.simplefilter("ignore")
    import torch
    from gliner2.classification import Classifier, ClassificationConfig, ClassificationSchema

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    texts = [text for _name, text in POSTINGS]
    names = [name for name, _text in POSTINGS]

    clf = Classifier.from_pretrained(args.checkpoint)
    clf.to(device=device)
    print(f"checkpoint {args.checkpoint} on {device}")
    print(f"fixtures: {', '.join(names)}")
    print()
    print(f"{'tasks':>6}  schema")
    for size in range(1, len(QUESTION_ORDER) + 1):
        schema = ClassificationSchema()
        for task_id in QUESTION_ORDER[:size]:
            question = next(q for q in QUESTIONS if q.id == task_id)
            schema.single(task_id, list(question.labels))
        results = clf.batch_classify(
            texts, schema, config=ClassificationConfig(batch_size=args.batch_size)
        )
        scores = [
            float(result.probabilities("fraudulent")["fraudulent"]) for result in results
        ]
        flag = "  <- inverted: a non-scam outscores the scam" if max(scores[1:]) > scores[0] else ""
        print(f"{size:>6}  {', '.join(QUESTION_ORDER[:size])}")
        print(f"        P(fraudulent) " + "  ".join(
            f"{name}={score:.4f}" for name, score in zip(names, scores)
        ) + flag)
    print()
    print("One task per schema is what this arm uses: it is the only width at which the "
          "scam outscores both non-scams.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
