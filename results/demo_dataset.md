# Demo -- labeled postings

- **Source**: `james-burton/fake_job_postings2` (split `test`), 3,182 rows; uniform random draw of 5 (`random.Random(0).sample`, no replacement)
- **n**: 5 postings
- **Rows drawn**: 5 of 3,182, indices 165-3104 (record ids are `test-<row index>`)
- **Seed**: `0` (same seed and `--n` reproduce this exact sample)
- **Questions asked**: `fraudulent`
- **High-confidence threshold**: `p_positive >= 0.9`
- **Arms**: `classifier_dev`
- **Generated**: 2026-09-19T14:51:14Z by `python -m triage.demo`

The sample is a uniform random draw over the whole split, not the first rows: a prefix of this corpus is a biased frame, because postings cluster (a single employer's campaign can occupy a run of consecutive rows), so a prefix would report that campaign's flag rate as if it were the corpus's.

## Arms

| arm (artifact name) | requested as | endpoint host | status | returned | usable | invalid | seconds | notes |
|---|---|---|---|---|---|---|---|---|
| `classifier_dev` | same | classifier.dev | ok | 5 | 5 | 0 | 0.3 | ok | classifications: 0, cached: 5 |

The endpoint column is the host each arm actually talked to, so a hosted run cannot be confused with a local one; API keys are never recorded. `(in-process)` means the model or the features ran on this machine with no network call.

`returned` counts the answers an arm handed back; `usable` counts those with a non-empty `value` that the arm did not itself mark `invalid`; `invalid` counts the rest. An arm that returned answers and 0 usable ones answered nothing, and `status: ok` there means only that the arm did not raise, not that its answers are labels. Those postings are counted as `(no answer)`, never as a negative label.

`unavailable`/`failed` means the arm produced nothing here and the reason is printed above; no other model, dataset or metric was substituted for it.

`llm_local_openai_api` is an arm requested as `llm_hosted` whose base URL resolved to loopback: the local llama-server reached through the OpenAI-compatible HTTP path, i.e. the same model as `llm_local`, not a hosted model. Rows produced that way are never called hosted.

## Probability convention

`p_positive` = P(`fraudulent`) for one answer: `probabilities["fraudulent"]` when the arm reports a distribution (classifier_dev, tfidf, gliner); otherwise the arm's self-reported confidence in its own label, mirrored when the label is the negative one (`legitimate` at 0.9 implies `p_positive` 0.1) -- the LLM arms. `n/a` means the arm reported neither field, so the posting cannot be thresholded for that arm. `confidence` is always the arm's confidence in the label it chose.

## Labels -- `fraudulent`

Labels: `legitimate`, `fraudulent`

| arm | legitimate | fraudulent | (no answer) |
|---|---|---|---|
| `classifier_dev` | 5 | 0 | 0 |

## Fraud-flag rate

Flagged = the arm's `value` is `fraudulent` (not a threshold on probability). `scored` is how many postings the arm produced a `p_positive` for; mean confidence is the arm's own certainty in the label it chose, and a mean near 0 with a mean `p_positive` near 1 is the degenerate-confidence pattern described under the flag table below.

| arm | flagged | rate | scored | mean p_positive | mean confidence |
|---|---|---|---|---|---|
| `classifier_dev` | 0/5 | 0.000 | 5/5 | 0.044 | 0.910 |

## Agreement on the fraud label

Compared = postings where both arms returned a label, so rows with an empty `value` drop out of the pair rather than counting as disagreement.

Only `classifier_dev` returned labels here, so there is no pair to compare. This demo does not score gold, so no accuracy is reported.

## Postings flagged at `p_positive >= 0.9`, or labelled `fraudulent`

At `p_positive >= 0.9` by every arm that could score a posting: **0/5** (`classifier_dev`). At `p_positive >= 0.9` by at least one arm: **0/5**. Labelled `fraudulent` by at least one arm, at any score: **0/5**.

No posting reached `p_positive >= 0.9` for any arm, and no arm labelled any posting `fraudulent`.

## This source is labeled -- but this run did not score it

`james-burton/fake_job_postings2` ships a `fraudulent` column, so accuracy is computable here; `triage.run` + `triage.evaluate` are the scoring path. This artifact deliberately reports only what it computed: flag counts and agreement, without reading gold. Read the flag rates as a sanity check on the demo wiring, and `results/report.md` for the metrics.

