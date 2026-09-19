# job-posting-triage: evaluation report

Results directory: `results/fullsplit`. Arms found: `classifier_dev`, `tfidf`. Arms in this report: `classifier_dev`, `tfidf`.

Alongside this file, `--report` writes `metrics.json`: the same numbers, from the same code path, for machine consumption (`n_a` there is the list in section 8).

## 1. What was run

| file | rows read | misparsed |
|---|---|---|
| classifier_dev.jsonl | 3182 | 0 |
| tfidf.jsonl | 3182 | 0 |

| arm | rows | source file(s) | splits | status | reason |
|---|---|---|---|---|---|
| classifier_dev | 3182 | classifier_dev.jsonl | test | no entry in _unavailable.json |  |
| tfidf | 3182 | tfidf.jsonl | test | no entry in _unavailable.json |  |

* no _unavailable.json in this directory: arms that failed without leaving rows are not visible here.

Gold source: 6364 row(s) scored against the dataset gold from `triage/data.py`, 0 against the `gold` field of their own result row (records absent from that dataset split, e.g. synthetic fixtures).

### Frame: which records each arm answered

A question is scored only where its gold exists, so the sample size per question differs by design. What must not differ is which records an arm answered for the same question: an arm on a different set of postings has rows that are not directly comparable with the others. `shared` is the intersection across every arm, `majority set` is the record set that the largest number of arms answered, and mismatches are measured against it.

| question | arms | union | shared | majority set | records per arm | frame |
|---|---|---|---|---|---|---|
| fraudulent | 2 | 3182 | 3182 | 3182 | classifier_dev=3182, tfidf=3182 | match |

Frame source: derived from the record ids in the results; no frame was recorded, so this cannot distinguish a random sample from a prefix.

Every arm answered the same records for every question it scored: no WARN.

Questions with scored rows: `fraudulent` (12728 arm-rows).
## 2. Metric definitions

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

## 3. Per-question metrics

Every table carries both floors: `majority_class` (most frequent gold label of the scored subset) and `dummy_stratified` (fitted on train+validation gold, see section 2). No arm is shown without them.

### `fraudulent` (binary, rare positive, 2 labels)

| arm | n | accuracy | macro-F1 | invalid_json_rate | #positives | F1(fraudulent) | precision(fraudulent) | recall(fraudulent) | PR-AUC (AP) | precision_ci95 | recall_ci95 | f1_ci95 | ci_note | TP/FP/FN/TN | no_call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classifier_dev | 3182 | 0.953 | 0.601 | 0.000 | 138 | 0.227 | 0.393 | 0.159 | 0.182 | [0.28, 0.52] | [0.11, 0.23] | [0.15, 0.31] | 138 positives in this frame | 22/34/116/3010 | 0 |
| tfidf | 3182 | 0.975 | 0.840 | 0.000 | 138 | 0.692 | 0.738 | 0.652 | 0.761 | [0.65, 0.81] | [0.57, 0.73] | [0.63, 0.75] | 138 positives in this frame | 90/32/48/3012 | 0 |
| majority_class | 3182 | 0.957 | 0.489 | 0.000 | 138 | 0.000 | n/a | 0.000 | n/a | n/a | [0.00, 0.03] | [0.00, 0.00] | 138 positives in this frame | 0/0/138/3044 | 0 |
| dummy_stratified | 3182 | 0.909 | 0.499 | 0.000 | 138 | 0.046 | 0.042 | 0.051 | 0.043 | [0.02, 0.08] | [0.02, 0.10] | [0.01, 0.08] | 138 positives in this frame | 7/159/131/2885 | 0 |

Per-class detail, over the same scored rows as the table above (all n of them, not a per-label subset): precision/recall/F1 come from sklearn with the question's declared label set, so a label nobody predicted contributes a zero to macro-F1 rather than disappearing. A precision cell is `n/a` when the arm never predicted that label: no row of this arm was predicted with this label, so precision has no denominator and is undefined.

| arm | label | precision | recall | F1 | support |
|---|---|---|---|---|---|
| classifier_dev | legitimate | 0.963 | 0.989 | 0.976 | 3044 |
| classifier_dev | fraudulent | 0.393 | 0.159 | 0.227 | 138 |
| tfidf | legitimate | 0.984 | 0.989 | 0.987 | 3044 |
| tfidf | fraudulent | 0.738 | 0.652 | 0.692 | 138 |
| majority_class | legitimate | 0.957 | 1.000 | 0.978 | 3044 |
| majority_class | fraudulent | n/a | 0.000 | 0.000 | 138 |
| dummy_stratified | legitimate | 0.957 | 0.948 | 0.952 | 3044 |
| dummy_stratified | fraudulent | 0.042 | 0.051 | 0.046 | 138 |

Confusion convention: the four cells are read over every scored row and they sum to n only after the `no_call` column, which counts rows where the arm produced no label from the declared set (an empty reply, a junk value, or a row it flagged invalid). A `no_call` row is neither a false positive nor a true negative: an empty reply made no call at all, so scoring it as a correct negative would be the exact failure this benchmark exists to catch. `no_call` rows are wrong in accuracy, they count as missed positives when their gold is the positive label (so recall and the positive-class F1 carry them), and `invalid_json_rate` counts the subset of them the arm itself flagged. Check the arithmetic with: `accuracy x n == TP + TN` and `TP + FP + FN + TN + no_call == n`.

Intervals: `precision_ci95` and `recall_ci95` are two-sided 95 percent Wilson score intervals on the counts already in the table (TP/(TP+FP) and TP/positives); `f1_ci95` is a deterministic bootstrap (2000 resamples of the scored rows with random.Random(0), percentiles by linear interpolation), so a rerun gives the same interval. `ci_note` records the frame: 138 positives in this frame.

With 138 positives in this frame, the positive-class estimates carry intervals roughly plus or minus ten percentage points wide (a little wider for precision, see the ci95 columns), the arms' intervals separate on recall and nearly separate on precision, and the same two arms were also run over the whole split where the prevalence is 4.3 percent and the point estimates matched (precision 0.74 both, recall 0.66 versus 0.65), so the ordering does not depend on the frame or on rebalancing. That whole-split comparison comes from a run outside this results directory; the intervals above are computed from the rows here.

Intervals in this frame: `classifier_dev` precision [0.28, 0.52] recall [0.11, 0.23]; `tfidf` precision [0.65, 0.81] recall [0.57, 0.73]; `majority_class` precision n/a recall [0.00, 0.03]; `dummy_stratified` precision [0.02, 0.08] recall [0.02, 0.10].

The same caveat applies to the multi-label questions (`required_experience`, `required_education`): their scored subsets are larger, so the intervals would be narrower, and they are not quantified here because precision and recall over a multi-label argmax are not a single binomial proportion.

### `required_experience`: n/a

No arm produced a row for this question in this results directory, so there is no table: nothing is invented to fill it.

### `required_education`: n/a

No arm produced a row for this question in this results directory, so there is no table: nothing is invented to fill it.

### `salary_range_stated`: n/a

No arm produced a row for this question in this results directory, so there is no table: nothing is invented to fill it.

## 4. Calibration

Brier and ECE are computed on `p_decision` (section 2), which is why an arm that cannot state a probability shows `n/a` rather than a substituted zero.

| arm | question | n scored | n with p_decision | excluded (null p_decision) | of which invalid | excluded but valid | Brier | ECE | invalid_json_rate |
|---|---|---|---|---|---|---|---|---|---|
| classifier_dev | fraudulent | 3182 | 3182 | 0 | 0 | 0 | 0.052 | 0.046 | 0.000 |
| tfidf | fraudulent | 3182 | 3182 | 0 | 0 | 0 | 0.028 | 0.033 | 0.000 |
| majority_class | fraudulent | 3182 | 0 | 3182 | 0 | 3182 | n/a | n/a | 0.000 |
| dummy_stratified | fraudulent | 3182 | 3182 | 0 | 0 | 0 | 0.091 | 0.091 | 0.000 |
| **classifier_dev (all questions)** | all | 3182 | 3182 | 0 | 0 | 0 | 0.052 | 0.046 | 0.000 |
| **tfidf (all questions)** | all | 3182 | 3182 | 0 | 0 | 0 | 0.028 | 0.033 | 0.000 |
| **dummy_stratified (all questions)** | all | 3182 | 3182 | 0 | 0 | 0 | 0.091 | 0.091 | 0.000 |
| **majority_class (all questions)** | all | 3182 | 0 | 3182 | 0 | 3182 | n/a | n/a | 0.000 |

Excluded rows are never dropped silently. *of which invalid* counts rows the arm itself marked `invalid` (its own output failed to parse or validate) that also carry no probability; *excluded but valid* counts rows that answered normally yet state no probability at all, which is why Brier/ECE cannot include them. Both sets still count as wrong in accuracy, macro-F1 and every other scored metric, and the n/a list in section 8 names them per arm and question.

### Reliability bins, `fraudulent`

| arm | bin | n | mean p_decision | empirical accuracy | gap |
|---|---|---|---|---|---|
| classifier_dev | [0.0, 0.1) | 16 | 0.052 | 0.812 | 0.761 |
| classifier_dev | [0.1, 0.2) | 23 | 0.141 | 0.478 | 0.337 |
| classifier_dev | [0.2, 0.3) | 12 | 0.236 | 0.833 | 0.598 |
| classifier_dev | [0.3, 0.4) | 19 | 0.342 | 0.684 | 0.343 |
| classifier_dev | [0.4, 0.5) | 22 | 0.436 | 0.682 | 0.245 |
| classifier_dev | [0.5, 0.6) | 25 | 0.539 | 0.880 | 0.341 |
| classifier_dev | [0.6, 0.7) | 44 | 0.649 | 0.932 | 0.283 |
| classifier_dev | [0.7, 0.8) | 65 | 0.752 | 0.938 | 0.186 |
| classifier_dev | [0.8, 0.9) | 109 | 0.851 | 0.936 | 0.084 |
| classifier_dev | [0.9, 1.0] | 2847 | 0.987 | 0.964 | 0.023 |
| tfidf | [0.1, 0.2) | 57 | 0.131 | 0.439 | 0.308 |
| tfidf | [0.2, 0.3) | 24 | 0.254 | 1.000 | 0.746 |
| tfidf | [0.3, 0.4) | 12 | 0.361 | 1.000 | 0.639 |
| tfidf | [0.4, 0.5) | 8 | 0.421 | 1.000 | 0.579 |
| tfidf | [0.5, 0.6) | 5 | 0.546 | 1.000 | 0.454 |
| tfidf | [0.6, 0.7) | 9 | 0.657 | 1.000 | 0.343 |
| tfidf | [0.7, 0.8) | 5 | 0.757 | 1.000 | 0.243 |
| tfidf | [0.8, 0.9) | 2 | 0.828 | 1.000 | 0.172 |
| tfidf | [0.9, 1.0] | 3060 | 0.968 | 0.984 | 0.016 |
| dummy_stratified | [0.9, 1.0] | 3182 | 1.000 | 0.909 | 0.091 |

Empty bins are omitted from the table; the ECE above sums over all ten bins, empty ones contributing zero.

## 5. Routing (`fraudulent`)

Acting means **auto-flag as scam**: the record leaves the human review queue and is treated as fraud. The sweep runs over `p_positive` (section 2). Coverage is the share of the evaluated records acted on; the remainder is what a human still has to read. A row with no `p_positive` can never be acted on, so it counts against coverage.

| arm | tau | acted | coverage | TP (scams flagged) | FP (legitimate flagged) | precision | recall | FN (scams missed) | positives acted/total | positives acted rate | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classifier_dev | 0.5 | 57 | 1.8% | 22 | 35 | 0.386 | 0.159 | 116 | 22/138 | 0.159 |  |
| classifier_dev | 0.7 | 34 | 1.1% | 19 | 15 | 0.559 | 0.138 | 119 | 19/138 | 0.138 |  |
| classifier_dev | 0.9 | 19 | 0.6% | 14 | 5 | 0.737 | 0.101 | 124 | 14/138 | 0.101 |  |
| classifier_dev | 0.95 | 15 | 0.5% | 10 | 5 | 0.667 | 0.072 | 128 | 10/138 | 0.072 |  |
| tfidf | 0.5 | 21 | 0.7% | 21 | 0 | 1.000 | 0.152 | 117 | 21/138 | 0.152 |  |
| tfidf | 0.7 | 7 | 0.2% | 7 | 0 | 1.000 | 0.051 | 131 | 7/138 | 0.051 |  |
| tfidf | 0.9 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 138 | 0/138 | 0.000 | nothing reaches p_positive >= 0.9 |
| tfidf | 0.95 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 138 | 0/138 | 0.000 | nothing reaches p_positive >= 0.95 |
| majority_class | 0.5 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 138 | 0/138 | 0.000 | nothing reaches p_positive >= 0.5 (3182 row(s) carry no p_positive) |
| majority_class | 0.7 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 138 | 0/138 | 0.000 | nothing reaches p_positive >= 0.7 (3182 row(s) carry no p_positive) |
| majority_class | 0.9 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 138 | 0/138 | 0.000 | nothing reaches p_positive >= 0.9 (3182 row(s) carry no p_positive) |
| majority_class | 0.95 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 138 | 0/138 | 0.000 | nothing reaches p_positive >= 0.95 (3182 row(s) carry no p_positive) |
| dummy_stratified | 0.5 | 166 | 5.2% | 7 | 159 | 0.042 | 0.051 | 131 | 7/138 | 0.051 |  |
| dummy_stratified | 0.7 | 166 | 5.2% | 7 | 159 | 0.042 | 0.051 | 131 | 7/138 | 0.051 |  |
| dummy_stratified | 0.9 | 166 | 5.2% | 7 | 159 | 0.042 | 0.051 | 131 | 7/138 | 0.051 |  |
| dummy_stratified | 0.95 | 166 | 5.2% | 7 | 159 | 0.042 | 0.051 | 131 | 7/138 | 0.051 |  |

`FN (scams missed)` counts positives below the threshold: the frauds the acted-on set does not catch. `positives acted/total` is the share of the positives that survive the filter, given as rate and count. At a threshold above every score nothing is acted on: coverage 0, precision and recall `n/a` with the reason stated, and no division by zero.

classifier.dev is a shared service and its probabilities shift slightly between fetches of the same posting, so the threshold-dependent cells above can move by one record between reruns while the label-based tables in section 3 do not. Treat the routing cells as indicative at the high thresholds, where the `acted` column shows how few postings carry the decision.

## 6. Cost and latency

| arm | rows | rows with latency | p50 latency_ms | p95 latency_ms | mean latency_ms | mean tokens_in | mean tokens_out | rows with token counts |
|---|---|---|---|---|---|---|---|---|
| classifier_dev | 3182 | 3182 | 11.0 | 11.0 | 11.0 | n/a | n/a | n/a |
| tfidf | 3182 | 3182 | 0.0 | 0.0 | 0.0 | n/a | n/a | n/a |

Latency is per result row, as the runner recorded it. `llm_local`/`llm_hosted` answer all four questions in one completion and the runner repeats that completion's timing on each of the four rows, so when projecting to postings the identical per-record values are counted once and one posting is one completion.

Cost note: Per-row latency is wall clock on the machine that ran the arm; arms that ran concurrently include each other's load in these numbers, so the projection is an upper bound for this machine rather than a benchmark.

### Projection to 10,000 postings/day

* `classifier_dev`: 3182 distinct records at 11.0 ms per posting -> 10,000 x 11.0 ms = 1.8 min (0.03 h) serial.
* `tfidf`: 3182 distinct records at 0.0 ms per posting -> 10,000 x 0.0 ms = 0.0 min (0.00 h) serial.
* `classifier_dev`: 3182 rows = 3182 classifications spent, one question-answer each (fraudulent=3182) over 3182 distinct record(s). Answers are cached in `results/cache.jsonl` keyed by `sha256(question, labels, instructions, tier, state)`, so reruns and `--limit` top-ups spend no additional quota. classifier.dev's free fast tier allows 20,000 classifications/day per IP, so this run used 15.9% of one day's quota; the number of records walked is set by the runner's `--limit`, and the cost of a window is the sum of its per-question rows.

## 7. Instructions ablation

n/a: no `classifier_dev_no_rubric` rows in this results directory. Produce them with `uv run python -m triage.run --arms classifier_dev --questions fraudulent --no-instructions` and re-run the evaluator; the ablation is reported whatever its direction.
## 8. Not available (n/a)

Every metric that could not be computed and every scoring caveat, with its reason. Nothing here is a substituted model, dataset or metric. The same list is written machine-readably as `n_a` in `metrics.json`.

* `fraudulent`: `dummy_stratified`: sklearn DummyClassifier(strategy='stratified', random_state=0) fitted on 12725 train+validation gold rows, predicting on the 3182 scored records; its predict_proba is one-hot on the sampled label, so it is a certainty-asserting floor
* `fraudulent`: `majority_class`: always predicts 'legitimate' (3044/3182 = 95.7% of the scored subset) and carries no distribution, so brier, ece, pr_auc and the routing sweep are null for it
* `majority_class` / `fraudulent`: `brier`: no row of majority_class carries a usable p_decision for fraudulent (confidence and probabilities are both absent or null)
* `majority_class` / `fraudulent`: `ece`: no row of majority_class carries a usable p_decision for fraudulent (confidence and probabilities are both absent or null)
* `majority_class` / `fraudulent`: `pr_auc`: no majority_class row carries p_positive for fraudulent
* `majority_class` / `fraudulent`: `precision_ci95`: no positives (or no positive predictions) in the scored subset, so the interval has no denominator
* `classifier_dev` / `salary_range_stated`: `leakage_guard`: produced no `salary_range_stated` rows, so the leakage guard does not cover it
* `tfidf` / `salary_range_stated`: `leakage_guard`: produced no `salary_range_stated` rows, so the leakage guard does not cover it

## 9. Leakage guard (`salary_range_stated`)

`salary_range` is platform metadata, not posting text, so an arm that sees only `title + description` cannot legitimately beat the majority rate. The guard is therefore deliberately one-directional:

* **PASS** when `accuracy <= floor + 3pp`: no evidence that the gold reached the model's input.
* **FAIL** when any arm is ABOVE `floor + 3pp`: that is the only direction leakage can push accuracy, so the process prints FAIL and exits non-zero.
* **BELOW** (`accuracy < floor - 3pp`) does not fail: it is below the floor, not leakage. That arm is not answering the question that was asked (see the no-rubric ablation) or is not reading the text, and it is reported with its own note.

n/a: no arm produced `salary_range_stated` rows in this results directory, so the guard has nothing to check and does not report PASS by default.
