# job-posting-triage: evaluation report

Results directory: `results`. Arms found: `classifier_dev`, `classifier_dev_no_rubric`, `tfidf`, `gliner`, `llm_local`, `llm_hosted_json_object`. Arms in this report: `classifier_dev`, `classifier_dev_no_rubric`, `tfidf`, `gliner`, `llm_local`, `llm_hosted_json_object`.

Alongside this file, `--report` writes `metrics.json`: the same numbers, from the same code path, for machine consumption (`n_a` there is the list in section 8).

## 1. What was run

| file | rows read | misparsed |
|---|---|---|
| classifier_dev.jsonl | 3147 | 0 |
| classifier_dev_no_rubric.jsonl | 3147 | 0 |
| gliner.jsonl | 3147 | 0 |
| llm_hosted_json_object.jsonl | 3147 | 0 |
| llm_local.jsonl | 3147 | 0 |
| tfidf.jsonl | 3147 | 0 |

| arm | rows | source file(s) | splits | status | reason |
|---|---|---|---|---|---|
| classifier_dev | 3147 | classifier_dev.jsonl | test | ok |  |
| classifier_dev_no_rubric | 3147 | classifier_dev_no_rubric.jsonl | test | ok |  |
| gliner | 3147 | gliner.jsonl | test | ok |  |
| llm_hosted_json_object | 3147 | llm_hosted_json_object.jsonl | test | ok |  |
| llm_local | 3147 | llm_local.jsonl | test | ok |  |
| tfidf | 3147 | tfidf.jsonl | test | ok |  |

Gold source: 18882 row(s) scored against the dataset gold from `triage/data.py`, 0 against the `gold` field of their own result row (records absent from that dataset split, e.g. synthetic fixtures).

### Frame: which records each arm answered

A question is scored only where its gold exists, so the sample size per question differs by design. What must not differ is which records an arm answered for the same question: an arm on a different set of postings has rows that are not directly comparable with the others. `shared` is the intersection across every arm, `majority set` is the record set that the largest number of arms answered, and mismatches are measured against it.

| question | arms | union | shared | majority set | records per arm | frame |
|---|---|---|---|---|---|---|
| fraudulent | 6 | 1000 | 1000 | 1000 | classifier_dev=1000, classifier_dev_no_rubric=1000, gliner=1000, llm_hosted_json_object=1000, llm_local=1000, tfidf=1000 | match |
| required_experience | 6 | 627 | 627 | 627 | classifier_dev=627, classifier_dev_no_rubric=627, gliner=627, llm_hosted_json_object=627, llm_local=627, tfidf=627 | match |
| required_education | 6 | 520 | 520 | 520 | classifier_dev=520, classifier_dev_no_rubric=520, gliner=520, llm_hosted_json_object=520, llm_local=520, tfidf=520 | match |
| salary_range_stated | 6 | 1000 | 1000 | 1000 | classifier_dev=1000, classifier_dev_no_rubric=1000, gliner=1000, llm_hosted_json_object=1000, llm_local=1000, tfidf=1000 | match |

Frame source: `results/_frame.json`, the runner's own record.

Every arm answered the same records for every question it scored: no WARN.

Questions with scored rows: `fraudulent` (8000 arm-rows), `required_experience` (5016 arm-rows), `required_education` (4160 arm-rows), `salary_range_stated` (8000 arm-rows).
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
| classifier_dev | 1000 | 0.947 | 0.642 | 0.000 | 53 | 0.312 | 0.500 | 0.226 | 0.276 | [0.31, 0.69] | [0.13, 0.36] | [0.18, 0.45] | 53 positives in this frame | 12/12/41/935 | 0 |
| classifier_dev_no_rubric | 1000 | 0.930 | 0.617 | 0.000 | 53 | 0.271 | 0.302 | 0.245 | 0.260 | [0.19, 0.45] | [0.15, 0.38] | [0.16, 0.39] | 53 positives in this frame | 13/30/40/917 | 0 |
| gliner | 1000 | 0.897 | 0.525 | 0.000 | 53 | 0.104 | 0.097 | 0.113 | 0.070 | [0.05, 0.20] | [0.05, 0.23] | [0.04, 0.18] | 53 positives in this frame | 6/56/47/891 | 0 |
| llm_hosted_json_object | 1000 | 0.948 | 0.651 | 0.005 | 53 | 0.329 | 0.600 | 0.226 | 0.282 | [0.39, 0.78] | [0.13, 0.36] | [0.19, 0.47] | 53 positives in this frame | 12/8/39/936 | 5 |
| llm_local | 1000 | 0.053 | 0.050 | 0.000 | 53 | 0.101 | 0.053 | 1.000 | 0.053 | [0.04, 0.07] | [0.93, 1.00] | [0.08, 0.13] | 53 positives in this frame | 53/947/0/0 | 0 |
| tfidf | 1000 | 0.970 | 0.842 | 0.000 | 53 | 0.700 | 0.745 | 0.660 | 0.780 | [0.60, 0.85] | [0.53, 0.77] | [0.59, 0.79] | 53 positives in this frame | 35/12/18/935 | 0 |
| majority_class | 1000 | 0.947 | 0.486 | 0.000 | 53 | 0.000 | n/a | 0.000 | n/a | n/a | [0.00, 0.07] | [0.00, 0.00] | 53 positives in this frame | 0/0/53/947 | 0 |
| dummy_stratified | 1000 | 0.906 | 0.514 | 0.000 | 53 | 0.078 | 0.082 | 0.075 | 0.055 | [0.03, 0.19] | [0.03, 0.18] | [0.02, 0.16] | 53 positives in this frame | 4/45/49/902 | 0 |

Per-class detail, over the same scored rows as the table above (all n of them, not a per-label subset): precision/recall/F1 come from sklearn with the question's declared label set, so a label nobody predicted contributes a zero to macro-F1 rather than disappearing. A precision cell is `n/a` when the arm never predicted that label: no row of this arm was predicted with this label, so precision has no denominator and is undefined.

| arm | label | precision | recall | F1 | support |
|---|---|---|---|---|---|
| classifier_dev | legitimate | 0.958 | 0.987 | 0.972 | 947 |
| classifier_dev | fraudulent | 0.500 | 0.226 | 0.312 | 53 |
| classifier_dev_no_rubric | legitimate | 0.958 | 0.968 | 0.963 | 947 |
| classifier_dev_no_rubric | fraudulent | 0.302 | 0.245 | 0.271 | 53 |
| gliner | legitimate | 0.950 | 0.941 | 0.945 | 947 |
| gliner | fraudulent | 0.097 | 0.113 | 0.104 | 53 |
| llm_hosted_json_object | legitimate | 0.960 | 0.988 | 0.974 | 947 |
| llm_hosted_json_object | fraudulent | 0.600 | 0.226 | 0.329 | 53 |
| llm_local | legitimate | n/a | 0.000 | 0.000 | 947 |
| llm_local | fraudulent | 0.053 | 1.000 | 0.101 | 53 |
| tfidf | legitimate | 0.981 | 0.987 | 0.984 | 947 |
| tfidf | fraudulent | 0.745 | 0.660 | 0.700 | 53 |
| majority_class | legitimate | 0.947 | 1.000 | 0.973 | 947 |
| majority_class | fraudulent | n/a | 0.000 | 0.000 | 53 |
| dummy_stratified | legitimate | 0.948 | 0.952 | 0.950 | 947 |
| dummy_stratified | fraudulent | 0.082 | 0.075 | 0.078 | 53 |

Confusion convention: the four cells are read over every scored row and they sum to n only after the `no_call` column, which counts rows where the arm produced no label from the declared set (an empty reply, a junk value, or a row it flagged invalid). A `no_call` row is neither a false positive nor a true negative: an empty reply made no call at all, so scoring it as a correct negative would be the exact failure this benchmark exists to catch. `no_call` rows are wrong in accuracy, they count as missed positives when their gold is the positive label (so recall and the positive-class F1 carry them), and `invalid_json_rate` counts the subset of them the arm itself flagged. Check the arithmetic with: `accuracy x n == TP + TN` and `TP + FP + FN + TN + no_call == n`.

Intervals: `precision_ci95` and `recall_ci95` are two-sided 95 percent Wilson score intervals on the counts already in the table (TP/(TP+FP) and TP/positives); `f1_ci95` is a deterministic bootstrap (2000 resamples of the scored rows with random.Random(0), percentiles by linear interpolation), so a rerun gives the same interval. `ci_note` records the frame: 53 positives in this frame.

With 53 positives in this frame, the positive-class estimates carry intervals roughly plus or minus ten percentage points wide (a little wider for precision, see the ci95 columns), the arms' intervals separate on recall and nearly separate on precision, and the same two arms were also run over the whole split where the prevalence is 4.3 percent and the point estimates matched (precision 0.74 both, recall 0.66 versus 0.65), so the ordering does not depend on the frame or on rebalancing. That whole-split comparison comes from a run outside this results directory; the intervals above are computed from the rows here.

Intervals in this frame: `classifier_dev` precision [0.31, 0.69] recall [0.13, 0.36]; `classifier_dev_no_rubric` precision [0.19, 0.45] recall [0.15, 0.38]; `gliner` precision [0.05, 0.20] recall [0.05, 0.23]; `llm_hosted_json_object` precision [0.39, 0.78] recall [0.13, 0.36]; `llm_local` precision [0.04, 0.07] recall [0.93, 1.00]; `tfidf` precision [0.60, 0.85] recall [0.53, 0.77]; `majority_class` precision n/a recall [0.00, 0.07]; `dummy_stratified` precision [0.03, 0.19] recall [0.03, 0.18].

The same caveat applies to the multi-label questions (`required_experience`, `required_education`): their scored subsets are larger, so the intervals would be narrower, and they are not quantified here because precision and recall over a multi-label argmax are not a single binomial proportion.

### `required_experience` (7-way, 6 ordered levels + not_applicable, 7 labels)

| arm | n | accuracy | macro-F1 | invalid_json_rate | MAE | +/-1 acc | MAE n | MAE excluded |
|---|---|---|---|---|---|---|---|---|
| classifier_dev | 627 | 0.555 | 0.481 | 0.000 | 0.451 | 0.929 | 548 | 79 |
| classifier_dev_no_rubric | 627 | 0.541 | 0.459 | 0.000 | 0.498 | 0.917 | 554 | 73 |
| gliner | 627 | 0.314 | 0.325 | 0.000 | 1.101 | 0.657 | 542 | 85 |
| llm_hosted_json_object | 627 | 0.536 | 0.499 | 0.005 | 0.393 | 0.937 | 476 | 151 |
| llm_local | 627 | 0.102 | 0.057 | 0.000 | 1.050 | 0.650 | 20 | 607 |
| tfidf | 627 | 0.617 | 0.429 | 0.000 | 0.459 | 0.912 | 567 | 60 |
| majority_class | 627 | 0.388 | 0.080 | 0.000 | 0.898 | 0.711 | 570 | 57 |
| dummy_stratified | 627 | 0.274 | 0.179 | 0.000 | 1.068 | 0.703 | 511 | 116 |

Per-class detail, over the same scored rows as the table above (all n of them, not a per-label subset): precision/recall/F1 come from sklearn with the question's declared label set, so a label nobody predicted contributes a zero to macro-F1 rather than disappearing. A precision cell is `n/a` when the arm never predicted that label: no row of this arm was predicted with this label, so precision has no denominator and is undefined.

| arm | label | precision | recall | F1 | support |
|---|---|---|---|---|---|
| classifier_dev | internship | 0.731 | 0.950 | 0.826 | 20 |
| classifier_dev | entry_level | 0.552 | 0.471 | 0.508 | 136 |
| classifier_dev | associate | 0.388 | 0.243 | 0.299 | 136 |
| classifier_dev | mid_senior | 0.632 | 0.868 | 0.731 | 243 |
| classifier_dev | director | 0.545 | 0.462 | 0.500 | 26 |
| classifier_dev | executive | 0.316 | 0.667 | 0.429 | 9 |
| classifier_dev | not_applicable | 0.120 | 0.053 | 0.073 | 57 |
| classifier_dev_no_rubric | internship | 0.679 | 0.950 | 0.792 | 20 |
| classifier_dev_no_rubric | entry_level | 0.551 | 0.397 | 0.462 | 136 |
| classifier_dev_no_rubric | associate | 0.321 | 0.191 | 0.240 | 136 |
| classifier_dev_no_rubric | mid_senior | 0.621 | 0.889 | 0.731 | 243 |
| classifier_dev_no_rubric | director | 0.577 | 0.577 | 0.577 | 26 |
| classifier_dev_no_rubric | executive | 0.222 | 0.667 | 0.333 | 9 |
| classifier_dev_no_rubric | not_applicable | 0.158 | 0.053 | 0.079 | 57 |
| gliner | internship | 0.696 | 0.800 | 0.744 | 20 |
| gliner | entry_level | 0.260 | 0.522 | 0.347 | 136 |
| gliner | associate | 0.266 | 0.154 | 0.195 | 136 |
| gliner | mid_senior | 0.759 | 0.272 | 0.400 | 243 |
| gliner | director | 0.333 | 0.577 | 0.423 | 26 |
| gliner | executive | 0.067 | 0.667 | 0.121 | 9 |
| gliner | not_applicable | 0.067 | 0.035 | 0.046 | 57 |
| llm_hosted_json_object | internship | 0.792 | 0.950 | 0.864 | 20 |
| llm_hosted_json_object | entry_level | 0.520 | 0.574 | 0.545 | 136 |
| llm_hosted_json_object | associate | 0.486 | 0.125 | 0.199 | 136 |
| llm_hosted_json_object | mid_senior | 0.692 | 0.794 | 0.739 | 243 |
| llm_hosted_json_object | director | 0.538 | 0.538 | 0.538 | 26 |
| llm_hosted_json_object | executive | 0.500 | 0.444 | 0.471 | 9 |
| llm_hosted_json_object | not_applicable | 0.108 | 0.193 | 0.138 | 57 |
| llm_local | internship | 0.250 | 0.050 | 0.083 | 20 |
| llm_local | entry_level | 0.400 | 0.015 | 0.028 | 136 |
| llm_local | associate | 1.000 | 0.015 | 0.029 | 136 |
| llm_local | mid_senior | 0.750 | 0.012 | 0.024 | 243 |
| llm_local | director | 0.250 | 0.038 | 0.067 | 26 |
| llm_local | executive | 0.000 | 0.000 | 0.000 | 9 |
| llm_local | not_applicable | 0.091 | 0.965 | 0.166 | 57 |
| tfidf | internship | 0.900 | 0.450 | 0.600 | 20 |
| tfidf | entry_level | 0.683 | 0.632 | 0.656 | 136 |
| tfidf | associate | 0.513 | 0.426 | 0.466 | 136 |
| tfidf | mid_senior | 0.605 | 0.889 | 0.720 | 243 |
| tfidf | director | 1.000 | 0.077 | 0.143 | 26 |
| tfidf | executive | n/a | 0.000 | 0.000 | 9 |
| tfidf | not_applicable | 0.842 | 0.281 | 0.421 | 57 |
| majority_class | internship | n/a | 0.000 | 0.000 | 20 |
| majority_class | entry_level | n/a | 0.000 | 0.000 | 136 |
| majority_class | associate | n/a | 0.000 | 0.000 | 136 |
| majority_class | mid_senior | 0.388 | 1.000 | 0.559 | 243 |
| majority_class | director | n/a | 0.000 | 0.000 | 26 |
| majority_class | executive | n/a | 0.000 | 0.000 | 9 |
| majority_class | not_applicable | n/a | 0.000 | 0.000 | 57 |
| dummy_stratified | internship | 0.032 | 0.050 | 0.039 | 20 |
| dummy_stratified | entry_level | 0.273 | 0.279 | 0.276 | 136 |
| dummy_stratified | associate | 0.230 | 0.235 | 0.233 | 136 |
| dummy_stratified | mid_senior | 0.419 | 0.370 | 0.393 | 243 |
| dummy_stratified | director | 0.034 | 0.038 | 0.036 | 26 |
| dummy_stratified | executive | 0.167 | 0.111 | 0.133 | 9 |
| dummy_stratified | not_applicable | 0.132 | 0.158 | 0.144 | 57 |

MAE and +/-1 use `triage.schema.ORDINAL` (`internship=0`, `entry_level=1`, `associate=2`, `mid_senior=3`, `director=4`, `executive=5`); rows whose gold or prediction is `not_applicable`, empty or outside the label set are excluded from both and counted in *MAE excluded*. `not_applicable` still counts in accuracy and macro-F1.

### `required_education` (7-way categorical, 7 labels)

| arm | n | accuracy | macro-F1 | invalid_json_rate |
|---|---|---|---|---|
| classifier_dev | 520 | 0.158 | 0.097 | 0.000 |
| classifier_dev_no_rubric | 520 | 0.179 | 0.123 | 0.000 |
| gliner | 520 | 0.117 | 0.092 | 0.000 |
| llm_hosted_json_object | 520 | 0.148 | 0.069 | 0.006 |
| llm_local | 520 | 0.117 | 0.034 | 0.000 |
| tfidf | 520 | 0.725 | 0.359 | 0.000 |
| majority_class | 520 | 0.544 | 0.101 | 0.000 |
| dummy_stratified | 520 | 0.333 | 0.150 | 0.000 |

Per-class detail, over the same scored rows as the table above (all n of them, not a per-label subset): precision/recall/F1 come from sklearn with the question's declared label set, so a label nobody predicted contributes a zero to macro-F1 rather than disappearing. A precision cell is `n/a` when the arm never predicted that label: no row of this arm was predicted with this label, so precision has no denominator and is undefined.

| arm | label | precision | recall | F1 | support |
|---|---|---|---|---|---|
| classifier_dev | bachelors | 0.750 | 0.053 | 0.099 | 283 |
| classifier_dev | high_school | 0.900 | 0.084 | 0.154 | 107 |
| classifier_dev | unspecified | 0.115 | 0.982 | 0.207 | 57 |
| classifier_dev | masters | 0.000 | 0.000 | 0.000 | 31 |
| classifier_dev | associate | n/a | 0.000 | 0.000 | 10 |
| classifier_dev | certification | 0.333 | 0.071 | 0.118 | 14 |
| classifier_dev | other | 1.000 | 0.056 | 0.105 | 18 |
| classifier_dev_no_rubric | bachelors | 0.870 | 0.071 | 0.131 | 283 |
| classifier_dev_no_rubric | high_school | 0.833 | 0.187 | 0.305 | 107 |
| classifier_dev_no_rubric | unspecified | 0.129 | 0.842 | 0.224 | 57 |
| classifier_dev_no_rubric | masters | 0.000 | 0.000 | 0.000 | 31 |
| classifier_dev_no_rubric | associate | 0.000 | 0.000 | 0.000 | 10 |
| classifier_dev_no_rubric | certification | 0.333 | 0.071 | 0.118 | 14 |
| classifier_dev_no_rubric | other | 0.053 | 0.222 | 0.086 | 18 |
| gliner | bachelors | 0.833 | 0.035 | 0.068 | 283 |
| gliner | high_school | 0.750 | 0.056 | 0.104 | 107 |
| gliner | unspecified | 0.158 | 0.614 | 0.251 | 57 |
| gliner | masters | 0.200 | 0.032 | 0.056 | 31 |
| gliner | associate | 0.000 | 0.000 | 0.000 | 10 |
| gliner | certification | 0.056 | 0.286 | 0.093 | 14 |
| gliner | other | 0.042 | 0.278 | 0.073 | 18 |
| llm_hosted_json_object | bachelors | 0.783 | 0.064 | 0.118 | 283 |
| llm_hosted_json_object | high_school | 0.600 | 0.028 | 0.054 | 107 |
| llm_hosted_json_object | unspecified | 0.114 | 0.965 | 0.203 | 57 |
| llm_hosted_json_object | masters | 0.000 | 0.000 | 0.000 | 31 |
| llm_hosted_json_object | associate | n/a | 0.000 | 0.000 | 10 |
| llm_hosted_json_object | certification | 0.000 | 0.000 | 0.000 | 14 |
| llm_hosted_json_object | other | 1.000 | 0.056 | 0.105 | 18 |
| llm_local | bachelors | 1.000 | 0.011 | 0.021 | 283 |
| llm_local | high_school | 1.000 | 0.009 | 0.019 | 107 |
| llm_local | unspecified | 0.110 | 1.000 | 0.199 | 57 |
| llm_local | masters | n/a | 0.000 | 0.000 | 31 |
| llm_local | associate | n/a | 0.000 | 0.000 | 10 |
| llm_local | certification | n/a | 0.000 | 0.000 | 14 |
| llm_local | other | n/a | 0.000 | 0.000 | 18 |
| tfidf | bachelors | 0.701 | 0.979 | 0.817 | 283 |
| tfidf | high_school | 0.824 | 0.785 | 0.804 | 107 |
| tfidf | unspecified | 0.611 | 0.193 | 0.293 | 57 |
| tfidf | masters | 1.000 | 0.032 | 0.062 | 31 |
| tfidf | associate | 1.000 | 0.200 | 0.333 | 10 |
| tfidf | certification | n/a | 0.000 | 0.000 | 14 |
| tfidf | other | 1.000 | 0.111 | 0.200 | 18 |
| majority_class | bachelors | 0.544 | 1.000 | 0.705 | 283 |
| majority_class | high_school | n/a | 0.000 | 0.000 | 107 |
| majority_class | unspecified | n/a | 0.000 | 0.000 | 57 |
| majority_class | masters | n/a | 0.000 | 0.000 | 31 |
| majority_class | associate | n/a | 0.000 | 0.000 | 10 |
| majority_class | certification | n/a | 0.000 | 0.000 | 14 |
| majority_class | other | n/a | 0.000 | 0.000 | 18 |
| dummy_stratified | bachelors | 0.546 | 0.484 | 0.513 | 283 |
| dummy_stratified | high_school | 0.221 | 0.234 | 0.227 | 107 |
| dummy_stratified | unspecified | 0.096 | 0.140 | 0.114 | 57 |
| dummy_stratified | masters | 0.043 | 0.032 | 0.037 | 31 |
| dummy_stratified | associate | 0.000 | 0.000 | 0.000 | 10 |
| dummy_stratified | certification | 0.182 | 0.143 | 0.160 | 14 |
| dummy_stratified | other | 0.000 | 0.000 | 0.000 | 18 |

### `salary_range_stated` (control, 2 labels)

| arm | n | accuracy | macro-F1 | invalid_json_rate | majority rate | accuracy - majority (pp) |
|---|---|---|---|---|---|---|
| classifier_dev | 1000 | 0.803 | 0.522 | 0.000 | 0.822 | -1.9 |
| classifier_dev_no_rubric | 1000 | 0.245 | 0.237 | 0.000 | 0.822 | -57.7 |
| gliner | 1000 | 0.265 | 0.264 | 0.000 | 0.822 | -55.7 |
| llm_hosted_json_object | 1000 | 0.798 | 0.516 | 0.005 | 0.822 | -2.4 |
| llm_local | 1000 | 0.795 | 0.506 | 0.000 | 0.822 | -2.7 |
| tfidf | 1000 | 0.834 | 0.522 | 0.000 | 0.822 | 1.2 |
| majority_class | 1000 | 0.822 | 0.451 | 0.000 | 0.822 | 0.0 |
| dummy_stratified | 1000 | 0.712 | 0.506 | 0.000 | 0.822 | -11.0 |

Per-class detail, over the same scored rows as the table above (all n of them, not a per-label subset): precision/recall/F1 come from sklearn with the question's declared label set, so a label nobody predicted contributes a zero to macro-F1 rather than disappearing. A precision cell is `n/a` when the arm never predicted that label: no row of this arm was predicted with this label, so precision has no denominator and is undefined.

| arm | label | precision | recall | F1 | support |
|---|---|---|---|---|---|
| classifier_dev | stated | 0.327 | 0.101 | 0.155 | 178 |
| classifier_dev | not_stated | 0.831 | 0.955 | 0.889 | 822 |
| classifier_dev_no_rubric | stated | 0.187 | 0.972 | 0.314 | 178 |
| classifier_dev_no_rubric | not_stated | 0.935 | 0.088 | 0.160 | 822 |
| gliner | stated | 0.179 | 0.871 | 0.297 | 178 |
| gliner | not_stated | 0.827 | 0.134 | 0.230 | 822 |
| llm_hosted_json_object | stated | 0.315 | 0.096 | 0.147 | 178 |
| llm_hosted_json_object | not_stated | 0.830 | 0.950 | 0.886 | 822 |
| llm_local | stated | 0.263 | 0.084 | 0.128 | 178 |
| llm_local | not_stated | 0.827 | 0.949 | 0.884 | 822 |
| tfidf | stated | 0.929 | 0.073 | 0.135 | 178 |
| tfidf | not_stated | 0.833 | 0.999 | 0.908 | 822 |
| majority_class | stated | n/a | 0.000 | 0.000 | 178 |
| majority_class | not_stated | 0.822 | 1.000 | 0.902 | 822 |
| dummy_stratified | stated | 0.188 | 0.185 | 0.186 | 178 |
| dummy_stratified | not_stated | 0.824 | 0.826 | 0.825 | 822 |

Control question: the answer is platform metadata, so an arm reading only the posting cannot beat the majority rate. See the leakage guard in section 9.

## 4. Calibration

Brier and ECE are computed on `p_decision` (section 2), which is why an arm that cannot state a probability shows `n/a` rather than a substituted zero.

| arm | question | n scored | n with p_decision | excluded (null p_decision) | of which invalid | excluded but valid | Brier | ECE | invalid_json_rate |
|---|---|---|---|---|---|---|---|---|---|
| classifier_dev | fraudulent | 1000 | 1000 | 0 | 0 | 0 | 0.054 | 0.046 | 0.000 |
| classifier_dev_no_rubric | fraudulent | 1000 | 1000 | 0 | 0 | 0 | 0.073 | 0.070 | 0.000 |
| gliner | fraudulent | 1000 | 1000 | 0 | 0 | 0 | 0.094 | 0.082 | 0.000 |
| llm_hosted_json_object | fraudulent | 1000 | 995 | 5 | 5 | 0 | 0.047 | 0.015 | 0.005 |
| llm_local | fraudulent | 1000 | 1000 | 0 | 0 | 0 | 0.947 | 0.947 | 0.000 |
| tfidf | fraudulent | 1000 | 1000 | 0 | 0 | 0 | 0.033 | 0.033 | 0.000 |
| majority_class | fraudulent | 1000 | 0 | 1000 | 0 | 1000 | n/a | n/a | 0.000 |
| dummy_stratified | fraudulent | 1000 | 1000 | 0 | 0 | 0 | 0.094 | 0.094 | 0.000 |
| classifier_dev | required_experience | 627 | 627 | 0 | 0 | 0 | 0.256 | 0.183 | 0.000 |
| classifier_dev_no_rubric | required_experience | 627 | 627 | 0 | 0 | 0 | 0.274 | 0.204 | 0.000 |
| gliner | required_experience | 627 | 627 | 0 | 0 | 0 | 0.450 | 0.465 | 0.000 |
| llm_hosted_json_object | required_experience | 627 | 624 | 3 | 3 | 0 | 0.273 | 0.181 | 0.005 |
| llm_local | required_experience | 627 | 627 | 0 | 0 | 0 | 0.204 | 0.241 | 0.000 |
| tfidf | required_experience | 627 | 627 | 0 | 0 | 0 | 0.206 | 0.094 | 0.000 |
| majority_class | required_experience | 627 | 0 | 627 | 0 | 627 | n/a | n/a | 0.000 |
| dummy_stratified | required_experience | 627 | 627 | 0 | 0 | 0 | 0.726 | 0.726 | 0.000 |
| classifier_dev | required_education | 520 | 520 | 0 | 0 | 0 | 0.814 | 0.810 | 0.000 |
| classifier_dev_no_rubric | required_education | 520 | 520 | 0 | 0 | 0 | 0.202 | 0.242 | 0.000 |
| gliner | required_education | 520 | 520 | 0 | 0 | 0 | 0.391 | 0.506 | 0.000 |
| llm_hosted_json_object | required_education | 520 | 517 | 3 | 3 | 0 | 0.759 | 0.791 | 0.006 |
| llm_local | required_education | 520 | 520 | 0 | 0 | 0 | 0.192 | 0.223 | 0.000 |
| tfidf | required_education | 520 | 520 | 0 | 0 | 0 | 0.172 | 0.071 | 0.000 |
| majority_class | required_education | 520 | 0 | 520 | 0 | 520 | n/a | n/a | 0.000 |
| dummy_stratified | required_education | 520 | 520 | 0 | 0 | 0 | 0.667 | 0.667 | 0.000 |
| classifier_dev | salary_range_stated | 1000 | 1000 | 0 | 0 | 0 | 0.194 | 0.196 | 0.000 |
| classifier_dev_no_rubric | salary_range_stated | 1000 | 1000 | 0 | 0 | 0 | 0.206 | 0.123 | 0.000 |
| gliner | salary_range_stated | 1000 | 1000 | 0 | 0 | 0 | 0.660 | 0.659 | 0.000 |
| llm_hosted_json_object | salary_range_stated | 1000 | 995 | 5 | 5 | 0 | 0.191 | 0.182 | 0.005 |
| llm_local | salary_range_stated | 1000 | 1000 | 0 | 0 | 0 | 0.606 | 0.589 | 0.000 |
| tfidf | salary_range_stated | 1000 | 1000 | 0 | 0 | 0 | 0.118 | 0.083 | 0.000 |
| majority_class | salary_range_stated | 1000 | 0 | 1000 | 0 | 1000 | n/a | n/a | 0.000 |
| dummy_stratified | salary_range_stated | 1000 | 1000 | 0 | 0 | 0 | 0.288 | 0.288 | 0.000 |
| **classifier_dev (all questions)** | all | 3147 | 3147 | 0 | 0 | 0 | 0.264 | 0.237 | 0.000 |
| **classifier_dev_no_rubric (all questions)** | all | 3147 | 3147 | 0 | 0 | 0 | 0.176 | 0.100 | 0.000 |
| **tfidf (all questions)** | all | 3147 | 3147 | 0 | 0 | 0 | 0.117 | 0.045 | 0.000 |
| **gliner (all questions)** | all | 3147 | 3147 | 0 | 0 | 0 | 0.394 | 0.410 | 0.000 |
| **llm_local (all questions)** | all | 3147 | 3147 | 0 | 0 | 0 | 0.566 | 0.535 | 0.000 |
| **dummy_stratified (all questions)** | all | 3147 | 3147 | 0 | 0 | 0 | 0.376 | 0.376 | 0.000 |
| **llm_hosted_json_object (all questions)** | all | 3147 | 3131 | 16 | 16 | 0 | 0.255 | 0.219 | 0.005 |
| **majority_class (all questions)** | all | 3147 | 0 | 3147 | 0 | 3147 | n/a | n/a | 0.000 |

Excluded rows are never dropped silently. *of which invalid* counts rows the arm itself marked `invalid` (its own output failed to parse or validate) that also carry no probability; *excluded but valid* counts rows that answered normally yet state no probability at all, which is why Brier/ECE cannot include them. Both sets still count as wrong in accuracy, macro-F1 and every other scored metric, and the n/a list in section 8 names them per arm and question.

Invalid rows, named:

* `llm_hosted_json_object`: 16 of 3147 rows are `invalid` (0.5% of its rows), and 16 of those are excluded from calibration; arm-reported reasons: 16 x 'output budget of 4096 tokens exhausted before an answer (finish_reason=length); raise --max-tokens'.

### Reliability bins, `fraudulent`

| arm | bin | n | mean p_decision | empirical accuracy | gap |
|---|---|---|---|---|---|
| classifier_dev | [0.0, 0.1) | 5 | 0.060 | 0.800 | 0.740 |
| classifier_dev | [0.1, 0.2) | 7 | 0.144 | 0.429 | 0.284 |
| classifier_dev | [0.2, 0.3) | 7 | 0.256 | 0.571 | 0.316 |
| classifier_dev | [0.3, 0.4) | 6 | 0.347 | 0.333 | 0.013 |
| classifier_dev | [0.4, 0.5) | 5 | 0.440 | 1.000 | 0.560 |
| classifier_dev | [0.5, 0.6) | 6 | 0.532 | 1.000 | 0.468 |
| classifier_dev | [0.6, 0.7) | 14 | 0.654 | 0.857 | 0.203 |
| classifier_dev | [0.7, 0.8) | 17 | 0.747 | 0.882 | 0.135 |
| classifier_dev | [0.8, 0.9) | 39 | 0.854 | 0.923 | 0.069 |
| classifier_dev | [0.9, 1.0] | 894 | 0.989 | 0.962 | 0.028 |
| classifier_dev_no_rubric | [0.0, 0.1) | 9 | 0.032 | 0.333 | 0.301 |
| classifier_dev_no_rubric | [0.1, 0.2) | 9 | 0.156 | 0.333 | 0.178 |
| classifier_dev_no_rubric | [0.2, 0.3) | 14 | 0.243 | 0.786 | 0.543 |
| classifier_dev_no_rubric | [0.3, 0.4) | 17 | 0.356 | 0.647 | 0.291 |
| classifier_dev_no_rubric | [0.4, 0.5) | 18 | 0.447 | 0.722 | 0.276 |
| classifier_dev_no_rubric | [0.5, 0.6) | 25 | 0.559 | 0.840 | 0.281 |
| classifier_dev_no_rubric | [0.6, 0.7) | 33 | 0.646 | 0.909 | 0.263 |
| classifier_dev_no_rubric | [0.7, 0.8) | 67 | 0.749 | 0.955 | 0.206 |
| classifier_dev_no_rubric | [0.8, 0.9) | 167 | 0.850 | 0.952 | 0.102 |
| classifier_dev_no_rubric | [0.9, 1.0] | 641 | 0.961 | 0.959 | 0.002 |
| gliner | [0.5, 0.6) | 15 | 0.550 | 0.733 | 0.183 |
| gliner | [0.6, 0.7) | 13 | 0.664 | 0.308 | 0.356 |
| gliner | [0.7, 0.8) | 13 | 0.763 | 0.769 | 0.006 |
| gliner | [0.8, 0.9) | 29 | 0.861 | 0.793 | 0.068 |
| gliner | [0.9, 1.0] | 930 | 0.991 | 0.913 | 0.078 |
| llm_hosted_json_object | [0.6, 0.7) | 5 | 0.620 | 0.800 | 0.180 |
| llm_hosted_json_object | [0.7, 0.8) | 45 | 0.723 | 0.867 | 0.144 |
| llm_hosted_json_object | [0.8, 0.9) | 69 | 0.835 | 0.899 | 0.064 |
| llm_hosted_json_object | [0.9, 1.0] | 876 | 0.959 | 0.962 | 0.004 |
| llm_local | [0.9, 1.0] | 1000 | 1.000 | 0.053 | 0.947 |
| tfidf | [0.1, 0.2) | 21 | 0.138 | 0.429 | 0.291 |
| tfidf | [0.2, 0.3) | 10 | 0.263 | 1.000 | 0.737 |
| tfidf | [0.3, 0.4) | 4 | 0.344 | 1.000 | 0.656 |
| tfidf | [0.4, 0.5) | 2 | 0.411 | 1.000 | 0.589 |
| tfidf | [0.5, 0.6) | 1 | 0.571 | 1.000 | 0.429 |
| tfidf | [0.6, 0.7) | 5 | 0.648 | 1.000 | 0.352 |
| tfidf | [0.7, 0.8) | 3 | 0.772 | 1.000 | 0.228 |
| tfidf | [0.8, 0.9) | 1 | 0.824 | 1.000 | 0.176 |
| tfidf | [0.9, 1.0] | 953 | 0.968 | 0.981 | 0.014 |
| dummy_stratified | [0.9, 1.0] | 1000 | 1.000 | 0.906 | 0.094 |

Empty bins are omitted from the table; the ECE above sums over all ten bins, empty ones contributing zero.

## 5. Routing (`fraudulent`)

Acting means **auto-flag as scam**: the record leaves the human review queue and is treated as fraud. The sweep runs over `p_positive` (section 2). Coverage is the share of the evaluated records acted on; the remainder is what a human still has to read. A row with no `p_positive` can never be acted on, so it counts against coverage.

| arm | tau | acted | coverage | TP (scams flagged) | FP (legitimate flagged) | precision | recall | FN (scams missed) | positives acted/total | positives acted rate | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classifier_dev | 0.5 | 24 | 2.4% | 12 | 12 | 0.500 | 0.226 | 41 | 12/53 | 0.226 |  |
| classifier_dev | 0.7 | 12 | 1.2% | 10 | 2 | 0.833 | 0.189 | 43 | 10/53 | 0.189 |  |
| classifier_dev | 0.9 | 8 | 0.8% | 7 | 1 | 0.875 | 0.132 | 46 | 7/53 | 0.132 |  |
| classifier_dev | 0.95 | 6 | 0.6% | 6 | 0 | 1.000 | 0.113 | 47 | 6/53 | 0.113 |  |
| classifier_dev_no_rubric | 0.5 | 44 | 4.4% | 13 | 31 | 0.295 | 0.245 | 40 | 13/53 | 0.245 |  |
| classifier_dev_no_rubric | 0.7 | 22 | 2.2% | 11 | 11 | 0.500 | 0.208 | 42 | 11/53 | 0.208 |  |
| classifier_dev_no_rubric | 0.9 | 11 | 1.1% | 7 | 4 | 0.636 | 0.132 | 46 | 7/53 | 0.132 |  |
| classifier_dev_no_rubric | 0.95 | 8 | 0.8% | 6 | 2 | 0.750 | 0.113 | 47 | 6/53 | 0.113 |  |
| gliner | 0.5 | 62 | 6.2% | 6 | 56 | 0.097 | 0.113 | 47 | 6/53 | 0.113 |  |
| gliner | 0.7 | 49 | 4.9% | 6 | 43 | 0.122 | 0.113 | 47 | 6/53 | 0.113 |  |
| gliner | 0.9 | 39 | 3.9% | 4 | 35 | 0.103 | 0.075 | 49 | 4/53 | 0.075 |  |
| gliner | 0.95 | 34 | 3.4% | 3 | 31 | 0.088 | 0.057 | 50 | 3/53 | 0.057 |  |
| llm_hosted_json_object | 0.5 | 20 | 2.0% | 12 | 8 | 0.600 | 0.226 | 41 | 12/53 | 0.226 |  |
| llm_hosted_json_object | 0.7 | 20 | 2.0% | 12 | 8 | 0.600 | 0.226 | 41 | 12/53 | 0.226 |  |
| llm_hosted_json_object | 0.9 | 7 | 0.7% | 7 | 0 | 1.000 | 0.132 | 46 | 7/53 | 0.132 |  |
| llm_hosted_json_object | 0.95 | 5 | 0.5% | 5 | 0 | 1.000 | 0.094 | 48 | 5/53 | 0.094 |  |
| llm_local | 0.5 | 1000 | 100.0% | 53 | 947 | 0.053 | 1.000 | 0 | 53/53 | 1.000 |  |
| llm_local | 0.7 | 1000 | 100.0% | 53 | 947 | 0.053 | 1.000 | 0 | 53/53 | 1.000 |  |
| llm_local | 0.9 | 1000 | 100.0% | 53 | 947 | 0.053 | 1.000 | 0 | 53/53 | 1.000 |  |
| llm_local | 0.95 | 1000 | 100.0% | 53 | 947 | 0.053 | 1.000 | 0 | 53/53 | 1.000 |  |
| tfidf | 0.5 | 10 | 1.0% | 10 | 0 | 1.000 | 0.189 | 43 | 10/53 | 0.189 |  |
| tfidf | 0.7 | 4 | 0.4% | 4 | 0 | 1.000 | 0.075 | 49 | 4/53 | 0.075 |  |
| tfidf | 0.9 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 53 | 0/53 | 0.000 | nothing reaches p_positive >= 0.9 |
| tfidf | 0.95 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 53 | 0/53 | 0.000 | nothing reaches p_positive >= 0.95 |
| majority_class | 0.5 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 53 | 0/53 | 0.000 | nothing reaches p_positive >= 0.5 (1000 row(s) carry no p_positive) |
| majority_class | 0.7 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 53 | 0/53 | 0.000 | nothing reaches p_positive >= 0.7 (1000 row(s) carry no p_positive) |
| majority_class | 0.9 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 53 | 0/53 | 0.000 | nothing reaches p_positive >= 0.9 (1000 row(s) carry no p_positive) |
| majority_class | 0.95 | 0 | 0.0% | 0 | 0 | n/a | 0.000 | 53 | 0/53 | 0.000 | nothing reaches p_positive >= 0.95 (1000 row(s) carry no p_positive) |
| dummy_stratified | 0.5 | 49 | 4.9% | 4 | 45 | 0.082 | 0.075 | 49 | 4/53 | 0.075 |  |
| dummy_stratified | 0.7 | 49 | 4.9% | 4 | 45 | 0.082 | 0.075 | 49 | 4/53 | 0.075 |  |
| dummy_stratified | 0.9 | 49 | 4.9% | 4 | 45 | 0.082 | 0.075 | 49 | 4/53 | 0.075 |  |
| dummy_stratified | 0.95 | 49 | 4.9% | 4 | 45 | 0.082 | 0.075 | 49 | 4/53 | 0.075 |  |

`FN (scams missed)` counts positives below the threshold: the frauds the acted-on set does not catch. `positives acted/total` is the share of the positives that survive the filter, given as rate and count. At a threshold above every score nothing is acted on: coverage 0, precision and recall `n/a` with the reason stated, and no division by zero.

classifier.dev is a shared service and its probabilities shift slightly between fetches of the same posting, so the threshold-dependent cells above can move by one record between reruns while the label-based tables in section 3 do not. Treat the routing cells as indicative at the high thresholds, where the `acted` column shows how few postings carry the decision.

## 6. Cost and latency

| arm | rows | rows with latency | p50 latency_ms | p95 latency_ms | mean latency_ms | mean tokens_in | mean tokens_out | rows with token counts |
|---|---|---|---|---|---|---|---|---|
| classifier_dev | 3147 | 3147 | 2.9 | 48.3 | 17.3 | n/a | n/a | n/a |
| classifier_dev_no_rubric | 3147 | 3147 | 6.5 | 12.5 | 8.2 | n/a | n/a | n/a |
| tfidf | 3147 | 3147 | 0.0 | 0.0 | 0.0 | n/a | n/a | n/a |
| gliner | 3147 | 3147 | 516.5 | 1096.4 | 605.6 | 253.9 | n/a | 3147 |
| llm_local | 3147 | 3147 | 12267.2 | 17674.0 | 12983.5 | 447.7 | 90.9 | 3147 |
| llm_hosted_json_object | 3147 | 3147 | 4051.0 | 11864.8 | 9687.1 | 647.0 | 795.7 | 3131 |

Latency is per result row, as the runner recorded it. `llm_local`/`llm_hosted` answer all four questions in one completion and the runner repeats that completion's timing on each of the four rows, so when projecting to postings the identical per-record values are counted once and one posting is one completion.

Cost note: Per-row latency is wall clock on the machine that ran the arm; arms that ran concurrently include each other's load in these numbers, so the projection is an upper bound for this machine rather than a benchmark.

### Projection to 10,000 postings/day

* `classifier_dev`: 1000 distinct records at 54.4 ms per posting -> 10,000 x 54.4 ms = 9.1 min (0.15 h) serial.
* `classifier_dev_no_rubric`: 1000 distinct records at 19.4 ms per posting -> 10,000 x 19.4 ms = 3.2 min (0.05 h) serial.
* `tfidf`: 1000 distinct records at 0.0 ms per posting -> 10,000 x 0.0 ms = 0.0 min (0.00 h) serial.
* `gliner`: 1000 distinct records at 1905.9 ms per posting -> 10,000 x 1905.9 ms = 317.6 min (5.29 h) serial.
* `llm_local`: 1000 distinct records at 12992.5 ms per posting -> 10,000 x 12992.5 ms = 2165.4 min (36.09 h) serial, 908,799 completion tokens at 90.9 tokens per row.
* `llm_hosted_json_object`: 1000 distinct records at 9925.2 ms per posting -> 10,000 x 9925.2 ms = 1654.2 min (27.57 h) serial, 7,956,838 completion tokens at 795.7 tokens per row.
* `classifier_dev`: 3147 rows = 3147 classifications spent, one question-answer each (fraudulent=1000, required_education=520, required_experience=627, salary_range_stated=1000) over 1000 distinct record(s). Answers are cached in `results/cache.jsonl` keyed by `sha256(question, labels, instructions, tier, state)`, so reruns and `--limit` top-ups spend no additional quota. classifier.dev's free fast tier allows 20,000 classifications/day per IP, so this run used 15.7% of one day's quota; the number of records walked is set by the runner's `--limit`, and the cost of a window is the sum of its per-question rows.

## 7. Instructions ablation

Both variants exist. `classifier_dev` sends the `instructions` rubric field to classifier.dev; `classifier_dev_no_rubric` drops it. Accuracy and F1 are shown side by side for the questions they cover, irrespective of which way the difference goes.

| question | arm | n | accuracy | macro-F1 | F1(positive) | Brier |
|---|---|---|---|---|---|---|
| fraudulent | classifier_dev | 1000 | 0.947 | 0.642 | 0.312 | 0.054 |
| fraudulent | classifier_dev_no_rubric | 1000 | 0.930 | 0.617 | 0.271 | 0.073 |
| required_experience | classifier_dev | 627 | 0.555 | 0.481 | n/a | 0.256 |
| required_experience | classifier_dev_no_rubric | 627 | 0.541 | 0.459 | n/a | 0.274 |
| required_education | classifier_dev | 520 | 0.158 | 0.097 | n/a | 0.814 |
| required_education | classifier_dev_no_rubric | 520 | 0.179 | 0.123 | n/a | 0.202 |
| salary_range_stated | classifier_dev | 1000 | 0.803 | 0.522 | n/a | 0.194 |
| salary_range_stated | classifier_dev_no_rubric | 1000 | 0.245 | 0.237 | n/a | 0.206 |

* `fraudulent`: dropping the rubric changes accuracy by -1.7pp (0.947 -> 0.930).
* `required_experience`: dropping the rubric changes accuracy by -1.4pp (0.555 -> 0.541).
* `required_education`: dropping the rubric changes accuracy by +2.1pp (0.158 -> 0.179).
* `salary_range_stated`: dropping the rubric changes accuracy by -55.8pp (0.803 -> 0.245).

## 8. Not available (n/a)

Every metric that could not be computed and every scoring caveat, with its reason. Nothing here is a substituted model, dataset or metric. The same list is written machine-readably as `n_a` in `metrics.json`.

* `fraudulent`: `dummy_stratified`: sklearn DummyClassifier(strategy='stratified', random_state=0) fitted on 12725 train+validation gold rows, predicting on the 1000 scored records; its predict_proba is one-hot on the sampled label, so it is a certainty-asserting floor
* `fraudulent`: `majority_class`: always predicts 'legitimate' (947/1000 = 94.7% of the scored subset) and carries no distribution, so brier, ece, pr_auc and the routing sweep are null for it
* `llm_hosted_json_object` / `fraudulent`: `invalid_json_rate`: invalid_json_rate = 0.5%: 5 of 1000 rows failed the arm's own output validation and are scored wrong (arm-reported reasons: 5 x 'output budget of 4096 tokens exhausted before an answer (finish_reason=length); raise --max-tokens'); 5 of them are also excluded from Brier/ECE because they carry no probability
* `llm_hosted_json_object` / `fraudulent`: `unparseable_value`: 5 row(s) carry a value outside the declared label set; they are scored as wrong, and a per-class table cannot place them in any class
* `majority_class` / `fraudulent`: `brier`: no row of majority_class carries a usable p_decision for fraudulent (confidence and probabilities are both absent or null)
* `majority_class` / `fraudulent`: `ece`: no row of majority_class carries a usable p_decision for fraudulent (confidence and probabilities are both absent or null)
* `majority_class` / `fraudulent`: `pr_auc`: no majority_class row carries p_positive for fraudulent
* `majority_class` / `fraudulent`: `precision_ci95`: no positives (or no positive predictions) in the scored subset, so the interval has no denominator
* `required_education`: `dummy_stratified`: sklearn DummyClassifier(strategy='stratified', random_state=0) fitted on 6832 train+validation gold rows, predicting on the 520 scored records; its predict_proba is one-hot on the sampled label, so it is a certainty-asserting floor
* `required_education`: `majority_class`: always predicts 'bachelors' (283/520 = 54.4% of the scored subset) and carries no distribution, so brier, ece, pr_auc and the routing sweep are null for it
* `classifier_dev` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_education`: `invalid_json_rate`: invalid_json_rate = 0.6%: 3 of 520 rows failed the arm's own output validation and are scored wrong (arm-reported reasons: 3 x 'output budget of 4096 tokens exhausted before an answer (finish_reason=length); raise --max-tokens'); 3 of them are also excluded from Brier/ECE because they carry no probability
* `llm_hosted_json_object` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_education`: `unparseable_value`: 3 row(s) carry a value outside the declared label set; they are scored as wrong, and a per-class table cannot place them in any class
* `llm_local` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_education`: `brier`: no row of majority_class carries a usable p_decision for required_education (confidence and probabilities are both absent or null)
* `majority_class` / `required_education`: `ece`: no row of majority_class carries a usable p_decision for required_education (confidence and probabilities are both absent or null)
* `majority_class` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_education`: `f1_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_education`: `pr_auc`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_education`: `precision_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_education`: `recall_positive`: required_education is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `required_experience`: `dummy_stratified`: sklearn DummyClassifier(strategy='stratified', random_state=0) fitted on 8022 train+validation gold rows, predicting on the 627 scored records; its predict_proba is one-hot on the sampled label, so it is a certainty-asserting floor
* `required_experience`: `majority_class`: always predicts 'mid_senior' (243/627 = 38.8% of the scored subset) and carries no distribution, so brier, ece, pr_auc and the routing sweep are null for it
* `classifier_dev` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_experience`: `invalid_json_rate`: invalid_json_rate = 0.5%: 3 of 627 rows failed the arm's own output validation and are scored wrong (arm-reported reasons: 3 x 'output budget of 4096 tokens exhausted before an answer (finish_reason=length); raise --max-tokens'); 3 of them are also excluded from Brier/ECE because they carry no probability
* `llm_hosted_json_object` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `required_experience`: `unparseable_value`: 3 row(s) carry a value outside the declared label set; they are scored as wrong, and a per-class table cannot place them in any class
* `llm_local` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_experience`: `brier`: no row of majority_class carries a usable p_decision for required_experience (confidence and probabilities are both absent or null)
* `majority_class` / `required_experience`: `ece`: no row of majority_class carries a usable p_decision for required_experience (confidence and probabilities are both absent or null)
* `majority_class` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_experience`: `f1_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_experience`: `pr_auc`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_experience`: `precision_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `required_experience`: `recall_positive`: required_experience is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `salary_range_stated`: `dummy_stratified`: sklearn DummyClassifier(strategy='stratified', random_state=0) fitted on 12725 train+validation gold rows, predicting on the 1000 scored records; its predict_proba is one-hot on the sampled label, so it is a certainty-asserting floor
* `salary_range_stated`: `majority_class`: always predicts 'not_stated' (822/1000 = 82.2% of the scored subset) and carries no distribution, so brier, ece, pr_auc and the routing sweep are null for it
* `classifier_dev` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `classifier_dev_no_rubric` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `dummy_stratified` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `gliner` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `salary_range_stated`: `invalid_json_rate`: invalid_json_rate = 0.5%: 5 of 1000 rows failed the arm's own output validation and are scored wrong (arm-reported reasons: 5 x 'output budget of 4096 tokens exhausted before an answer (finish_reason=length); raise --max-tokens'); 5 of them are also excluded from Brier/ECE because they carry no probability
* `llm_hosted_json_object` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_hosted_json_object` / `salary_range_stated`: `unparseable_value`: 5 row(s) carry a value outside the declared label set; they are scored as wrong, and a per-class table cannot place them in any class
* `llm_local` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `llm_local` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `salary_range_stated`: `brier`: no row of majority_class carries a usable p_decision for salary_range_stated (confidence and probabilities are both absent or null)
* `majority_class` / `salary_range_stated`: `ece`: no row of majority_class carries a usable p_decision for salary_range_stated (confidence and probabilities are both absent or null)
* `majority_class` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `majority_class` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `salary_range_stated`: `f1_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `salary_range_stated`: `pr_auc`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `salary_range_stated`: `precision_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified
* `tfidf` / `salary_range_stated`: `recall_positive`: salary_range_stated is a multi-label question: it has no positive class, so positive-class metrics (pr_auc, precision, recall, f1, and the 95 percent intervals) are not defined for it and are not quantified

## 9. Leakage guard (`salary_range_stated`)

`salary_range` is platform metadata, not posting text, so an arm that sees only `title + description` cannot legitimately beat the majority rate. The guard is therefore deliberately one-directional:

* **PASS** when `accuracy <= floor + 3pp`: no evidence that the gold reached the model's input.
* **FAIL** when any arm is ABOVE `floor + 3pp`: that is the only direction leakage can push accuracy, so the process prints FAIL and exits non-zero.
* **BELOW** (`accuracy < floor - 3pp`) does not fail: it is below the floor, not leakage. That arm is not answering the question that was asked (see the no-rubric ablation) or is not reading the text, and it is reported with its own note.

Floor compared against: 0.822 (majority label 'not_stated', 822/1000 of the records present in these results). It is computed from the gold of the records actually present in the results, not from the full split, and 'above' is the only direction that can indicate leakage.

| arm | n | accuracy | floor (majority rate) | delta (pp) | verdict | note |
|---|---|---|---|---|---|---|
| classifier_dev | 1000 | 0.803 | 0.822 | -1.9 | PASS | within 3pp above the majority rate ('not_stated') and at most 1.9pp below it |
| classifier_dev_no_rubric | 1000 | 0.245 | 0.822 | -57.7 | BELOW | 57.7pp BELOW the majority rate ('not_stated'): below the floor, not leakage. The arm is not answering the question that was asked (see the no-rubric ablation) or is not reading the text |
| gliner | 1000 | 0.265 | 0.822 | -55.7 | BELOW | 55.7pp BELOW the majority rate ('not_stated'): below the floor, not leakage. The arm is not answering the question that was asked (see the no-rubric ablation) or is not reading the text |
| llm_hosted_json_object | 1000 | 0.798 | 0.822 | -2.4 | PASS | within 3pp above the majority rate ('not_stated') and at most 2.4pp below it |
| llm_local | 1000 | 0.795 | 0.822 | -2.7 | PASS | within 3pp above the majority rate ('not_stated') and at most 2.7pp below it |
| tfidf | 1000 | 0.834 | 0.822 | 1.2 | PASS | within 3pp above the majority rate ('not_stated') |

Floors are excluded from the guard: `majority_class`, `dummy_stratified` are built from the gold itself, so the check would be vacuous on them (the stratified floor also samples labels without looking at the text, so its accuracy sits below the majority rate by construction).

Verdict: **PASS** (classifier_dev=PASS, classifier_dev_no_rubric=BELOW, gliner=BELOW, llm_hosted_json_object=PASS, llm_local=PASS, tfidf=PASS).
