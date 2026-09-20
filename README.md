# job-posting-triage

Four engines label the same job ads on four questions: scam, experience, education,
and whether the ad names a salary. Labels, rubrics and gold answers all come from
`triage/schema.py`, so every arm answers the same question in the same words.

Data: Kaggle's Real / Fake Job Posting Prediction (CC0), 3,182-row test split. This
run takes a random 1,000 with seed 0. Every table:
[`results/report.md`](results/report.md). One-page summary:
[`docs/index.html`](docs/index.html).

## The scam question

53 of the 1,000 ads are scams. Calling every ad legitimate scores 0.947 accuracy, so
that is the floor.

| arm | fitted on this data? | accuracy | F1 (fraud) | precision | recall | PR-AUC | ECE | predicts fraud |
|---|---|---|---|---|---|---|---|---|
| `tfidf` | yes, 12,725 labelled postings | 0.970 | 0.700 [0.59, 0.79] | 0.745 [0.60, 0.85] | 0.660 [0.53, 0.77] | 0.780 | 0.033 | 47/1000 |
| `llm_hosted_json_object` | no | 0.948 | 0.329 [0.19, 0.47] | 0.600 [0.39, 0.78] | 0.226 [0.13, 0.36] | 0.282 | 0.015 | 20/1000 |
| `classifier_dev` | no | 0.947 | 0.312 [0.18, 0.45] | 0.500 [0.31, 0.69] | 0.226 [0.13, 0.36] | 0.276 | 0.046 | 24/1000 |
| `classifier_dev_no_rubric` | no | 0.930 | 0.271 [0.16, 0.39] | 0.302 [0.19, 0.45] | 0.245 [0.15, 0.38] | 0.260 | 0.070 | 43/1000 |
| `gliner` | no | 0.897 | 0.104 [0.04, 0.18] | 0.097 [0.05, 0.20] | 0.113 [0.05, 0.23] | 0.070 | 0.082 | 62/1000 |
| `llm_local` | no | 0.053 | 0.101 [0.08, 0.13] | 0.053 [0.04, 0.07] | 1.000 [0.93, 1.00] | 0.053 | 0.947 | 1000/1000 |
| `majority_class` (floor) | n/a, it is the gold | 0.947 | 0.000 [0.00, 0.00] | n/a | 0.000 [0.00, 0.07] | n/a | n/a | 0/1000 |
| `dummy_stratified` (floor) | sampled from the train gold | 0.906 | 0.078 [0.02, 0.16] | 0.082 [0.03, 0.19] | 0.075 [0.03, 0.18] | 0.055 | 0.094 | 49/1000 |

Intervals are 95 percent: Wilson for precision and recall, bootstrap for F1. ECE
measures the probability an arm puts on its own answer, and the floor rows put none, so
theirs reads `n/a`. The last column counts ads called fraudulent. The report's routing
table counts only the ones caught above a confidence threshold.

- `tfidf` learned from 12,725 labelled ads in this dataset. The other three learned
  nothing. PR-AUC: 0.780 tfidf, 0.282 hosted, 0.276 Jev, 0.260 Jev without the rubric,
  0.070 gliner, 0.053 local. The base rate is 0.053.
- Jev ties the 0.947 floor and the hosted model sits one ad above it. gliner reaches
  0.897, `llm_local` 0.053. With 53 scams the zero-shot intervals overlap, so they are
  inseparable here. tfidf is not: F1 [0.59, 0.79] against a 0.47 ceiling elsewhere.
- Jev knows when it is right (ECE 0.046) and cannot rank the scams. At tau 0.9 it scores
  0.875 precision on 8 ads, catching 7 of the 53; at 0.95 it scores 1.000 on 6.
- `llm_local` called all 1,000 ads fraudulent at confidence 1.0, with valid JSON every
  time. The same code against a hosted model reaches F1 0.329.
- Jev runs on a shared service, so its row drifts: a second fetch of the same 1,000 ads
  gave accuracy 0.945 and F1 0.286.
- gliner reads the label names and no rubric, and its window cuts 82 of the 1,000 ads.
- Hand-reading the misses: false negatives are ordinary engineering and operations ads
  tagged fraudulent, false positives are income-opportunity ads tagged legitimate.

## The engines

| arm | what it is | what it costs |
|---|---|---|
| `classifier_dev` | Jev, a decision model behind a classifier API. One request carries up to 1,000 ads and returns a calibrated probability per label. No tokens generated. | Free and keyless, with a daily quota. |
| `gliner` | GLiNER2.5-base, 194M parameters, DeBERTa-v2-base. One schema per question, four forward passes per ad. | Local. No generation, no parse step, four passes per ad. |
| `llm_local` | Qwen2.5-1.5B-Instruct at Q4_K_M, served by llama-server on this laptop. One chat completion covers all four questions. | Local, about 1 GB of weights, no key. |
| `llm_hosted` | deepseek-v4.1-flash through the OpenCode Go gateway, on the same code path. | Tokens and an API key. |
| `tfidf` | TF-IDF over unigrams and bigrams, then one logistic regression per question. 215,899 features for fraud. | Local, seconds, no weights. |

`classifier_dev_no_rubric` is Jev with the rubric text dropped, the ablation for what
the instruction buys. The LLM arms take `--response-format
json_schema|json_object|prompt`; a non-default choice gets its own arm label. The hosted
gateway rejects `json_schema` with HTTP 400, so that arm puts the shape in the prompt.
Both validate replies and record failures as `invalid = true`.

GLiNER's scores depend on the rest of its schema. Asked for fraud alone, the scam
fixture scores 0.966 against 0.026 for an ordinary engineering posting. Put three tasks
in the schema and the order flips: 0.190 against 0.780. Hence one question per pass,
four passes per ad. `scripts/gliner_pooling_probe.py` reproduces this.

## Run it

```bash
uv sync
uv run hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
llama-server -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf --jinja -c 16384 -np 4 --cache-ram 128 --port 8080
uv run python -m triage.data --check
uv run python -m triage.run --arms all --split test --limit 1000 --seed 0 --concurrency 4
uv run python -m triage.evaluate --report
uv run python web/build_site.py
```

- Keep `--cache-ram 128`. The default 8 GiB prompt cache filled memory here and pushed
  the machine into swap; decode fell from about 22 tokens a second to about 2.4.
- `--limit 1000 --seed 0` samples at random, since this corpus clusters and mirrors can
  sit in upload order: the first 1,000 test rows hold 37 scams against 53 for a random
  1,000. Drop `--limit` to run all 3,182.
- Jev needs no key (20,000 classifications a day per IP) and caches answers in
  `results/cache.jsonl`, which is gitignored. A first run on a fresh clone spends about
  3,100 of that quota.
- No test suite. `uv run python scripts/check_readme_against_metrics.py` compares every
  cell of the table above against `results/metrics.json` and exits non-zero on drift.

## What it does not prove

- The ads are public, dated 2014 to 2018, and widely mirrored. Both language models may
  have seen them while training, so their rows are an upper bound. The unlabelled LinkedIn
  demo is the only clean surface, and it has no gold labels.
- This run samples 1,000 ads with seed 0 and holds 53 of the split's 138 scams, with lower
  counts where the platform recorded nothing (627 for experience, 520 for education).
  `results/fullsplit/report.md` scores the cheap arms over all 3,182.
- Long ads are cut at 6,000 characters (11 of 3,182). gliner's window cuts 82 of the 1,000.
- Latency is wall clock on a busy laptop. The local LLM's median completion is 12.3 s and
  its run took 3,249 s for 1,000 ads at concurrency 4. The classifier arm's per-ad figure
  is amortized over batches of up to 1,000 answers. Cache-served rows report no latency.
- The router is a demonstration; precision on 53 positives is coarse.
- No ad text is committed. Rows carry ids, labels, scores and timings. Two files quote
  excerpts: `results/demo_linkedin.md` (200 ads from the LinkedIn mirror, 200 characters
  each) and the summary page (400 characters of one Kaggle ad). The Kaggle original is
  CC0; the LinkedIn mirror is a scrape with no stated licence.

## Layout

```
triage/schema.py       the four questions: labels, rubrics, gold answers
triage/data.py         dataset load, per-question views, --check
triage/arms/           one file per engine, behind one interface
triage/run.py          the runner, with the classifier.dev cache
triage/evaluate.py     metrics, calibration, routing, the leakage guard, report.md
triage/demo.py         triage arbitrary postings, including the LinkedIn sample
web/build_site.py      renders docs/ from results/metrics.json
scripts/llm_mode_probe.py      three postings, three output constraints, one model
scripts/gliner_pooling_probe.py  one schema or four, and why it matters here
scripts/check_readme_against_metrics.py  fails when the README table drifts
results/               one JSONL per arm, report.md, metrics.json, the plots
results/fullsplit/     the same evaluator over the whole test split, cheap arms only
docs/                  the generated summary page, at geckguy.github.io/job-posting-triage
```
