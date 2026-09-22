# Dataset analysis

Everything here is computed directly from `data/raw/gold_dataset.csv`, not taken from the plan PDF.
Reproduced by `scripts/02_build_folds.py` once M2 lands.

---

## 1 · Shape

| File | Rows | Notes |
|---|---|---|
| `gold_dataset.csv` | **1,064** | 37 columns, no duplicate ids |
| `train.csv` / `val.csv` / `test.csv` | 744 / 160 / 160 | pairwise disjoint; union = gold exactly |
| `validation_results.csv` | 1,064 | Claude vs Llama, per article; `llama_error` = 0 throughout |

Columns fall into four groups: content (`clean_headline`, `clean_text`, `original_language`,
`source_domain`, `url`, `published_at`), event/geo (`event_date`, `event_time`, `event_location`,
`country`, `state_region`, `city`), labels with rationale (per task: a label, a `_description`, a
**verbatim `_evidence` span**, a `_confidence`), and meta (`llama_*`, `agree_*`, `label_status`,
integer label columns, four `prop_*` sub-flags).

---

## 2 · Label distribution

| Narrative | n | % | | Severity | n | % |
|---|---|---|---|---|---|---|
| Security | 824 | 77.4% | | Low | 823 | 77.3% |
| Political | 155 | 14.6% | | High | 241 | 22.7% |
| Civilian | 41 | 3.9% | | | | |
| Other | 26 | 2.4% | | Propaganda = True | **7** | 0.7% |
| Investigation | 18 | 1.7% | | | | |
| **Radicalization** | **0** | **0%** | | | | |

Propaganda sub-flags: glorification 2 · justification 4 · victimhood 1 · **recruitment 0**.

`label_status`: high_confidence 766 · medium_confidence 265 · **review_needed 33**.

Language: English 1,032 · Hindi 29 · Kannada 3. All 32 Indic articles have bodies.

Dates span 2003 → 2026, but **891 of 1,064 (83.7%) are 2026** and only 173 predate it — so the
plan's suggested "train pre-2026 / test 2026" chronological hold-out is not feasible as written.

---

## 3 · The missing-body problem

`clean_text` is empty for **490 of 1,064 rows (46.1%)** — 489 `news.google.com` entries plus one.
Those URLs are Google's encoded redirect form, so no body was ever fetched.

### It is confounded with severity

| | n | High-severity rate |
|---|---|---|
| headline-only | 482 | **9.5%** |
| has body | 571 | **34.0%** |

A logistic regression fed **only the `has_body` flag** — zero text — scores **F1(High) = 0.478**.

Corroborating: annotator agreement is *higher* on headline-only rows (narrative 0.776 vs 0.716,
severity 0.876 vs 0.733) while annotator *confidence* is much *lower* (0.657 vs 0.828). Both
annotators default to Security/Low on thin input and agree trivially.

→ Decision [D2](DECISIONS.md#d2--body_only-is-the-primary-data-mode--2026-09-22).

### It changes what truncation means

Real XLM-R tokenizer, 1,053 deduped rows:

| max_length | all rows seen in full | **bodied rows seen in full** |
|---|---|---|
| 256 | 47.2% | **2.6%** |
| 512 | 55.2% | **17.3%** |
| 1024 | 79.1% | 61.5% |
| 2048 | 94.6% | 90.0% |

Median tokens: 370 across all rows, **859 across bodied rows** (p90 2,037; max 7,432).
Headline-only rows: median 22 tokens.

The plan's own figures reproduce (47.5% / 54.4% at 256 / 512) — but they average the headline-only
half in. At 512 the median *real* article is cut roughly in half. This is the argument for
mmBERT's 8,192 context, and for keyword features computed over the full document
([D4](DECISIONS.md#d4--keywords-are-a-model-input-measured-as-an-ablation--2026-09-22)).

---

## 4 · Duplicates

Char 3–5gram TF-IDF cosine on source-suffix-stripped headlines:

| threshold | pairs | cross-split pairs | conflicting labels |
|---|---|---|---|
| 0.95 | 12 | 8 | 0 |
| 0.90 | 25 | 14 | 1 |
| 0.85 | 39 | **22** | 1 |
| 0.80 | 59 | 31 | 2 |

31 redundant rows (2.9%) at 0.85. Exact normalised-headline dedupe catches only 11 and leaves
**28 pairs / 20 rows still leaking**. → [D5](DECISIONS.md#d5--group-aware-folds-not-row-dropping--2026-09-22).

---

## 5 · The label ceiling

`agree_narrative` **0.7434** · `agree_severity` **0.7989** — reproduces the plan exactly.
24.2% of articles carry `narrative_confidence` below 0.70.

Claude vs Llama narrative confusion shows where the genuine ambiguity is: 70 articles Claude called
Political were Security to Llama, and 63 the reverse — the Security/Political boundary is the main
error mode for annotators *and* will be for the model.

**A model reporting 96% narrative accuracy has not solved the task.** Expect high eighties at best;
treat anything dramatically above as a leakage alarm.

---

## 6 · Baselines (5-fold CV, TF-IDF + LogReg)

| Metric | body_only (571) | all (1053) |
|---|---|---|
| Majority-class narrative macro-F1 | 0.172 | 0.175 |
| Narrative macro-F1, 5-class | 0.473 ± 0.057 | 0.487 ± 0.067 |
| Narrative macro-F1, 3-class | 0.627 ± 0.045 | 0.638 ± 0.052 |
| Severity F1 (High) | 0.686 ± 0.048 | 0.631 ± 0.080 |
| Propaganda F1 (positive) | **0.000** | **0.000** |
| `has_body` flag alone → F1(High) | — | **0.478** |

Severity is *better* on half the data: removing the headline-only shortcut strengthens the real
signal.

---

## 7 · The keyword rubric against the data

### Narrative — too generic for the rare classes

| Class | keyword recall | keyword precision |
|---|---|---|
| Security | 0.506 | 0.860 |
| Political | 0.432 | 0.275 |
| Civilian | 0.487 | 0.170 |
| Investigation | 0.667 | **0.103** |
| Other | 0.192 | 0.096 |

### Severity — genuinely keyword-driven

| group | fires on | High rate among hits (base 22.8%) |
|---|---|---|
| `sev_High` | 232 | **62.9%** (2.8× lift) |
| `sev_Low` | 249 | 15.3% |

31 severity terms alone score F1(High) **0.685**, versus 0.686 for a 50,000-feature TF-IDF model.

### Propaganda — a mining tool, not a classifier

57 articles fire at least one propaganda keyword, recovering **5 of the 7 known positives (71%
recall)** at 8.8% precision → **52 candidates for re-annotation**.

| group | fires on | known positives caught |
|---|---|---|
| justification | 5 | 3 |
| glorification | 9 | 2 |
| victimhood | 48 | 1 |
| recruitment | 1 | 0 |

`victimhood` is carrying almost nothing — "targeted" alone drives most of its 48 hits
("targeted strike", "targeted killing"). Listed in `keywords.yaml` under `low_precision`.

---

## 8 · Unused signal worth revisiting

The plan's output contract ignores several columns the corpus actually populates:
`country` (92.0% filled, 29 values, India 728), `state_region` (26.2%), `city` (19.5%),
`event_location` (50.8%), the verbatim `narrative_evidence` / `severity_evidence` spans, the
per-task confidences, `label_status`, and the Llama predictions. Geo is what makes a dashboard
worth looking at; the evidence spans are a ready-made target for the LLM stage's `evidence` field.
