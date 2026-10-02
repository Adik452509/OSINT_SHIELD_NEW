# Progress

Running log. Newest first. See [DECISIONS.md](DECISIONS.md) for the reasoning behind choices and
[ROADMAP.md](ROADMAP.md) for what is planned.

---

## 2026-10-04 — M6 · Ablations (in progress)

Reference for every ablation: mmBERT-small, mean pooling, keywords on, frozen embeddings, sdpa.
One variable changes per run; each is paired against its reference over the same 15 (fold, seed).

**Tooling.** `--set section.key=value` overrides (typo'd keys are an error; `5e-5` parses as a
float), `--vs BASE_TAG` paired comparison, `scripts/06_probe_vram.py` to measure a configuration's
real memory need before a long run.

| # | ablation | tag | result |
|---|---|---|---|
| 1 | fixed 10-epoch schedule vs early stopping | `m6-fixed` | **tie** (+0.003 / +0.004 / +0.011) → **adopted, D13**; new reference |
| 2 | `max_length` 1024 vs 512 | `m6-1024` | **no reliable gain** (−0.014 / −0.017 / +0.011), 2× cost → **keep 512, D14** |
| 3 | XLM-R bake-off | — | ⏳ |
| opt | `epochs=6` · unfrozen embeddings · `per_keyword` · `markers` | — | optional |

`m6-fixed` against TF-IDF: narrative 5-class 0.492 (+0.026, **4/5 — beats the bar**), 3-class
0.629 (+0.025, 3/5), severity 0.677 (−0.010, level). Full learning curve peaks at epoch 6 (0.519).

D12 corrected: the keyword-arm seed spread (0.004) did not replicate (0.035 here) — withdrawn.

---

## 2026-10-02/03 — M5 · Full cross-validation ✅ — parity with TF-IDF after the D11 fixes

### Done

`src/osint_shield/training/cv.py` (resumable, config-locked, atomic per-run writes),
`scripts/05_train_cv.py` (paired TF-IDF on identical folds, conservative verdicts, `--compare`,
`--tag`). Two arms × 15 runs, 26.5 min each, peak VRAM 2.80 GB, 0 leakage alarms.

### Results — mmBERT-small, `body_only`, `pooling: cls`

| task | `group_counts` | `off` | TF-IDF | verdict |
|---|---|---|---|---|
| narrative 5-class | 0.323 ± 0.102 | 0.330 ± 0.069 | 0.466 | **below**, 0/5 folds |
| narrative 3-class | 0.481 ± 0.131 | 0.534 ± 0.080 | 0.604 | **below** |
| severity F1(High) | 0.622 ± 0.073 | 0.598 ± 0.085 | 0.687 | inconclusive / below |
| propaganda F1 | 0.000 | 0.000 | 0.000 | diagnostic (5 pos/seed) |

Pooled 5-class F1 (`group_counts` / `off`): Security 0.830 / 0.854, Political 0.434 / 0.523,
Civilian 0.259 / 0.312, **Investigation 0.160 / 0.000**, Other 0.042 / 0.062.

Keyword fusion vs off, paired: severity +0.024 (10/15), narrative 3-class −0.052 (5/15); std of
the paired differences 0.13–0.17. **Undecided** — swamped by run-to-run instability.

Best epoch median 8; 20% at the 10-epoch ceiling — D8's trigger **not** hit.

### Diagnosis (D11)

1. **Pooling bug.** mmBERT's config says `classifier_pooling: mean`; we pooled the `<bos>` token,
   which under 128-token local attention mostly sees the first ~64 tokens.
2. **Gameable monitor (my D10).** Severity F1(High) rewarded untrained all-High heads; 3/30 runs
   restored epoch-2/4 checkpoints and scored at majority level.
3. **Noisy selection + unstable training.** Same config/seed/fold: 0.523 in M4, 0.286 in M5.

Fixed: mean pooling for mmBERT, `combined` monitor on severity macro-F1, per-epoch test-fold
tracking (record only) to measure selection quality. The CV is now a development set; `test.csv`
stays held out for the final number.

### v2 — after the D11 fixes (`off` arm, 2026-10-03)

| task | M5 (`cls`) | **v2 (`mean`)** | TF-IDF | verdict |
|---|---|---|---|---|
| narrative 5-class | 0.330 | **0.472 ± 0.080** | 0.466 | matches the bar, 3/5 |
| narrative 3-class | 0.534 | **0.616 ± 0.061** | 0.604 | matches the bar, 3/5 |
| severity F1(High) | 0.598 | **0.656 ± 0.047** | 0.687 | inconclusive, 2/5 |
| propaganda F1 | 0.000 | 0.067 ± 0.249 | 0.000 | diagnostic — 1 of 15 runs caught its positive |

Pooled 5-class F1: Security 0.858, Political 0.596, Civilian **0.446** (was 0.312),
Investigation **0.254** (was 0.000), Other **0.304** (was 0.062).

**Attribution.** Both fixes landed together, but the monitor fix is bounded: it removed the one
collapsed run (0.172); excluding that run, the old arm averaged 0.341. So ≈ +0.01 is the monitor
and ≈ +0.13 is mean pooling.

No collapsed runs (worst 0.330, was 0.096). Seed spread 0.041, fold spread 0.053. Best epoch
median 6, 0% at the ceiling — D8's undertraining worry is ruled out.

**Selection quality** (test fold tracked per epoch, record only):

| | selected | final epoch | best possible |
|---|---|---|---|
| narrative 5-class | 0.472 | 0.477 | 0.546 |
| severity F1(High) | 0.656 | 0.631 | 0.689 |

Mean test-fold narrative by epoch: 0.185 / 0.278 / 0.434 / 0.468 / **0.492 / 0.490 / 0.488** /
0.458 / 0.459 / 0.448 (epochs 8–10 average 11 / 8 / 4 runs — survivorship bias). Peaks at 5–7,
then overfits. Early stopping ≈ final epoch for narrative; helps severity slightly.

**Candidate for M6:** a fixed ~6-epoch schedule with no early stopping. It removes the noisy
47-row selection and returns those 47 rows (~14%) to training. Not applied yet — the keyword arm
must run under the identical config to stay a valid paired comparison.

### v2 — keyword arm (`group_counts`) and the paired comparison

| task | **keywords on** | keywords off | TF-IDF | verdict (on) |
|---|---|---|---|---|
| narrative 5-class | **0.489 ± 0.073** | 0.472 | 0.466 | +0.023, 3/5 — inconclusive |
| narrative 3-class | **0.625 ± 0.049** | 0.616 | 0.604 | +0.021, 3/5 — inconclusive |
| severity F1(High) | **0.666 ± 0.042** | 0.656 | 0.687 | −0.021, 2/5 — inconclusive |
| propaganda F1 | 0.000 | 0.067 | 0.000 | diagnostic |

Paired, same fold and seed, 15 runs: narrative 5-class **+0.017 ± 0.072 (10/15)**, 3-class
+0.009 ± 0.061 (10/15), severity +0.010 ± 0.069 (5/15). A 10/15 sign split arises by chance
~30% of the time, and the runs share folds, so the effect is **not statistically established**.

Pooled 5-class F1, on / off: Security 0.852 / 0.858, Political 0.571 / 0.596, Civilian 0.412 /
0.446, **Investigation 0.329 / 0.254**, Other 0.323 / 0.304.

Seed spread, narrative: **0.004 with keywords vs 0.041 without** (seed means 0.488/0.485/0.495 vs
0.520/0.419/0.477) — suggestive of stabilisation; three means per arm is thin evidence.

Selection: narrative selected 0.489 / final 0.491 / best possible 0.561; severity selected
**0.666 / final 0.684** / best possible 0.711.

### M5 verdict

- **The encoder reaches parity with TF-IDF, not a clear win.** On ~400 labelled training
  articles with a 74.3% annotator ceiling, that is the honest result. The plan's projections
  (3-class 0.78–0.85, severity 0.82–0.88) were optimistic, as M3 anticipated.
- **Keyword fusion stays on** — D12.
- **Early stopping adds nothing on average.** Selected minus final epoch across 30 tracked runs:
  narrative −0.005 / −0.002, severity +0.025 / −0.018. It costs 47 training rows (12%) to pick an
  epoch. A fixed schedule is M6's first candidate.
- Remaining suspects for the severity gap: **truncation** (74.8% of inputs cut at 512 tokens;
  TF-IDF reads every word; casualty evidence often sits deep in an article) and the lost 12% of
  training rows.

### Next

**M6 · Ablations and bake-off**, one variable at a time against this config (mmBERT-small,
mean pooling, keywords on, `--tag v2` as the reference). Proposed order by expected value:
fixed schedule → 1024 tokens (VRAM probe first) → XLM-R bake-off → optional arms
(unfrozen embeddings, `per_keyword`, `markers`).

---

## 2026-10-02 — M4 · Multi-task model + training loop ✅

### Done

- `src/osint_shield/models/` — `MultiTaskModel` (shared encoder, one linear head per task,
  keyword fusion at the pooled vector with train-fold standardisation stored as buffers),
  `class_weights` (training-fold labels only, propaganda capped at 10×), `MultiTaskLoss`,
  `freeze_input_embeddings`.
- `src/osint_shield/training/` — `Trainer` (bf16 AMP, gradient accumulation with a flushed final
  step, linear warmup/decay, early stopping with a `min_epochs` floor), `ArticleDataset` with
  dynamic padding, and `fold.py`: group-aware inner split, disjointness assertions, overfit check.
- `src/osint_shield/runtime.py` — HF cache, device, seeding.
- `scripts/04_train_one_fold.py` — pre-flight VRAM check, friendly OOM handling.
- Not in the original M4 file list: `runtime.py`, `training/dataset.py`, `training/fold.py`.

**Verified** `pytest` all green; overfit check passed; fold 0 passed every M4 check.

### Results

**Overfit check** (40 rows, scored on themselves): narrative / severity / propaganda F1 **1.000 /
1.000 / 1.000**, loss 3.98 → 0.00005, peak VRAM 2.62 GB, 41 s.

**Fold 0** (mmBERT-small, `body_only`, seed 42; train 333 / inner-val 47 / test 96):

| | fold 0 | TF-IDF clean bar (5-fold mean) |
|---|---|---|
| narrative 5-class | 0.523 | 0.466 |
| narrative 3-class | 0.584 | 0.604 |
| severity F1(High) | 0.596 | 0.687 |
| propaganda F1 | 0.000 (1 pos, 0 predicted) | 0.000 |

Peak VRAM 2.8 GB; 114 s per fold → M5's 15 runs ≈ 30 min for mmBERT-small. Narrative accuracy
0.781, below the 0.92 leakage alarm. Single fold, single seed: ±0.05 noise, not a verdict.

### Findings

- **OOM on the first attempt** → D9. The GPU also drives the display (1.36 GB used idle), and M1
  had under-measured (fp16 `GradScaler` skipped the first step, so AdamW state was never
  allocated). Fixed with SDPA attention and frozen word embeddings: mmBERT batch 8 4.73 → 2.63 GB,
  XLM-R batch 4 5.25 → 2.71 GB.
- **`min_epochs` vindicated.** Inner-val narrative F1 was flat for 3 epochs (0.13/0.11/0.16). The
  plan's patience-2-no-floor would have stopped there.
- **Single-task monitor sacrificed severity** → D10, monitor is now `combined`.
- **Best epoch = final epoch** with LR at zero — D8's revisit trigger, deferred to M5's 15-run
  evidence because the 47-row inner-val signal is too noisy to act on from one run.

### Next

**M5 · Full cross-validation** — 5 folds × 3 seeds, mmBERT-small first, with best-epoch
distribution recorded.

---

## 2026-09-25 — M3 · Baselines ✅

### Done

`src/osint_shield/evaluation/{metrics,baselines}.py`, `scripts/03_baselines.py`,
`runs/baselines/baseline_results.json`. Five baselines × four tasks × two data
modes × two fold protocols.

**Verified** `pytest` → **90 passed**. Every Phase 1 reference reproduced within
tolerance: `body_only` leaky narrative 5-class **0.473 exactly**, 3-class **0.627
exactly**, severity 0.692 (vs 0.686), propaganda 0.000.

### The bar for M5 — `body_only`, clean folds (n=476)

| Task | majority | keyword_rules | keyword_logreg | **tfidf_logreg** |
|---|---|---|---|---|
| narrative 5-class | 0.172 | 0.443 | 0.383 | **0.466 ± 0.015** |
| narrative 3-class | 0.287 | 0.517 | 0.474 | **0.604 ± 0.038** |
| severity F1(High) | 0.000 | 0.667 | 0.675 | **0.687 ± 0.072** |
| propaganda F1(pos) | 0.000 | **0.118** | 0.063 | 0.000 |

### Three findings

**1. Rules beat learning on the rubric.** `keyword_rules` (argmax over group
counts, zero training) beats `keyword_logreg` by ~6 points on narrative, and lands
within 0.023 of a 50,000-feature TF-IDF model using 61 terms. → D4 amended: fuse
**raw counts** into the encoder, not a pre-learned projection.

**2. The rubric is the only propaganda signal that exists.** Every learned method
scores exactly 0.000; `keyword_rules` scores 0.100–0.150 with no training data.
→ D3 amended: propaganda ships as rules + LLM zero-shot; the trained head stays as
a diagnostic.

**3. The `has_body` confound, quantified in model terms.** In `all` mode,
`tfidf_logreg` severity F1(High) is **0.657 on bodied articles and 0.327 on
headline-only ones** — the same model, half the performance, because the
headline-only regime is 9.5% High against 34.0%. The `has_body`-only control
(zero text) scores **0.476**, confirming the Phase 1 measurement of 0.478.

### Leakage cost — `tfidf_logreg`, leaky minus clean

| mode | narrative 5 | narrative 3 | severity |
|---|---|---|---|
| `body_only` | −0.007 | −0.023 | −0.005 |
| `all` | **−0.051** | −0.037 | −0.015 |

Duplicate leakage was inflating `all`-mode narrative macro-F1 by **5 points**. The
headline-only half of the corpus is where the syndicated copies live, so it carries
most of the leak. This is a reportable result, not just hygiene.

### Error structure — pooled 3-class confusion, `body_only` clean

| truth ↓ / pred → | Other | Political | Security |
|---|---|---|---|
| Other | 14 | 5 | 26 |
| Political | 4 | 42 | 25 |
| Security | 13 | 28 | 319 |

Per-class F1: Security **0.874**, Political **0.575**, Other **0.368**.
The Security↔Political boundary is the main error mode in both directions (28 and
25) — exactly as the plan predicted. `Other` is largely swallowed by Security (26
of 45), consistent with it being a residual bucket rather than a class.

No leakage alarm fired under either protocol.

### Consequence for M5 targets

The plan's projections were written against an assumed TF-IDF bar. Measured against
the real clean bar, the gaps the encoder must close are:

| Task | clean bar | plan target | gap |
|---|---|---|---|
| narrative 5-class | 0.466 | 0.50 – 0.62 | +0.03 to +0.15 |
| narrative 3-class | 0.604 | 0.78 – 0.85 | **+0.18 to +0.25** |
| severity F1(High) | 0.687 | 0.82 – 0.88 | **+0.13 to +0.19** |

The 3-class and severity targets look optimistic. Treat them as aspirations, not
expectations, and report what is measured.

### Next

**M4 · Multi-task model + training loop** — shared encoder, three heads, one fold
end to end on the GPU, with the 40-row overfit check as the gradient-path proof.

---

## 2026-09-25 — M2 · Data layer ✅

### Done

- `src/osint_shield/data/loaders.py` — corpus → normalised frame. Labels mapped
  from the `narrative` **string** via `taxonomy.yaml`, never the corpus's own
  non-contiguous `narrative_label` column. `has_body` materialised. Original split
  membership recovered so `test.csv` can be held out exactly.
- `src/osint_shield/data/dedupe.py` — normalisation, exact-duplicate flagging, and
  char 3–5gram TF-IDF cosine clustering via connected components.
- `src/osint_shield/data/splits.py` — hold-out resolution with leak removal, and
  `StratifiedGroupKFold` fold assignment.
- `scripts/02_build_folds.py` — the runnable pipeline with regression checks.
- `docs/DATA_REPORT.md`, `data/processed/folds_{body_only,all}.csv`,
  `holdout_{body_only,all}.csv`.

**Verified** `pytest` → **66 passed**. `M2 PASSED`.

### Results

| | `body_only` | `all` |
|---|---|---|
| CV pool | **476** | **891** |
| Hold-out (`test.csv`) | 94 | 159 |
| Fold size spread | 1 row | 2 rows |
| Propaganda positives in pool | 5 of 7 | 5 of 7 |
| Duplicate groups spanning folds | **0** | **0** |

- Exact duplicates: **11**. Near-dup clusters at 0.85: **22 multi-row groups, 26
  redundant rows, largest 6**.
- **12 clusters spanned two original splits** — the leakage this milestone exists
  to close.
- Hold-out is **159**, not 160: one `test.csv` row was an exact duplicate.
- **3 rows dropped** from the CV pool because a near-duplicate sits in the
  hold-out (1 of them bodied). Reconciliation is exact: 476 + 94 + 1 = 571.
- Rare classes per test fold in `body_only`: **Investigation 2–3, Other 2–3,
  Civilian 4–5**. One error moves such a class's recall by 33–50 points — this is
  why the collapsed 3-class figure is the honest headline.

### Notes

- **Largest cluster is the "Pralay missile" story, 6 articles.** Several of its
  pairs score *below* 0.85 (0.752–0.913) and only group by chaining through pairs
  that clear the threshold. Without grouping, six near-identical articles would
  have scattered across folds — the single largest leak in the corpus.
- **Known benign false positive:** "29th Meeting of the Working Mechanism…" groups
  with "30th Meeting of the…" at 0.98. Different events, near-identical
  boilerplate headlines. Grouping is still correct — a model trained on one and
  tested on the other succeeds by pattern-matching, not understanding.
- **TF-IDF is corpus-relative.** `similarity_groups` must receive the whole corpus
  in one call; similarity values are not comparable across calls of different
  sizes. M10's streaming ingestion will need incremental matching against stored
  vectors, not batch re-clustering.
- Bug found and fixed: `normalise_headline` stripped punctuation before collapsing
  whitespace, so tabs and newlines were deleted rather than converted to spaces,
  welding adjacent words. No effect on this corpus (0 affected headlines) but it
  would bite on live RSS.

### Consequence for M3

The measured baselines (narrative macro-F1 0.473, severity F1(High) 0.686) came
from a **flat** `StratifiedKFold` over 571 rows with duplicates scattered across
folds and `test.csv` included. The M2 pool is 476 rows, group-aware, hold-out
excluded — a different and stricter protocol, so those numbers will **not**
reproduce exactly, and the clean figure should be lower.

M3 therefore runs **both** protocols. The gap between them quantifies how much of
the original score was duplicate leakage, which is itself a reportable result.

### Next

**M3 · Baselines** — majority, TF-IDF + LogReg, keyword-rules, plus the
`has_body`-only control, across both protocols and both data modes.

---

## 2026-09-22 — M0 · Project scaffold ✅

### Done

- New project at `C:\Users\LENOVO\osint-shield`, separate from the old `Osnit-Sheild-` repo.
  Fresh git history; nothing copied from the old codebase.
- Directory structure: `src/` package, `data/{raw,processed,external,resources}`, `configs/`,
  `docs/`, `scripts/`, `tests/`, `runs/`.
- **Config system** — `configs/base.yaml` plus one file per model candidate, single-level
  `extends` with recursive merge. `src/osint_shield/config.py`, `paths.py`.
- **Label taxonomy** — `data/resources/taxonomy.yaml`. 5 contiguous narrative classes;
  `Radicalization` declared but not trained (D6).
- **Keyword rubric** — `data/resources/keywords.yaml`, transcribed verbatim from the PDF, plus
  `src/osint_shield/keywords/matcher.py`: word-boundary matching, group-count and per-keyword
  fusion features, marker injection, low-precision exclusion.
- Dataset placed in `data/raw/` (gitignored). Keywords PDF committed to `docs/reference/`.
- `.gitignore`, `.env.example`, `README.md`, `pyproject.toml`.
- Virtualenv at `.venv` with core deps. **torch deliberately not installed yet** — needs the CUDA
  wheel, not the PyPI default.

**Verified** `pytest` → **32 passed**.

### Environment findings

| | |
|---|---|
| GPU | **NVIDIA RTX 3050 Laptop, 6 GB**, driver 591.66 |
| CPU / RAM | Ryzen 5 7235HS, 4c/8t · 11.7 GB |
| PyTorch (old project) | `2.10.0+cpu` — **CUDA unavailable**, wrong wheel |
| Ollama | **installed & running**, v0.34.2 |
| Llama 3.2 | ❌ **not pulled** — only `qwen2.5-coder:7b` present |

The plan PDF assumed the Llama 3.2 / Ollama install from the annotation phase was already on the
training machine. Ollama is here; the model is not.

### Decisions made

D1 CV over `train ∪ val`, `test.csv` held out · D2 `body_only` primary · D3 propaganda retained as
a trained but *diagnostic* task · D4 keywords fused as a measured ablation · D5 group-aware folds ·
D6 five contiguous narrative classes · D7 train on this laptop, CUDA wheel required.

### Analysis carried forward

All figures measured directly from `gold_dataset.csv`, not taken from the plan PDF.

- 1,064 rows; splits are disjoint and their union is the gold set (744/160/160).
- Narrative: Security 824 · Political 155 · Civilian 41 · Other 26 · Investigation 18 ·
  **Radicalization 0**. Severity: Low 823 / High 241. Propaganda: **7 positives**.
- **490 rows (46.1%) have no body text**; High-severity rate 9.5% headline-only vs 34.0% bodied;
  a `has_body`-only classifier scores F1(High) = **0.478**.
- Inter-annotator agreement **74.34% narrative / 79.89% severity** — reproduces the plan exactly.
- Token lengths (real XLM-R tokenizer): median 370 all rows / **859 bodied**; at 512 only
  **17.3% of bodied articles** are seen in full.
- Near-duplicates: 31 redundant rows at 0.85 cosine; 20 still leak after exact-headline dedupe.
- Baselines reproduced exactly against the reference scripts — see README.
- Keyword rubric: severity keywords alone score F1(High) 0.685 vs 0.686 for full TF-IDF;
  propaganda keywords recover 5/7 known positives from a 57-article candidate pool.

### Next

**M1 · Environment + GPU proof.** Blocked on two installs — see below.

---

## Open items

| # | Item | Status |
|---|---|---|
| 1 | Install CUDA PyTorch | ✅ `torch 2.11.0+cu128`, RTX 3050 visible |
| 2 | `ollama pull llama3.2:3b` | ✅ pulled and answering |
| 3 | Verify SemEval-2020 Task 11 is actually obtainable (D3) | ⏳ M8 |
| 4 | Rotate the API keys committed to the old repo's `.env` | ⏳ user action |
| 5 | Confirm scope: research result (M0–M7) vs full system (M0–M11) | ⏳ |
| 6 | Correct the plan's "`[CLS]` → 768-dim" claim: mmBERT-small is **384** | ⏳ M11 write-up |
| 7 | ~~1024-token ablation confirmed feasible (3.66 GB at 8×512)~~ — **wrong, see D9.** M1 under-measured; re-probe 1024 with frozen embeddings + sdpa before relying on it | ⏳ M6 |
