# Progress

Running log. Newest first. See [DECISIONS.md](DECISIONS.md) for the reasoning behind choices and
[ROADMAP.md](ROADMAP.md) for what is planned.

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
| 7 | 1024-token ablation now confirmed feasible on 6 GB (3.66 GB at 8×512) | ⏳ M6 |
