# Progress

Running log. Newest first. See [DECISIONS.md](DECISIONS.md) for the reasoning behind choices and
[ROADMAP.md](ROADMAP.md) for what is planned.

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
