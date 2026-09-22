# Progress

Running log. Newest first. See [DECISIONS.md](DECISIONS.md) for the reasoning behind choices and
[ROADMAP.md](ROADMAP.md) for what is planned.

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
| 1 | Install CUDA PyTorch (`--index-url .../cu128`), ~2.5 GB | ⏳ needs go-ahead |
| 2 | `ollama pull llama3.2:3b`, ~2 GB | ⏳ needs go-ahead |
| 3 | Verify SemEval-2020 Task 11 is actually obtainable (D3) | ⏳ M8 |
| 4 | Rotate the API keys committed to the old repo's `.env` | ⏳ user action |
| 5 | Confirm scope: research result (M0–M7) vs full system (M0–M11) | ⏳ |
