# Roadmap

Ordered, testable milestones. **One at a time — each stops for confirmation before the next
begins.** Every milestone lists what it produces, what it depends on, and how we verify it.

M0–M6 are a complete and defensible research result on their own. M7–M10 build the operational
system on top.

Effort figures assume the RTX 3050 laptop (see [DECISIONS.md](DECISIONS.md) D7).

---

## ✅ M0 · Project scaffold — done 2026-09-22

Structure, config system with inheritance, label taxonomy, keyword rubric, git, tests.

**Deliverables** `pyproject.toml`, `configs/*.yaml`, `data/resources/{keywords,taxonomy}.yaml`,
`src/osint_shield/{config,paths}.py`, `src/osint_shield/keywords/matcher.py`, `.env.example`
**Verify** `pytest` → 32 passed ✅

---

## M1 · Environment + GPU proof

**Goal.** PyTorch sees the RTX 3050, and both candidate models load and do a forward pass.

**Files** `scripts/01_check_env.py`, `docs/ENVIRONMENT.md`
**Depends on** M0
**Verify**
- `torch.cuda.is_available()` is `True` and names the RTX 3050
- `mmBERT-small` and `xlm-roberta-base` each load and run one batch at `max_length=512`
- peak VRAM is recorded for both, at batch 8 and batch 4
- `ollama list` shows `llama3.2`

**Risk.** mmBERT may need a `transformers` version bump or `trust_remote_code`. Find out now, not
in the middle of a 15-run bake-off.

---

## M2 · Data layer: load, dedupe, group, fold

**Goal.** One reproducible path from `gold_dataset.csv` to fold assignments.

**Files** `src/osint_shield/data/{loaders,dedupe,splits,stats}.py`, `scripts/02_build_folds.py`
**Depends on** M0
**Deliverables** `data/processed/folds.parquet`, `docs/DATA_REPORT.md`
**Verify**
- reproduces the known counts: 1,064 rows → 11 exact-dup headlines → 1,053; `body_only` = 571
- fuzzy grouping finds ~31 redundant rows at 0.85; **zero** near-dup pairs span two folds
- every fold's narrative distribution is within ±2 rows of stratified expectation
- the 7 propaganda positives are spread across folds, none has zero
- `test.csv` ids appear in **no** CV fold (D1)
- golden-file test on a 50-row fixture

---

## M3 · Baselines

**Goal.** The bar every later number is compared against.

**Files** `src/osint_shield/evaluation/{baselines,metrics,report}.py`, `scripts/03_baselines.py`
**Depends on** M2
**Deliverables** `runs/baselines/baseline_results.json`
**Verify** reproduces the numbers already measured, in both data modes:

| Metric | body_only (571) | all (1053) |
|---|---|---|
| Narrative macro-F1, 5-class | 0.473 ± 0.057 | 0.487 ± 0.067 |
| Narrative macro-F1, 3-class | 0.627 ± 0.045 | 0.638 ± 0.052 |
| Severity F1 (High) | 0.686 ± 0.048 | 0.631 ± 0.080 |
| Propaganda F1 (positive) | 0.000 | 0.000 |

Plus the `has_body`-only control at F1(High) ≈ 0.478, and the keyword-rules baseline at ≈ 0.685.
Deviation beyond ±0.02 means the fold logic changed — investigate before continuing.

---

## M4 · Multi-task model + training loop

**Goal.** Shared encoder, three heads, one fold trains end to end on the GPU.

**Files** `src/osint_shield/models/{encoder,heads,losses}.py`,
`src/osint_shield/training/{trainer,callbacks}.py`, `scripts/04_train_one_fold.py`
**Depends on** M1, M2
**Verify**
- overfits a 40-row subset to >0.95 train macro-F1 — proves the gradient path works
- class weights are computed on the training fold only (asserted in test)
- the propaganda positive weight is capped at 10×, not ~150×
- early stopping fires on inner-val macro-F1; the outer test fold is never touched
- peak VRAM under 6 GB with AMP on
- one fold completes in a sane wall-clock time

---

## M5 · Full CV + keyword fusion ablation

**Goal.** 5 folds × 3 seeds, and the answer to whether keyword fusion helps.

**Files** `src/osint_shield/training/train_cv.py`, `scripts/05_train_cv.py`
**Depends on** M4
**Deliverables** `runs/cv_<model>_<mode>/{cv.csv,oof.csv,summary.json}`
**Verify**
- beats the M3 TF-IDF bar convincingly — if not, it is a bug, not a model-size problem
- keyword arms `off` / `group_counts` / `per_keyword` / `markers` reported side by side (D4)
- severity metrics broken out by `has_body`
- **leakage alarm**: narrative accuracy above 0.92 halts and triggers a duplicate check —
  the annotator ceiling is 74.3% (plan §5.2)
- seed spread reported, not hidden

---

## M6 · Model bake-off + remaining ablations

**Goal.** Settle model choice and the open config questions on evidence.

**Files** `scripts/06_bakeoff.py`, `scripts/07_ablations.py`
**Depends on** M5
**Deliverables** `runs/model_comparison.csv`, `runs/ablation_table.csv`
**Ablations** mmBERT-small vs xlm-roberta-base · seq len 256/512/1024 · headline-only vs
headline+body · class weighting on/off · 5-class vs collapsed 3-class · keyword arms
**Verify** each ablation changes exactly one variable against a fixed baseline; winner chosen on
mean macro-F1 *and* its spread, not mean alone.

---

## M7 · Error analysis

**Goal.** Separate model error from label ambiguity.

**Files** `scripts/08_error_analysis.py`
**Depends on** M6
**Deliverables** `docs/ERROR_ANALYSIS.md`, confusion matrices
**Verify** manual read of 30 Security/Political errors, cross-referenced against
`narrative_confidence` and `agree_narrative`; the 33 `review_needed` articles inspected.

---

## M8 · Propaganda: mining + transfer

**Goal.** Move propaganda from 7 positives toward a real training set (D3).

**Files** `src/osint_shield/keywords/mining.py`, `scripts/09_mine_propaganda.py`
**Depends on** M2
**Deliverables** `data/processed/propaganda_candidates.csv` (~57 articles), re-annotation guide
**Verify** recovers ≥5 of the 7 known positives; candidate pool reviewed with the rubric criteria;
`exclude_low_precision` measurably improves precision over the 8.8% baseline.

---

## M9 · Inference service

**Goal.** `predict.py` and a FastAPI wrapper.

**Files** `src/osint_shield/inference/{predict,calibration,versioning}.py`,
`src/osint_shield/api/`, `scripts/10_serve.py`
**Depends on** M6
**Verify** single and batch paths agree; confidence threshold calibrated on validation folds
(~0.70 start); `model_version` on every record; latency measured against the ~20 ms target.

---

## M10 · LLM stage + live pipeline

**Goal.** The two-stage system end to end.

**Files** `src/osint_shield/llm/`, `src/osint_shield/ingestion/`, `src/osint_shield/storage/`
**Depends on** M9, `ollama pull llama3.2:3b`
**Verify** escalation routes ~25% of traffic; LLM output validates against the fixed JSON schema;
evidence spans are verbatim from the source; review queue captures low-confidence predictions;
ingestion dedupes by URL hash *and* normalised headline.

---

## M11 · Write-up

**Deliverables** `docs/REPORT.md` → `report.pdf`
Methods, results with CV means and standard deviations, and a limitations section that states the
rare-class, propaganda and `has_body` problems plainly.
