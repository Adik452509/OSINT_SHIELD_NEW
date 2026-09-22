# OSINT Shield

Transformer fine-tuning and real-time classification for OSINT defence & security news.

A fine-tuned multilingual encoder classifies every incoming article in ~20 ms and assigns a
confidence score. Only articles that are **high-severity or low-confidence** are escalated to a
local LLM for narrative summary and reasoning. Running a language model over every article in a
live feed is neither affordable nor necessary — the encoder is what makes the filter cheap.

This is a clean rebuild. The previous implementation lives in `../Osnit-Sheild-` and is reference
only; nothing is copied from it wholesale.

---

## Status

Phase 2 complete — project scaffold, config system, label taxonomy and keyword rubric are in place
and tested. No training code yet. See [docs/PROGRESS.md](docs/PROGRESS.md) for what is done and
[docs/ROADMAP.md](docs/ROADMAP.md) for the milestone plan.

---

## The three things that shape this project

**1. The corpus is smaller than it looks.** 1,064 labelled articles, but `Security` is 77.4% of
them and four of the five narrative classes have under 45 examples. `Radicalization` has zero.
Macro-F1 is the primary metric precisely because accuracy hides this.

**2. 46% of articles have no body text.** 490 rows are headline-only (all `news.google.com` RSS
entries). Worse, body-presence is confounded with severity — 34.0% of bodied articles are High
severity against 9.5% of headline-only ones. A classifier fed *nothing but* the `has_body` flag
scores **F1(High) = 0.478**. Default `data.mode` is therefore `body_only`; `all` is run as the
measurement of that confound, never as the headline number.

**3. Propaganda has 7 positives.** It stays in the training objective by explicit decision, with a
down-weighted loss and a capped positive-class weight — but its score is reported as *diagnostic*,
always alongside the number of positives that were actually in the test fold. The real path to a
propaganda classifier is keyword mining → re-annotation → transfer learning, not gradient descent
on five examples.

---

## Install

**Target hardware:** NVIDIA RTX 3050 Laptop 6 GB · Ryzen 5 7235HS (4c/8t) · 11.7 GB RAM · Windows 11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch with CUDA - NOT from plain PyPI, which ships a CPU-only wheel.
# Check https://pytorch.org/get-started/locally/ for the current index URL.
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"
```

Verify the GPU is actually visible before training anything:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, you installed the CPU wheel. Uninstall and reinstall from the CUDA index.

### Data

The corpus is **gitignored** and never committed. `data/raw/` should contain:

```
gold_dataset.csv          1,064 rows - the full corpus
train.csv / val.csv / test.csv   744 / 160 / 160, disjoint, union = gold
validation_results.csv    Claude vs Llama agreement, per article
osint_dataset.zip         the original archive
```

### Local LLM (pipeline stage 5)

```powershell
ollama pull llama3.2:3b
ollama list
```

> Ollama 0.34.2 is installed and running on this machine, but **`llama3.2` is not pulled yet** —
> only `qwen2.5-coder:7b`. The plan assumed the Llama install from the annotation phase was
> already here; it is not.

Do not run the LLM stage during training — 6 GB of VRAM will not hold both.

---

## Layout

```
configs/            base.yaml + one file per model candidate (single-level `extends`)
data/
  raw/              corpus CSVs            (gitignored)
  processed/        dedupe groups, folds   (gitignored)
  resources/        keywords.yaml, taxonomy.yaml   (committed - small, versioned)
docs/               PROGRESS, DECISIONS, ANALYSIS, ROADMAP + reference PDFs
scripts/            numbered, runnable, one per milestone
src/osint_shield/
  config.py         YAML loading with inheritance; label maps
  paths.py          canonical paths resolved from repo root
  keywords/         rubric matching, fusion features, mining
  data/ models/ training/ evaluation/ inference/ llm/ ingestion/ api/ storage/
tests/              pytest
runs/               experiment output     (gitignored)
```

## Usage

```powershell
pytest                      # full suite
pytest -m "not slow"        # skip anything needing a model download
ruff check . ; black .
```

Configs inherit with a single `extends` key and override any subset:

```python
from osint_shield import load_config
cfg = load_config("mmbert_small.yaml")
cfg["model"]["name"]        # 'jhu-clsp/mmBERT-small'  (override)
cfg["training"]["epochs"]   # 5                        (inherited from base.yaml)
```

## Evaluation rules

Non-negotiable, because they are what make the numbers mean anything:

- **Macro-F1**, not accuracy. Predicting `Security` for everything scores 77.4% accuracy and
  0.175 macro-F1.
- **Report mean ± std across 5 folds × 3 seeds.** The spread is itself the result.
- **Class weights are computed inside the training fold only.** Computing them on the full corpus
  leaks the test distribution.
- **Severity metrics are always broken out by `has_body`.**
- **The label ceiling is 74.3%** (narrative inter-annotator agreement; severity 79.9%). A model
  reporting 96% narrative accuracy has not solved the task — check for leakage and duplicate
  articles across folds.
- **`test.csv` stays untouched** until the very end and is reported once.

## Baselines to clear

Measured on this corpus, 5-fold CV, TF-IDF + LogReg. A fine-tuned encoder that does not beat the
TF-IDF row convincingly means there is a bug in the training loop, not a need for a bigger model.

| Metric | body_only (571) | all (1053) |
|---|---|---|
| Narrative macro-F1, 5-class | 0.473 ± 0.057 | 0.487 ± 0.067 |
| Narrative macro-F1, 3-class | 0.627 ± 0.045 | 0.638 ± 0.052 |
| Severity F1 (High) | 0.686 ± 0.048 | 0.631 ± 0.080 |
| Propaganda F1 (positive) | 0.000 | 0.000 |
