# Decision log

Decisions that are not derivable from the code, with the evidence behind them.
Append; don't rewrite history. Dated absolutely.

---

## D1 · Cross-validate over `train ∪ val` (904), not all 1,064 — 2026-09-22

**Decision.** 5-fold stratified *group* CV over the 904 rows of `train.csv ∪ val.csv`.
`test.csv` (160 rows) is held out untouched and reported once at the end.

**Why.** The plan PDF §4.2 asks for both "cross-validation over all 1,064 articles" *and* "keep
the existing held-out test.csv untouched as a final confirmation set". These contradict —
`test.csv` is a subset of the 1,064, so doing both leaks the confirmation set into every fold.

**Cost.** Investigation drops from 18 CV articles to ~15. Accepted: a confirmation set that was in
training confirms nothing.

---

## D2 · `body_only` is the primary data mode — 2026-09-22

**Decision.** `data.mode: body_only` (571 rows after dedupe). `mode: all` (1,053) is run as a
measurement of the confound, and reported as such, never as the headline number.

**Why.** 490 of 1,064 rows have empty `clean_text` — all `news.google.com` RSS entries plus one.
Body-presence is confounded with severity:

| | n | High-severity rate |
|---|---|---|
| headline-only | 482 | **9.5%** |
| has body | 571 | **34.0%** |

A logistic regression fed *only* the `has_body` flag — zero text — scores **F1(High) = 0.478**.
Training on the mixed corpus teaches "short input → Low", which is a collection artifact, and the
wrong failure direction for an alerting system.

Measured confirmation: severity F1(High) is **better** on half the data (0.686 body_only vs 0.631
all). Removing the shortcut strengthens the real signal.

In `mode: all`, headline-only rows get a `[NO_BODY]` marker prepended so the model can at least
distinguish the two regimes rather than inferring it from length.

---

## D3 · Propaganda stays a trained task — 2026-09-22

**Decision.** The propaganda head stays in the training objective. `loss_weight: 0.3`,
`class_weight_cap: 10.0`, `report_as: diagnostic`.

**Why.** User directive, 2026-09-22, reaffirmed after the concern below was raised.

**The concern, recorded so the write-up stays honest.** 7 positives corpus-wide (0.66%); all 7
have bodies, so `body_only` loses none, but 5 folds means **~1.4 positives per test fold**. A
TF-IDF + LogReg baseline scores **0.000** positive-class F1 in both data modes — that is an
empirical result, not a prediction. An always-"No" classifier scores 99.34% accuracy. Keeping the
head does not make the task learnable; it just means the number must be reported with its context.

**Therefore, whenever a propaganda number is reported it carries `prop_n_pos_test` beside it**, and
the write-up labels it *not attempted at scale* rather than presenting it as a result.

`class_weight: balanced` on 7 positives would hand the positive class a ~150× multiplier and the
model would predict Yes for everything — hence the cap at 10×.

`prop_recruitment` has **zero** positives corpus-wide, so a sub-type head for it cannot exist. The
other three sub-types have 2 / 4 / 1. No sub-type head is trainable today.

**The actual path to a working propaganda classifier**, in priority order:

1. **Keyword mining → re-annotation.** Measured: the rubric's propaganda terms fire on 57
   articles and recover **5 of the 7 known positives (71% recall)**, leaving **52 currently-negative
   candidates** for review. This is the cheapest route to a real training set and it is what the
   keyword rubric is for. Caveat: `victimhood` is carrying almost no weight — "targeted" alone
   accounts for most of its 48 hits ("targeted strike", "targeted killing") and only 1 true
   positive. Tighten before mining (`exclude_low_precision: true`).
2. **Transfer pre-finetuning** on an external propaganda-technique corpus. SemEval-2020 Task 11 is
   the right fit conceptually (Flag-Waving → glorification, Loaded Language → justification, Appeal
   to Fear → victimhood). **Availability unverified** — that corpus historically required
   registering with the task organisers rather than a straight download. Fallbacks: `proppy`
   (article-level), SemEval-2021 Task 6.
3. **LLM zero-shot at pipeline stage 5** using the documented rubric criteria — a working detector
   today while the labels are grown.

---

## D4 · Keywords are a model input, measured as an ablation — 2026-09-22

**Decision.** Rubric keyword features are fused into the encoder at `[CLS]`, computed over the
**full untruncated document**. Arms: `off` / `group_counts` (11-dim) / `per_keyword` (~130-dim) /
`markers`. `group_counts` is the default.

**Why.** User directive — the rubric is the basis on which the data was classified, so the model
should be able to use it.

**What the evidence actually says.** On a linear bag-of-words test (5-fold CV, real data):

| Task (body_only) | keywords only | TF-IDF | TF-IDF + keywords |
|---|---|---|---|
| Narrative 5-class | 0.374 | **0.473** | 0.455 ↓ |
| Narrative 3-class | 0.500 | **0.627** | 0.580 ↓ |
| Severity F1(High) | 0.685 | 0.686 | **0.699 ↑** |

- **Severity is genuinely a keyword task.** 31 rubric terms alone score 0.685 versus 0.686 for a
  50,000-feature TF-IDF model. The groups are well calibrated: the High list fires on 232 articles
  where the High rate is 62.9% against a 22.8% base (2.8× lift); the Low list drops it to 15.3%.
- **For narrative, concatenating keywords onto TF-IDF hurts** — TF-IDF already contains every
  rubric word plus 50k more, so the extra columns are collinear noise. Per-class precision is poor
  for the rare classes (Investigation recall 0.667 but precision **0.103**; "evidence",
  "investigation", "operation" are everywhere in Security articles).

**Why fuse them into the transformer anyway.** The bag-of-words test cannot show the mechanism that
matters here. TF-IDF reads the whole article; the encoder at `max_length=512` does not:

| max_length | all rows seen in full | **bodied rows seen in full** |
|---|---|---|
| 256 | 47.2% | **2.6%** |
| 512 | 55.2% | **17.3%** |
| 1024 | 79.1% | 61.5% |
| 2048 | 94.6% | 90.0% |

Median bodied article is **859 tokens** (real XLM-R tokenizer). Keyword features computed over the
full document carry signal from the truncated tail that the encoder physically cannot reach. That
is the testable claim, and `off` is in the ablation set so it can fail.

**Honesty caveat for the write-up.** These keywords are the rubric the annotator used to produce
the labels. Keyword features therefore partly predict *the annotation process*, not only the world.
Legitimate — reproducing the annotation is the task — but it must be stated, and it means
keyword-fusion gains will not fully transfer to live unlabelled news.

---

## D5 · Group-aware folds, not row-dropping — 2026-09-22

**Decision.** Near-duplicates are clustered (char 3–5gram TF-IDF cosine ≥ 0.85 on
source-suffix-stripped headlines) and the whole cluster is assigned to one fold. Rows are not
dropped.

**Why.** Exact normalised-headline dedupe removes 11 rows but leaves **28 near-duplicate pairs /
20 redundant rows still leaking across folds**. Dropping rows would cost the rare classes examples
they cannot spare; grouping fixes the leak at no data cost.

Scale, for context: 31 redundant rows out of 1,064 (2.9%) — real, must be fixed, but it will not
overturn any result.

---

## D6 · Five contiguous narrative classes, not six — 2026-09-22

**Decision.** Train 5 classes with ids 0–4 built from `taxonomy.yaml`. `Radicalization` is declared
in the taxonomy but not trained.

**Why.** `Radicalization` has **zero** examples in the corpus and no keyword list in the rubric PDF.
The corpus's own `narrative_label` column is non-contiguous (`0=Security, 1=Civilian, 2=Political,
4=Investigation, 5=Other`; id 3 unused), so it is ignored entirely in favour of string → id mapping
through the taxonomy file. A sixth logit over zero examples is a permanently dead unit, and it
silently joins the macro-F1 denominator the moment the model predicts it once.

---

## D7 · Train on this laptop; CUDA build of PyTorch required — 2026-09-22

**Decision.** Primary and only training target is the user's laptop: **RTX 3050 Laptop 6 GB**,
Ryzen 5 7235HS (4c/8t), 11.7 GB RAM, Windows 11. Colab/Kaggle and the M2 Mac are backup only.

**Consequences.**
- The PyTorch currently installed in the old project is `2.10.0+cpu` — **CUDA unavailable**. The
  CUDA wheel must be installed from PyTorch's own index; plain `pip install torch` gives CPU-only.
- 6 GB VRAM: AMP fp16 is on by default. mmBERT-small runs batch 8 @ 512; xlm-roberta-base drops to
  batch 4 × accum 4 for the same effective batch of 16.
- `gradient_checkpointing` turns on for `max_length ≥ 1024`.
- `num_workers: 0` — Windows process-spawn overhead, only 4 physical cores.
- Ollama and training cannot share the GPU. Do not run the LLM stage during a training run.
- Everything macOS-specific in the reference scripts (`caffeinate`, `torch.backends.mps`,
  `~/osint_nlp/...` paths) is dropped.
