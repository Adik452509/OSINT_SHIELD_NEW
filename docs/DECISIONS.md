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

### Amendment, 2026-09-25 (M3) — the rubric is the *only* propaganda signal

Measured across every mode and protocol:

| Model | propaganda F1(positive) |
|---|---|
| majority | 0.000 |
| `tfidf_logreg` (50k features) | **0.000** |
| `keyword_logreg` | 0.045 – 0.079 |
| **`keyword_rules`** | **0.100 – 0.150** |

Every *learned* method scores exactly zero, at every data size. The rubric rules are the only
thing that detects propaganda at all — and they do it with **no training data**, which means the
score does not degrade on unseen articles the way a 5-example classifier would.

0.15 is still a bad classifier and must not be reported as a result. But it vindicates keeping the
task alive, and it fixes the near-term design: **propaganda detection ships as rules + LLM
zero-shot**, with the trained head carried alongside as a diagnostic until mining and transfer
grow the positives. The keyword path is the product; the head is the experiment.

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

### Amendment, 2026-09-25 (M3) — apply the rubric, don't learn from it

M3 measured a pure **rule** baseline (argmax over keyword-group counts, *no learning at all*)
against a **learned** one (logistic regression over the same counts). The rule wins almost
everywhere, on `body_only` / clean folds:

| Task | `keyword_rules` | `keyword_logreg` | `tfidf_logreg` |
|---|---|---|---|
| narrative 5-class | **0.443** | 0.383 | 0.466 |
| narrative 3-class | **0.517** | 0.474 | 0.604 |
| severity F1(High) | 0.667 | **0.675** | 0.687 |
| propaganda F1(pos) | **0.118** | 0.063 | **0.000** |

Two consequences:

1. **Rule beats learned on narrative by ~6 points.** Eleven group-count features are too few for
   `class_weight="balanced"` logistic regression not to over-correct on a 77%-majority problem.
   The argmax rule encodes the inductive bias that actually generated the labels. So the fusion
   arm should hand the encoder the **raw counts** and let attention weigh them — not a
   pre-learned linear projection, and not a hard rule prediction.
2. **`keyword_rules` at 0.443 is within 0.023 of a 50,000-feature TF-IDF model** on 5-class
   narrative, using 61 terms and zero training. That is a strong argument that the rubric carries
   most of the available narrative signal, and it sets a floor the encoder must clear comfortably.

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

---

## D8 · Training schedule recalibrated for the smaller fold — 2026-10-02

**Decision.** `epochs: 10`, `patience: 3`, `min_epochs: 3` (plan: 5 / 2 / none). `amp_dtype: bf16`
(was fp16). Dynamic padding (`padding: longest`). Inner validation split is group-aware.

**Why.**

- **Epochs.** The plan's "5 epochs, best epoch typically 3 or 4" was calibrated on 744 training
  rows. Here the `body_only` inner-train split is ~335 rows — about 21 optimizer steps an epoch at
  effective batch 16, so 5 epochs is ~105 steps in total. That is likely undertrained. Doubling
  the ceiling costs nothing when early stopping still governs where the run ends.
- **`min_epochs`.** Randomly initialised heads often sit at majority-class F1 (0.172) for the first
  epochs while warmup completes. With patience 2 and no floor, a run can stop at epoch 3 *before
  it has started learning* — indistinguishable in the logs from genuine early convergence.
- **bf16.** Same memory as fp16, but with fp32's exponent range, so no loss scaling and no
  overflow. M1 confirmed the RTX 3050 supports it. The trainer falls back to fp16 automatically
  if bf16 is unavailable, and raises a descriptive error on any non-finite loss.
- **Dynamic padding.** Pads each batch to its own longest item. Results are identical; compute is
  not wasted on padding.
- **Group-aware inner split.** Uses `StratifiedGroupKFold`, so near-duplicates cannot straddle
  inner-train and inner-val. Otherwise early stopping would select an epoch on the strength of
  articles the model had effectively already seen — the D5 leak, one level down.

**Revisit** in M6 if the M5 learning curves show the best epoch consistently at the ceiling (raise
it) or at `min_epochs` (lower it).

---

## D9 · Freeze word embeddings, use SDPA attention — 2026-10-02

**Decision.** `model.freeze_embeddings: true` and `model.attn_implementation: sdpa`, for **both**
candidates, so the bake-off stays like-for-like.

**What happened.** The first M4 overfit run hit CUDA out-of-memory on batch 1 — for a model M1 had
measured at 3.66 GB on a 6 GB card.

**Two causes, both measured.**

1. **The RTX 3050 also drives the display.** `nvidia-smi` shows `Disp.A: On`, and M1's adapter
   list shows no active integrated GPU. Windows, the browsers, VS Code and every other window draw
   from the same 6 GB: **1.36 GB in use with nothing training**. Real headroom is ~4.6 GB, and it
   moves with whatever is open.
2. **M1's measurement was wrong.** M1 ran a single fp16 step. fp16's `GradScaler` starts at a
   scale of 65536, the first step overflows, and the optimiser step is *skipped* — so AdamW's two
   per-parameter state tensors were never allocated. Two real bf16 steps, with a padded mask:

| Model · attention · batch | full fine-tune | **frozen embeddings** |
|---|---|---|
| mmBERT-small · eager · 8 | 4.73 GB | — |
| mmBERT-small · sdpa · 8 | 3.36 GB (0.60 free) | **2.63 GB (2.24 free)** |
| mmBERT-small · sdpa · 4 | 2.66 GB | 1.78 GB |
| xlm-roberta-base · sdpa · 4 | **5.25 GB (0.00 free)** | **2.71 GB (2.24 free)** |

XLM-R does not fit at all without the freeze: weights + gradients + AdamW state come to ~4.4 GB
before a single activation.

**Why freezing is principled, not just a workaround.** It is the conclusion of the plan's own §3.1:
the word-embedding table is lookup rows, most of which this corpus never touches (192 M of XLM-R's
278 M parameters; ~98 M of mmBERT-small's 140 M), and the transformer body is what can learn — and
overfit. Freezing removes their gradients and AdamW state (~12 bytes per parameter) and removes
capacity to memorise a ~335-row fold. Position embeddings and the body stay trainable.

**Cost to record.** Domain vocabulary (Indic-script names, unit and weapon names) cannot adapt its
embeddings. Tested in M6 as an ablation: mmBERT-small with embeddings unfrozen fits at sdpa batch 4
(1.65 GB margin), so the comparison is runnable.

**Corrections this forces.**
- M1's VRAM figures and its "1024-token ablation is feasible" claim are withdrawn. Re-probe 1024
  with this configuration before M6 relies on it.
- Training must run with browsers and other GPU-heavy apps closed. `04_train_one_fold.py` now
  reports free VRAM before loading anything and warns below 3.5 GB.

---

## D10 · Early stopping watches both tasks — 2026-10-02

**Decision.** `training.monitor: combined` — the mean of inner-val narrative macro-F1 and severity
F1(High). Was `narrative_macro_f1`.

**Why.** In the M4 fold-0 run, the **inner-validation** curves (`history.csv`) diverged:

| epoch | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| val narrative macro-F1 | 0.165 | 0.233 | 0.290 | 0.251 | 0.389 | 0.389 | 0.300 | **0.502** |
| val severity F1(High) | 0.000 | 0.432 | **0.585** | 0.545 | 0.552 | 0.556 | 0.485 | 0.485 |

Monitoring narrative alone restored epoch 10, by which point severity had decayed from its peak.
A shared encoder with a single-task monitor trades the other task away by construction — that
argument holds independently of this one run's noise, which is why this change is made now and the
others are not.

**The rule this respects.** The decision uses inner-validation curves only. The fold-0 **test**
numbers (severity 0.596 against a 0.687 bar) are not an input: tuning against the outer test fold
leaks it into model selection and inflates every later number.

> **Superseded in part by D11:** `combined` originally averaged with severity **F1(High)**, which
> an untrained all-High predictor games. It now averages with severity **macro**-F1.

**Deliberately not changed yet.** Best epoch was the final epoch, with LR already decayed to zero —
D8's revisit trigger. But the inner-val set is 47 rows, roughly one example per rare class, and its
macro-F1 jumped 0.30 → 0.50 between the last two epochs; one run cannot separate undertraining
from noise. M5 records best epoch across 15 runs. If most land at the ceiling, an LR/epoch sweep is
M6's first job.

---

## D11 · M5 diagnosis: three defects before any bigger model — 2026-10-02

**Trigger.** M5 put mmBERT-small **below** the TF-IDF bar on clean CV — narrative 5-class
0.323–0.330 against 0.466, 0/5 folds won. The plan's rule (§5.1): *if the transformer does not
clear TF-IDF, look for a bug before looking for a better model.* Three defects were found.

**1. Wrong pooling for mmBERT.** mmBERT's own config ships `classifier_pooling: mean`; M5 used
`cls`. Position 0 is `<bos>`, which masked-LM pretraining never trains as a sentence summary, and 2
of every 3 ModernBERT layers use **local attention with a 128-token window** — so the `<bos>` vector
mostly sees the first ~64 tokens of a 512-token article. The plan's diagram says `[CLS]`, and that
was carried over without checking the model's design. **Fix:** `mmbert_small.yaml` → `pooling: mean`.
XLM-R keeps `cls`: every layer there is global, and RoBERTa-family classifiers read `<s>` natively.

**2. The D10 monitor was gameable.** `combined` averaged narrative macro-F1 with severity
**F1(High)**. At ~30% prevalence, an untrained head that predicts *everything* High scores F1(High)
≈ 0.46–0.55 on a 47-row inner-val set. Three of 30 runs restored such an untrained epoch-2/4
checkpoint and scored at or below majority on the test fold (narrative 0.096, 0.171, 0.172).
**Fix:** `combined` now uses severity **macro**-F1, where the all-High predictor scores ≈ 0.23.
F1(High) remains the reported metric.

**3. Selection on 47 rows is close to noise, and training is unstable.** Evidence:
- One run peaked at inner-val narrative 0.645 and scored **0.245** on its test fold.
- The same config and seed on the same fold scored **0.523** in M4 and **0.286** in M5. GPU kernels
  are not bit-deterministic (epoch-1 loss 3.1885 vs 3.1845), and on ~335 rows that divergence grows
  into a 0.24 swing — the known instability of small-data transformer fine-tuning.
- Training loss reaches ~0 (as low as 0.007): the model memorises its fold; generalisation, not
  capacity, is the constraint.

**Fix (measurement first):** `training.track_test_fold: true` scores the test fold every epoch
**for the record only** — after the selection decision, RNG-neutral, never read by it. The next run
then reports selected vs final-epoch vs best-possible scores and a mean learning curve, which says
whether to keep early stopping, switch to a fixed schedule, or enlarge the inner-val split.

**The development-set rule, restated.** From here the 5-fold CV is used to *choose* the
configuration, so its numbers become optimistic development estimates. That is exactly why
`test.csv` was held out in D1: the honest number is the single final evaluation on it. M4's caution
against tuning on a test fold applied while the CV was the evaluation; the role has now moved to
`test.csv`, which stays untouched.

**Keyword fusion is undecided, not refuted.** Paired over 15 runs: severity +0.024 (better in
10/15), narrative 3-class −0.052 (5/15). The std of the paired differences is 0.13–0.17 —
the instability is 5–10× the effect. One suggestive signal: pooled Investigation F1 **0.160 with
keywords vs 0.000 without**, the rare class whose rubric terms (FIR, NIA, chargesheet) are most
distinctive. Rerun once training is stable.

**Outcome (v2 rerun, 2026-10-03).** With mean pooling and the macro-F1 monitor, the `off` arm moved
from 0.330 to **0.472** narrative macro-F1 — level with TF-IDF (0.466). The monitor fix is bounded
at ≈ +0.01 (it removed one collapsed run), so ≈ +0.13 is attributable to pooling. No run collapsed;
best-epoch median 6, none at the ceiling.

---

## D12 · Keyword fusion stays on — directionally positive, not established — 2026-10-03

**Decision.** `keywords.mode: group_counts` remains the default for M6 and beyond.

**Evidence** (M5 v2, 15 paired runs — same fold, same seed, only the keyword arm differs):

| | Δ (keywords − off) | runs won |
|---|---|---|
| narrative 5-class | +0.017 ± 0.072 | 10/15 |
| narrative 3-class | +0.009 ± 0.061 | 10/15 |
| severity F1(High) | +0.010 ± 0.069 | 5/15 |

Not statistically established: a 10/15 split arises by chance ~30% of the time, and the 15 runs
are not independent (three seeds share each fold).

**Why keep it anyway.**
1. **Consistent on the class the rubric targets most sharply.** Pooled Investigation F1 0.329 vs
   0.254 (v2) and 0.160 vs 0.000 (v1) — two configurations, same direction.
2. ~~**Possibly stabilising.** Seed spread on narrative 0.004 with keywords, 0.041 without.~~
   **Withdrawn 2026-10-04:** the M6 fixed-schedule run, keywords on, has seed spread 0.035. The
   0.004 was chance — three seed means per arm cannot support a claim about stability.
3. **No measured cost** on any task, and it is the user's directive (D4).

**How to report it.** "Keyword fusion was directionally positive — chiefly on the rare,
lexically distinctive Investigation class — but the gain is within run-to-run variation at this
corpus size." Not "keywords improve accuracy."

---

## D13 · Fixed training schedule, no early stopping — 2026-10-04

**Decision.** `training.early_stopping: false`. No inner-validation split; every non-test row
trains; 10 epochs; the final epoch is the model. The M6 run `m6-fixed` is the new reference for
the remaining ablations.

**Evidence** — paired against the early-stopping reference (`v2`), same 15 (fold, seed) pairs:

| | early stopping | fixed 10 | Δ | runs better |
|---|---|---|---|---|
| narrative 5-class | 0.489 | 0.492 | +0.003 ± 0.076 | 7/15 |
| narrative 3-class | 0.625 | 0.629 | +0.004 ± 0.057 | 9/15 |
| severity F1(High) | 0.666 | 0.677 | +0.011 ± 0.045 | 9/15 |

On score it is a tie. Against TF-IDF on the identical folds: narrative 5-class **+0.026, 4/5 folds
— "beats the bar"** under the conservative rule, the first time; 3-class +0.025 (3/5); severity
−0.010, level.

**Why adopt it, given a tie.** The decisive reason is downstream: the final model in M9 trains on
*all* labelled data, leaving nothing to early-stop on — only a fixed recipe works there, and this
run shows the recipe costs nothing. It also returns 47 rows (14%) to training and removes a
selection step that 30 tracked runs showed adds nothing on average.

**The first unbiased learning curve.** Every run reached every epoch:
0.211 / 0.289 / 0.410 / 0.509 / 0.506 / **0.519** / 0.501 / 0.482 / 0.489 / 0.492.
Peak at epoch 6; epochs 7–10 give back ~0.027. Because the same 15 runs make every point, the
within-run comparison is reasonably trustworthy — but a 6-epoch schedule decays LR faster, so
`training.epochs=6` is its own ablation (queued, optional) rather than a reading of this curve.

**Pooled 5-class F1:** Security 0.878, Political 0.595, Civilian **0.500** (best so far),
Investigation 0.271, Other 0.286.

---

## D14 · Keep `max_length` 512 — truncation is measured, not assumed — 2026-10-04

**Decision.** `tokenizer.max_length: 512` stays.

**Evidence** — `m6-1024` paired against `m6-fixed`, same 15 (fold, seed). Truncated inputs fall
from **74.8% to 29.8%**:

| | 512 | 1024 | Δ | runs better |
|---|---|---|---|---|
| narrative 5-class | 0.492 | 0.478 | −0.014 ± 0.101 | 6/15 |
| narrative 3-class | 0.629 | 0.612 | −0.017 ± 0.053 | 5/15 |
| severity F1(High) | 0.677 | 0.688 | +0.011 ± 0.037 | 10/15 |

**Why.** No reliable gain on any task; severity moves the predicted way but within noise. Training
time doubles (69.9 vs 35.2 min) and so would live inference latency.

**This answers a question the plan posed.** §2.3: news follows the inverted-pyramid convention,
so "truncation is defensible — but it should be an ablation result rather than an assumption."
It now is: **at 512 tokens the cut tail carries no measurable classification signal** on this
corpus, for either task.

**Confound to state.** Batch 8 at 1024 left 0.34 GB free (`06_probe_vram.py`), so this arm ran
batch 4 × accumulation 4 against the reference's 8 × 2 — same effective batch of 16, but two
settings differ.

**Consequence for D4 (cross-reference).** The case for keyword fusion leaned on reaching past the truncation point.
Since the truncated tail turns out to carry little signal, keyword fusion's (small) value must
come from somewhere else — most plausibly the explicit rubric prior on rare, lexically distinctive
classes such as Investigation (D12).

---

## D15 · Final model: mmBERT-small, 6 epochs — 2026-10-05

**Decision.** The model carried into M7 and beyond:

| | |
|---|---|
| encoder | `jhu-clsp/mmBERT-small`, **mean pooling** (D11), SDPA attention, word embeddings **frozen** (D9) |
| input | headline `<eos>` body, **512 tokens** (D14), dynamic padding, `body_only` training data (D2) |
| keyword fusion | `group_counts`: 11 rubric-group counts, log1p, standardised on the training rows (D4, D12) |
| heads | narrative 5-class · severity 2-class · propaganda 2-class at loss weight 0.3, positive weight capped 10× (D3) |
| schedule | **fixed 6 epochs**, no early stopping (D13), AdamW lr 2e-5, 10% warmup then linear decay, weight decay 0.01, grad clip 1.0 |
| batch | 8 × accumulation 2 (effective 16), bf16 autocast |

**How it was chosen — a pre-registered rule.** Before any fair-round result existed, PROGRESS.md
fixed the rule: a model wins the head-to-head if it leads narrative 5-class by > 0.02 (paired)
and wins ≥ 9 of 15 paired runs; anything else is a tie resolved in favour of mmBERT-small
(half the parameters, faster live inference on a display-shared GPU, the plan's primary candidate).

**Fair round.** Round 1 ran both models at 10 epochs, a recipe tuned entirely on mmBERT. Each was
then given the epoch count its own learning curve indicated:

| paired, 15 runs | effect of the epoch change |
|---|---|
| mmBERT 10 → **6** | narrative **+0.032** (10/15), 3-class +0.017, severity +0.015 (10/15); std 0.089 → 0.058 |
| XLM-R 10 → 15 | narrative +0.011 (8/15), 3-class +0.007, severity −0.022 (6/15) — no real gain |

XLM-R's still-rising curve at epoch 10 was flattening; more epochs bought nothing.

**Head-to-head** — XLM-R (15) minus mmBERT (6): narrative −0.004 (6/15), 3-class +0.003 (9/15),
severity **−0.036 (3/15)**. Under the rule: a tie on narrative → mmBERT-small. Not a narrow call:
mmBERT wins severity in **12 of 15** paired runs, the most consistent signal in the bake-off, and is
steadier (narrative std 0.058 vs 0.093).

**Against TF-IDF, identical folds:** narrative 5-class **0.524 vs 0.466, +0.057, 4/5 — beats the
bar**; 3-class **0.646 vs 0.604, +0.042, 4/5 — beats the bar**; severity 0.692 vs 0.687, +0.005,
3/5 — matches. The first configuration to clear the plan's §5.1 test on two tasks.

**Winner's curse — stated before M7.** This recipe was selected from ~8 configurations scored on
the same 5 folds, so 0.524 is an optimistically biased estimate. The held-out `test.csv` score in
M7 is expected to land somewhat lower. That number, not this one, is the headline.

**Kept for the write-up, not used:** XLM-R as the comparison model (plan §3.1 asks for it
regardless of outcome); the 1024-token arm (D14); the early-stopping arm (D13).

---

## D16 · The held-out result, and what it decides — 2026-10-05

**The result** (full tables: `docs/FINAL_RESULTS.md`). D15 recipe trained on all 476 development
articles, scored once on the held-out set, 3 seeds:

| bodied held-out, n = 94 | encoder | TF-IDF | rubric | CV estimate |
|---|---|---|---|---|
| narrative 5-class | 0.415 ± 0.014 | 0.449 | 0.385 | 0.524 |
| narrative 3-class | 0.602 ± 0.039 | 0.673 | 0.513 | 0.646 |
| severity F1(High) | 0.664 ± 0.045 | 0.656 | 0.730 | 0.692 |

Paired bootstrap, encoder − TF-IDF: −0.023 [−0.181, +0.053], −0.040 [−0.133, +0.064],
+0.039 [−0.087, +0.171]. **Statistically indistinguishable.**

**The research conclusion.** On ~500 labelled articles with 74.3% inter-annotator agreement, a
fine-tuned 42 M-parameter multilingual encoder matches but does not beat a TF-IDF linear model.
The binding constraint is label quality: 85.3% accuracy where both annotators agreed, 50.0% where
they did not, with errors concentrated 2× on disagreed articles.

**What it decides for the system.**
1. **The encoder stays the deployed classifier.** At equal accuracy it is the only option that
   (a) covers Hindi and Kannada — TF-IDF knows only the English vocabulary it was fit on — and
   (b) emits the per-class confidence the plan's escalation rule (§6.2) routes on.
2. **Headline-only routing is an open question, and it is answered on development data, not
   test data.** The 65 held-out headline-only articles hint the rubric is stronger there (0.528
   vs 0.276 narrative) but are too few to decide (CI [−0.41, +0.14]). The ~415 headline-only
   articles in the development pool were never used to train the body-only model, so they give a
   6× larger, test-clean evaluation set → M7b.
3. **The held-out number is frozen.** No further changes are evaluated against `test.csv`.

**Deployed model:** seed 42, `runs/m7/model_seed42/` — fixed in advance, not chosen by score.

---

## D17 · Headline-only articles: hybrid narrative, and confidence is not to be trusted — 2026-10-05

**Decision.** For articles that arrive with no body:
- **narrative** = the keyword rubric's label when any narrative rubric term appears in the
  headline (34% of headlines), the encoder's label otherwise;
- **severity** = the encoder;
- these articles are **routed by type, not by confidence** (below).

Articles with a body are unchanged: the encoder decides both tasks.

**Evidence** — M7b, 415 development headline-only articles the bodied-only model never trained on;
`test.csv` untouched. Rule pre-registered in PROGRESS.md: switch from the encoder only on paired
bootstrap P ≥ 0.90.

| narrative | 5-class | 3-class | P(beats encoder) |
|---|---|---|---|
| encoder | 0.209 | 0.348 | — |
| TF-IDF | 0.177 | 0.296 | 0.01 |
| rubric | 0.208 | 0.370 | 0.49 |
| **hybrid** | **0.235** | **0.413** | **0.98** |

Severity: encoder 0.217, rubric 0.240 (P 0.58 — does not qualify), TF-IDF 0.050.

**Three findings beyond the routing choice.**

1. **The small held-out sample misled; the rule protected us.** On the 65 held-out headline-only
   articles the rubric scored 0.528 vs the encoder's 0.276. On 415 they are equal (0.208 vs
   0.209). The earlier gap rested on 1 Investigation and 2 Other articles. Acting on it would
   have deployed the wrong router.

2. **Headline-only classification is barely possible with anything we have.** A majority-class
   predictor scores ~0.18 on narrative 5-class here; the best method reaches 0.235, and TF-IDF
   sits at the majority level. A headline carries too little signal, and these labels were
   themselves assigned from the headline alone (annotator confidence 0.657). **The real remedy is
   fetching the article body for Google-News items — an M10 engineering task, and the
   highest-value one there.**

3. **The encoder is over-confident precisely where it is least accurate.** Mean narrative
   confidence is **0.872 on headline-only inputs versus 0.823 on bodied ones**; only 12% of
   headline-only articles fall below the 0.70 escalation threshold. Short inputs collapse to a
   confident "Security". The plan's confidence-based escalation (§6.2) would therefore let most
   headline-only errors through unexamined — hence routing these articles **by type**: every
   headline-only article carries a `headline_only` flag and `needs_review: true` regardless of
   confidence.

**Capacity finding for M10 (cross-reference).** On bodied held-out articles **56% would escalate** to the LLM (low
confidence or predicted High), not the plan's ~25% estimate. With Llama 3.2 sharing a 6 GB GPU
with the display, that is a throughput problem. The threshold must be calibrated against a
measured budget in M9/M10, as plan §6.2 anticipates — using development data, never `test.csv`.

---

## D18 · Propaganda re-annotation protocol — 2026-10-05

**Decisions (user, 2026-10-05).**
1. **Own-voice rule.** An article is propaganda only if its own voice — headline, narration,
   chosen framing — uses a technique. Reporting someone else's loaded rhetoric with attribution
   does not count. Full definitions: `docs/PROPAGANDA_GUIDE.md`.
2. **Review budget ≈ 100 articles**, reviewed by the user.

**Why re-annotate rather than mine.** Inspecting the 7 original positives showed:
- The rubric's propaganda terms reproduce those articles' phrases nearly verbatim ("brave sons",
  "shield in crisis", "stall India's programs", "appeasement politics", "vote bank"). Keyword
  recall of "5 of 7" (D3) is therefore **partly circular** — the list finds articles like the
  seven, and is weak evidence about propaganda in general.
- Three of seven rationales are a generic template ("Article contains justification elements in
  its framing").
- Two of seven are the same story (Bengal CM, Hindi and English) → six distinct positives.
- Under the own-voice rule some originals may not qualify — so all seven are re-reviewed.

**Protocol.**
- **Screen** all 1,064 articles with Llama 3.2 using the guide's definitions (not keywords),
  structured JSON output, temperature 0; evidence quotes are checked verbatim against the text.
- **Review sheet (~100)** = the 7 originals + rubric hits with low-precision terms removed (17)
  + the strongest Llama flags + a **random sample of unflagged articles** (to estimate how many
  positives the screen misses).
- **Blind review.** Rows are shuffled; the reviewer sees text only — not the source bucket, not
  Llama's answer, not the original label. The key is kept in a separate file.
- Positives that emerge are a new label set (`propaganda_v2`); the original column is kept.
