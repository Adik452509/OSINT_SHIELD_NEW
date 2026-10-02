# Final results — held-out evaluation (M7)

Evaluated once on 2026-10-02 17:13. Model: `jhu-clsp/mmBERT-small`, recipe D15. Trained on 476 articles; scored on the 159 held-out articles of `test.csv`, untouched since Phase 1.

## HELD-OUT, articles with body text — THE HEADLINE NUMBER (n = 94)

| task | encoder (3 seeds) | seed 42 (deployed) | TF-IDF | keyword rubric | majority | CV estimate |
|---|---|---|---|---|---|---|
| narrative 5-class | 0.415 ± 0.014 | 0.426 | 0.449 | 0.385 | 0.171 | 0.524 |
| narrative 3-class | 0.602 ± 0.039 | 0.633 | 0.673 | 0.513 | 0.285 | 0.646 |
| severity F1(High) | 0.664 ± 0.045 | 0.694 | 0.656 | 0.730 | 0.000 | 0.692 |
| propaganda F1(pos) | 0.000 | — | — | — | — | — |

Propaganda is diagnostic only: 2 positive(s) in this subset.

## held-out, headline-only articles — reported separately (n = 65)

| task | encoder (3 seeds) | seed 42 (deployed) | TF-IDF | keyword rubric | majority |
|---|---|---|---|---|---|
| narrative 5-class | 0.288 ± 0.010 | 0.276 | 0.225 | 0.528 | 0.225 |
| narrative 3-class | 0.416 ± 0.045 | 0.369 | 0.299 | 0.538 | 0.299 |
| severity F1(High) | 0.335 ± 0.106 | 0.308 | 0.000 | 0.000 | 0.000 |
| propaganda F1(pos) | 0.000 | — | — | — | — | 

Propaganda is diagnostic only: 0 positive(s) in this subset.

## held-out, all articles (n = 159)

| task | encoder (3 seeds) | seed 42 (deployed) | TF-IDF | keyword rubric | majority |
|---|---|---|---|---|---|
| narrative 5-class | 0.405 ± 0.007 | 0.413 | 0.424 | 0.411 | 0.174 |
| narrative 3-class | 0.574 ± 0.018 | 0.595 | 0.605 | 0.535 | 0.291 |
| severity F1(High) | 0.616 ± 0.031 | 0.635 | 0.588 | 0.667 | 0.000 |
| propaganda F1(pos) | 0.000 | — | — | — | — | 

Propaganda is diagnostic only: 2 positive(s) in this subset.

Held-out minus CV estimate, narrative 5-class: **-0.109** (the CV figure was the best of ~8 configurations on the same folds).

## Error analysis — deployed model, bodied held-out articles

**narrative accuracy by Claude/Llama agreement**

| group | n | accuracy |
|---|---|---|
| annotators disagreed | 26 | 0.500 |
| annotators agreed | 68 | 0.853 |

**severity accuracy by Claude/Llama agreement**

| group | n | accuracy |
|---|---|---|
| annotators disagreed | 27 | 0.630 |
| annotators agreed | 67 | 0.821 |

**narrative accuracy by annotator confidence**

| group | n | accuracy |
|---|---|---|
| < 0.70 | 7 | 0.571 |
| 0.70-0.85 | 44 | 0.795 |
| >= 0.85 | 43 | 0.744 |

**narrative accuracy by label status**

| group | n | accuracy |
|---|---|---|
| high_confidence | 69 | 0.812 |
| medium_confidence | 23 | 0.565 |
| review_needed | 2 | 1.000 |

Of 23 narrative errors, **56%** fall on articles the annotators disagreed on (base rate 28%), and **26%** match Llama's label.

Per-class F1, 5-class (deployed model, bodied): Civilian 0.667, Investigation 0.000, Other 0.000, Political 0.611, Security 0.853.

## Uncertainty — paired bootstrap (`scripts/07b_holdout_ci.py`)

4,000 resamples of the held-out articles. Reads the deployed model's saved predictions — the
encoder is not re-scored. TF-IDF and the rubric are deterministic refits.

**Bodied (n = 94).** Narrative classes present: Security 70, Political 14, Civilian 6,
**Investigation 2, Other 2**. High severity 37.

| | encoder − TF-IDF | 95% CI | P(encoder better) | encoder − rubric | 95% CI | P(encoder better) |
|---|---|---|---|---|---|---|
| narrative 5-class | −0.023 | [−0.181, +0.053] | 0.24 | +0.041 | [−0.078, +0.182] | 0.77 |
| narrative 3-class | −0.040 | [−0.133, +0.064] | 0.23 | +0.120 | [−0.036, +0.275] | 0.94 |
| severity F1(High) | +0.039 | [−0.087, +0.171] | 0.72 | −0.035 | [−0.177, +0.098] | 0.30 |

**Headline-only (n = 65).** Security 53, Political 9, Other 2, Investigation 1; High severity 7.
Rubric vs encoder, narrative 5-class: 0.528 vs 0.276, encoder − rubric −0.252 [−0.413, +0.136].
Too few rare-class articles to decide — see M7b.

## Interpretation

1. **The encoder and TF-IDF are statistically indistinguishable on unseen data.** Every paired CI
   includes zero. The cross-validation advantage (+0.057 narrative, D15) did not replicate.
2. **The 0.109 drop from the CV estimate is expected, not a defect:** the CV figure was the best
   of ~8 configurations on the same folds (winner's curse), and 5-class macro-F1 here rests on
   four rare-class articles — each one moves the score by ~0.1. The drop is consistent across
   all three seeds (0.426 / 0.423 / 0.395), and TF-IDF's own scores moved comparably between CV
   and held-out.
3. **Label ambiguity is the binding constraint.** Accuracy is 85.3% where Claude and Llama agreed
   and 50.0% where they disagreed; 56% of errors fall on disagreed articles, twice their 28%
   share. The model is not the bottleneck — the annotation ceiling (74.3% agreement) is.
4. **The headline number stands as reported.** Adjusting the model in response to these results
   would be test-set tuning.
