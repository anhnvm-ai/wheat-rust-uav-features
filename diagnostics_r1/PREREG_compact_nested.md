# Pre-registration: compact feature subsets (Table 10) with nested member weights (31 October 2026)

Written and committed **before** the experiment was run. The script committed with this file is the one that will be run:
`exp_E_compact_nested.py` (Experiment E). Protocol, gates, statistics and the wording for each possible outcome are fixed
here and will not be changed after seeing results. Any deviation will be reported as such. This follows
`PREREG_final_robustness.md` (tag `pre_final_robustness`) and `PREREG_aux_hyperparam.md` (tag `pre_aux_hyperparam`).

## Why, and what changes relative to `PREREG_aux_hyperparam.md`
`PREREG_aux_hyperparam.md` stated that Tables 10 and 11 would not be re-run and would keep the original weighting rule. After
experiments C and D were complete, an external evaluation (31 October 2026) recommended re-running Table 10 with nested
member weights, keeping its feature-ranking procedure. After that re-run, the compact-feature analysis is no longer an
exception to nested selection. This changes the earlier statement for Table 10 and for the cross-validation column of Table 11.

## Fixed in advance, whatever the results
- The primary analyses keep the HS block as submitted and the Table 4 hyperparameters. The compact-feature analysis uses the
  submitted 149-feature representation, for consistency with the original compact-model selection and the late test-set
  submission; corrected-HS robustness is evaluated separately in Table 8(c). No corrected-HS version of Table 10 is run.
- Table 11: the public/private test-set scores, the agreement and the timings are historical post-deadline submissions and
  measurements; they are retained unchanged. Only its cross-validation column is taken from this experiment.
- Table B1 keeps its fixed 30-feature list (LightGBM-gain ranking on all 600 tiles, June 2026). Its selection frequencies and
  stability statistics are updated from this experiment if they differ from the June values.
- The result is reported whatever it is. If the 30-feature subset is detectably worse than the full set, Table 10 and the text
  report this with the wording below; how the manuscript then describes the compact model (and whether the title changes) is
  decided by the authors afterwards, without changing the pre-registered result.

## What was already known when this was written
- The primary analysis (experiment A, S0): 66.7% nested Macro-F1, per fold in `exp_A_folds.jsonl` and `exp_A_probs.jsonl`.
- The published Table 10 and Figure 2 (`R2_compact.py`, run 14 June 2026; package versions not recorded): weights computed on
  the validation fold being scored and the selected columns kept in rank order. Macro-F1 65.6, 65.8, 66.6, 67.2, 66.9, 66.9
  and 66.9% for K = 10, 20, 30, 50, 70, 100 and 149; the K = 149 value (66.9%) differs from the original 10x5 evaluation of
  the same model (66.8%) because the columns were permuted. K = 100 was part of that run and of Figure 2 but not of the table.
- Code tests before this commit: one fold (r = 0, k = 0) at K = 149 was identical to experiment A in every checked field; one
  fold at K = 30 was run to check the output format (its score was not looked at).

## Protocol (Experiment E)
**Folds.** The 50 outer folds of the primary analysis: `StratifiedKFold(5, shuffle=True, random_state=42+r)`, r = 0...9,
tiles in the order of the feature table (= `folds_10x5.csv`).

**Feature selection (unchanged from `R2_compact.py`).** In each outer training part, the 149 features are ranked once by
LightGBM gain (`rank_infold`, copied verbatim: StandardScaler fitted on the training part; LightGBM with 150 trees, learning
rate 0.05, 20 leaves, maximum depth 5, minimum child samples 15, subsample 0.7, column subsample 0.7, L1 and L2 penalties 1.0,
seed 42; descending gain). The same ranking is used for every K in {10, 20, 30, 50, 70, 100, 149}, the grid of `R2_compact.py`.

**Implementation correction.** After selecting the top-K feature identities, columns will be restored to their canonical order
in the 149-feature table before model fitting. This prevents feature ordering from becoming an unintended experimental factor
and makes K = 149 an exact control that must reproduce the primary-analysis input matrix. The intended factor is the selected
feature subset, not a column permutation (with column subsampling in the tree members, a permutation changes the fitted models
for the same seed).

**Ensemble (as in the primary analysis).** StandardScaler fitted on the outer training part; member weight = Macro-F1 of the
member in an inner stratified 5-fold CV (seed 42) of the outer training part, on the selected features; five members with the
Table 4 settings refitted on the outer training part; the outer fold is scored once.

**Name and scope of the design.** Outer-fold-honest feature selection with nested member weighting (not "fully nested feature
selection"): feature selection is performed once within each outer-training partition, before the inner cross-validation
used to estimate ensemble weights. Consequently, the outer-fold estimate remains held out from both feature selection and
weight estimation, although the inner weight estimates are conditional on a feature subset selected using the full
outer-training partition. Re-ranking inside each inner fold would be a different experiment and is not run.

## Gates (nothing is interpreted before all pass)
- **E0 (before this commit; descriptive):** `R2_compact.py`'s procedure (rank-ordered columns, weights on the scored fold)
  rerun in the current environment and compared with the published June values and top-30 selection frequencies. The June
  package versions were not recorded, so exact agreement is not required; the result is recorded below before the commit.
- **E1 (hard reproducibility gate):** every K = 149 fold must equal the primary analysis (experiment A, S0) exactly: the
  same fold assignment and input matrix (all 149 columns in canonical order), nested member weights, Macro-F1, accuracy,
  equal-weight Macro-F1 and accuracy, confusion matrix and predicted probabilities (stored to 8 decimals), and the Macro-F1
  with weights on the scored fold must equal `diag_nested_perfold.json`; the mean must be 66.72084104320107%. The script
  checks each K = 149 fold as soon as it is computed (they are run first) and stops at the first mismatch. If E1 fails, no
  Table 10 result is used until the cause is found.
- **E2:** folds equal `folds_10x5.csv`.

## Statistics
Per K: mean ± SD (divisor n) of the 50 fold-wise nested Macro-F1 values, and the Nadeau-Bengio corrected 95% CI of the mean
(J = 50, ratio 1/4, t on 49 df; Figure 2 band). Paired: Δ_K = Macro-F1(K) minus Macro-F1(149), the mean of the 50 fold-wise
differences, with its corrected 95% CI, two-sided p and the number of folds with Δ > 0. Top-30 selection frequencies over the
50 folds, mean pairwise Jaccard similarity, number of distinct features and number selected in every fold.

**Table 10 layout.** K | Macro-F1 (%) (mean ± SD) | Δ vs. K = 149 (points) | corrected 95% CI of Δ, for all seven K. The
retention column is dropped.

## Wording fixed in advance
A difference is called detectable only if its corrected 95% CI excludes zero.

| Item | Outcome | Wording |
|---|---|---|
| K = 30 vs. 149 | CI includes 0 | "Using 30 selected features did not detectably change Macro-F1 relative to the full 149-feature representation (Δ points, corrected 95% CI)." |
| K = 30 vs. 149 | CI entirely < 0 | "Using 30 selected features lowered Macro-F1 by \|Δ\| points relative to the full 149-feature representation (corrected 95% CI)." |
| K = 30 vs. 149 | CI entirely > 0 | "Using 30 selected features raised Macro-F1 by Δ points relative to the full 149-feature representation (corrected 95% CI)." |
| Other K | — | One sentence listing the K whose CI excludes 0, with direction ("lowered"/"raised"), or stating that no subset size differed detectably from K = 149. |
| Table 11 note | — | "Cross-validation values were recomputed under the revised outer-fold evaluation of Table 10. Public/private test scores are historical post-deadline submissions obtained with the original fixed 30-feature model and are retained unchanged for traceability." (with the full model named alongside) |
| Method | — | The design is described with the "Name and scope" sentences above. |

## Result of gate E0 before this commit (31 October 2026)
`exp_E_compact_nested.py <out_dir> gate` (file `exp_E_gate0.json`) reran `R2_compact.py`'s procedure in the current
environment (Python 3.13.18, numpy 2.1.3, scikit-learn 1.6.1, SciPy 1.16.3, LightGBM 4.6.0, XGBoost 3.4.1):
- Feature selection is reproduced exactly: the top-30 selection frequencies stored in June (40 most frequent features), the
  mean pairwise Jaccard similarity (0.544), the number of distinct features (62) and the number selected in every fold (6)
  are identical, so the in-fold ranking used here is the one of the published analysis and Table B1 is unchanged.
- Macro-F1 of the original procedure (rank-ordered columns, weights on the scored fold): 65.59, 65.81, 66.63, 67.27, 66.89,
  66.92 and 66.73% for K = 10, 20, 30, 50, 70, 100 and 149, against the published 65.58, 65.76, 66.61, 67.22, 66.86, 66.88
  and 66.89%: within 0.17 points (largest at K = 149; the June package versions were not recorded). As stated above, exact
  agreement was not required; the published values are kept as the original values.
- Fold r = 0, k = 0 at K = 149 with canonical column order was identical to experiment A in every field checked by gate E1.

## Order
Commit with tag `pre_compact_nested`; the authors push; then run `exp_E_compact_nested.py <out_dir> run` (K = 149 first) and
`summarise`; check all gates; only then edit the manuscript. Interrupted runs resume from the per-fold checkpoint files.
