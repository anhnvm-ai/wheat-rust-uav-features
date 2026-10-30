# Pre-registration: auxiliary analyses under the nested protocol, and hyperparameter selection (29 October 2026)

Written and committed **before** either experiment was run. The scripts committed with this file are the ones that will
be run: `exp_C_auxiliary_nested.py` (Experiment C) and `exp_D_nested_hyperparams.py` (Experiment D). Protocols, gates,
statistics and the wording for each possible outcome are fixed here and will not be changed after seeing results. Any
deviation will be reported as such. This follows `PREREG_final_robustness.md` (tag `pre_final_robustness`).

## Fixed in advance, whatever the results
- The primary analyses of the manuscript keep the HS block as submitted and the Table 4 hyperparameters. Rows with
  corrected HS features and the nested hyperparameter search are robustness checks; neither replaces the primary analysis.
- Tables 10 and 11 (compact feature subsets) are not re-run; they remain analyses with the original weighting rule, and
  the manuscript states this.

## What was already known when this was written
- Experiment A (`exp_A_summary.json`): nested 10x5 results for the full model (HS as submitted 66.7%, corrected 66.4%), HS
  only (submitted, corrected), MS only, RGB + MS and MS + corrected HS. These seven configurations are recomputed here as
  gates and must be identical.
- Experiment B (`exp_B_summary.json`): fair deep network on the 50 folds.
- The published values of Table 8 (3x5, weights on the scored fold), Table 9 (10x5, weights on the scored fold) and
  Figure 5 (aggregated out-of-fold probabilities with weights on the scored fold: ECE 0.023).
- `protocol_sensitivity_summary.json` (computed on 29 October 2026 from saved outputs, no model fitted): scored-fold epoch
  selection raised the deep network by 5.2 points (five folds), scored-fold ensemble weights changed the ensemble by 0.1 point.
- Code tests before this commit: one fold (r = 0, k = 0) of the primary analysis was recomputed by both scripts (identical to
  experiment A); the tuned and default rows were timed on one fold with permuted labels only.

## Common protocol (identical to the primary analysis and to experiment A)
50 outer folds, `StratifiedKFold(5, shuffle=True, random_state=42+r)`, r = 0...9, tiles in the order of the feature table
(= `folds_10x5.csv`). In each outer fold: StandardScaler fitted on the training part; inner stratified 5-fold CV (seed 42)
on the outer training part; member weight = inner out-of-fold Macro-F1 of the member; members refitted on the outer
training part; the outer fold is scored once. Column subsets keep the column order of the feature table.
Corrected HS features: `exp_A_corrected_hs_features.npy` (built by experiment A).

## Experiment C: Tables 8(a), 8(b), 9 under the nested protocol; Figure 5 from nested probabilities
**Configurations.** Table 8(a): full, without HS, without RGB, without MS, MS only, RGB only, HS only, with the HS block
as submitted (`_sub`, primary) and with corrected HS features (`_cor`). Table 8(b): the cumulative groups of
`Av_feature_ablation_v7.py` (band statistics 102; + MS indices and ratios 116; + further HS-block entries 132; full 149),
both HS versions. Table 9: full tile, canopy-masked (`features_v7_canopy.csv`), canopy-masked + 12 context descriptors
(`features_v7_canopy_plus.csv`); `_sub` uses the files as they are (HS block as in the original canopy analysis), `_cor`
replaces their HS columns by the corrected HS features of the full tile (not masked).
**Figure 5.** ECE (10 bins), multi-class Brier score, split-and-swap isotonic recalibration and mean confidence of correct and
wrong predictions, computed as in `Av_consolidated_eval.py` on the aggregated out-of-fold probabilities of experiment A
(S0 primary, S5 secondary). No model is fitted.

**Gates (stop and debug if any fails; nothing is interpreted before all pass).**
- R (run before this commit): the original-protocol values of Table 8 (3x5) and Table 9 (10x5), recomputed with the
  original column order and weighting, lie within 0.5 points of the published values (the environment of the original runs
  was not recorded, so exact equality is not expected). Result in `exp_C_gate_R.json`.
- C1: the seven configurations shared with experiment A (full, without HS, MS only, HS only, both HS versions where
  applicable) give per-fold nested and equal-weight Macro-F1 identical to experiment A (max |diff| < 1e-9).
- C2: the full model's per-fold Macro-F1 with weights on the scored fold equals `diag_nested_perfold.json` (< 1e-9).
- Folds equal `folds_10x5.csv`.

**Statistics.** Per configuration: mean ± SD (divisor n) over 50 folds, Nadeau-Bengio corrected 95% CI (J = 50, ratio 1/4,
t on 49 df). Paired differences (mean of fold-wise differences, same correction, two-sided p): Table 8(a) each row minus the
full model of the same HS version; Table 8(b) each step minus the previous step; Table 9 masked minus full (Δ1), masked +
context minus masked (Δ2), masked + context minus full (Δ3). Aggregated out-of-fold class-wise F1 for Table 9.

**Wording fixed in advance.** A difference is called detectable only if its corrected 95% CI excludes zero.

| Item | Outcome | Wording |
|---|---|---|
| Any Table 8 difference | CI excludes 0 | "changed Macro-F1 by X points (CI)" with its sign |
| Any Table 8 difference | CI includes 0 | "did not detectably change Macro-F1 (X points, CI)" |
| Table 9 | Δ1 CI < 0 and Δ2 CI > 0 | "Restricting the RGB and MS statistics to canopy pixels lowered Macro-F1, and adding canopy-context descriptors recovered it, so information on tile composition contributes to the predictive signal." |
| Table 9 | Δ1 CI < 0, Δ2 CI includes 0 | "Masking lowered Macro-F1; the context descriptors did not detectably recover it." |
| Table 9 | Δ1 CI includes 0 | "Restricting the statistics to canopy pixels did not detectably change Macro-F1 under the nested protocol; this check does not support a contribution of tile composition." |
| Table 9 | Δ1 CI > 0 | "Restricting the statistics to canopy pixels improved Macro-F1." |
| Figure 5 | raw ECE ≤ 0.05 | keep "low expected calibration error"; otherwise "an expected calibration error of X" |
| Figure 5 | isotonic ECE < raw ECE | "isotonic recalibration lowered ECE to X"; otherwise "did not improve ECE (X)" |

The HS-as-submitted results are the ones interpreted in the text; corrected-HS results are reported alongside and any
difference in conclusion between the two versions is stated.

## Experiment D: hyperparameter selection by nested cross-validation (Reviewer 4, Q2.1)
**Rows** (each with nested member weights, both HS versions; `sub` primary): `fixed` (Table 4 settings, i.e. the primary
analysis); `default` (library defaults; only random_state, n_jobs and logging set); `tuned` (per member, the setting with
the highest inner out-of-fold Macro-F1 on the outer training part is chosen from the grid; the member weight is that
inner Macro-F1; the chosen settings are refitted on the outer training part; ties go to the first setting in grid order).
**Grid.** LightGBM, XGBoost, RandomForest, ExtraTrees: max_depth {3, 5, 7} x n_estimators {100, 200, 300}, other settings
as in Table 4. SVM (RBF): C {0.3, 1, 3} x gamma {0.5, 1, 2} x gamma_scale, gamma_scale = 1 / (n_features x Var(X_train))
(scikit-learn's 'scale' rule). The deep network keeps its fixed settings.

**Gates.** D1: the `fixed` row equals experiment A per fold (S0 for `sub`, S5 for `cor`; max |diff| < 1e-9). D2 (run before
this commit, `exp_D_gate.json`): folds equal `folds_10x5.csv`; the gamma rule equals scikit-learn's 'scale' value.

**Statistics.** As in Experiment C (J = 50). Paired: tuned minus fixed (primary), default minus fixed (descriptive), and,
secondary, fair deep network (primary fold score of experiment B) minus tuned ensemble. Frequencies of the selected settings.

**Wording fixed in advance (Δ = tuned minus fixed, HS as submitted).**

| Outcome | Wording |
|---|---|
| CI includes 0 | "Selecting the hyperparameters by nested cross-validation did not detectably change Macro-F1 (Δ, CI); the estimate obtained with the fixed settings is close to that of a fully nested search over the tested grid." |
| CI entirely < 0 | "Nested selection gave lower Macro-F1 than the fixed settings (Δ, CI); estimates obtained with the fixed settings may be optimistic by about \|Δ\| points." |
| CI entirely > 0 | "Nested selection gave higher Macro-F1 (Δ, CI); the fixed settings were not tuned to these data. The primary analysis keeps the fixed settings, as fixed before the run." |
| DL minus tuned, CI includes 0 | "The comparison with the fair deep network is unchanged when the ensemble's hyperparameters are selected by nested cross-validation." |
| DL minus tuned, CI excludes 0 | the direction and size are stated, and the difference from the primary comparison is reported. |

## Result of gate R before this commit (30 October 2026)
Gate R did **not** pass. Thirteen of the fourteen original-protocol values were reproduced within 0.41 points, but
"without HS" (3x5) gave 66.78% against the published 66.23% (difference 0.55 points; tolerance 0.5). To find the cause,
the original functions of `V7_assets.py` (`cv_f1`, `mdls`) were copied verbatim and rerun in the current environment
(`exp_C_gate_R_verbatim_check.py`, `.json`): they give 66.778% (without HS) and 66.255% (full model), identical to this
script. This script therefore implements the original procedure; the gap to the published values comes from the computing
environment of the original runs (June 2026; package versions not recorded). The tolerance was not changed after seeing
the result; we proceed with this documented deviation, which is also stated in the repository README. It does not concern
the nested analyses, whose gates C1, C2 and D1 require exact equality with stored per-fold results. Before the commit, one
fold (r = 0, k = 0) was recomputed on the machine that will run the experiments: both scripts reproduced experiment A
exactly (same package versions as experiment A: numpy 2.1.3, scikit-learn 1.6.1, LightGBM 4.6.0, XGBoost 3.4.1).

## Order
Commit with tag `pre_aux_hyperparam`; the author pushes; then run, with `NJOBS=2` on the two-core Colab machine and in
any order, D (`sub`), C (Table 9 and Table 8 configurations), C `calib`, D (`cor`); `summarise`; check all gates; only
then edit the manuscript. Interrupted runs resume from the per-fold checkpoint files.
