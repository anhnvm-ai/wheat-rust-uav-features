# Code — Compact Red-Edge Features and Selective Prediction for UAV Wheat-Rust Scouting

These scripts produce the computed results of the manuscript (`main.tex`, Elsevier cas-dc), except the
items listed under Known gaps.
The pipeline is "V7": 149 engineered features (RGB 30 + MS 49 + HS 70) and a five-model soft-voting
ensemble (LightGBM, XGBoost, Random Forest, Extra Trees, SVM).

## Setup

**Data.** The data is the ICPR 2026 *Beyond Visible Spectrum* competition set. Each split has three folders:

```
<ROOT>/train/{RGB,MS,HS}/   600 labelled tiles (file stem = <Label>_...)
<ROOT>/val/{RGB,MS,HS}/     300 test tiles (labels withheld; scored on Kaggle)
<ROOT>/outputs/reviewer_experiments/   all results are written here
```

The image data and the extracted feature tables (`features_v7*.csv`) are not redistributed; they can be
regenerated from the competition data with the scripts below. The other output files listed in the tables
below are included in `outputs/`.

**Paths.** Every script has the data root hard-coded as
`ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")`. In `legacy_dl/config.py` the same root is
called `ROOT_DIR`. Edit that line to run the code somewhere else.

**Packages.** See `requirements.txt`. `Av_lime_v7.py` also needs `pip install lime`.
`B1_dl_baseline.py`, `diagnostics_r1/diag_dl_earlystop.py`, `diagnostics_r1/diag_dl_protoC.py`,
`diagnostics_r1/exp_B_dl_protocolB_10x5.py` and `legacy_dl/` need a GPU. Everything else runs on CPU.

Run the scripts from any directory. They import `feat_v7_lib.py` and `r2_lib.py` from this folder.
All evaluations use the same fixed protocol (defined in `r2_lib.py`):
- stratified CV, 5 folds repeated 10 times (seed 42 + r);
- `StandardScaler` fitted inside each fold;
- class order fixed as Health, Other, Rust;
- seed 42;
- the folds of every repetition are listed in `folds_10x5.csv` (written by `make_folds_10x5.py`); the matched
  deep baseline uses all 50 folds (its first, five-fold runs used repetition 0).

## Which script produces what

| Manuscript item (revised numbering) | Script | Output (`outputs/reviewer_experiments/`) |
|---|---|---|
| 149-feature extraction (Section 3.4; Table 3, `tab:featcount`) | `feat_v7_lib.py` (library), `V7_assets.py` | `features_v7.csv` |
| Table 2 (`tab:ndvi`), tile NDVI by class | computed from the `ms_NDVI` column (no separate script) | `features_v7.csv` |
| Table A1 (`tab:original`), weighted ensemble with validation-fold weights (as submitted) | `Av_consolidated_eval.py` | `V7_consolidated_eval.json` |
| Table C2 (`tab:otherprot`), single-model rows (SVM, ET, RF, LGB, XGB) | *no script, see Known gaps* | `V7_permodel.json` |
| Table A1, matched multimodal DL as submitted (epoch chosen on the scored fold) | `B1_dl_baseline.py` (GPU) | `B1_dl_baseline.json` |
| Table C2 (`tab:otherprot`), ResNet-18 and naive late fusion (results and class-wise values); Table C1 (`tab:naivedl`) | `legacy_dl/04_baseline_deep_learning.py` (GPU) | `outputs/results/dl_baseline_detailed.json` |
| Table A2 (`tab:checks`), panel (a), in-sample Health-prior weight α (the submitted operating-point table); original calibration (ECE 0.023, Appendix A; former Figure 5) | `Av_consolidated_eval.py` | `V7_consolidated_eval.json`, `V7_reliability_diagram.png` |
| Section 4.9 and Table 11 (`tab:fullcompact`), full-model late submission (0.790 private, 0.653 public) | `V7_honest_eval.py` | `submission_v7_repro.csv` |
| Original values of the modality ablation (3×5, weights on the scored fold; Appendix A); original SHAP values (Appendix A, "Original SHAP run") | `V7_assets.py` | `V7_assets.json` |
| Original values of the feature-group ablation (3×5, weights on the scored fold; Appendix A) | `Av_feature_ablation_v7.py` | `V7_feature_ablation.json` |
| Original SHAP figure (not in the revised manuscript; Figure 3 now uses the October 2026 re-run, see below) | `figures_r1/fig_shap_r1.py` (reads `V7_assets.json`; the submitted figure was drawn by `Av_shap_figure_v7.py`) | `V7_shap_permodality_r1.png` |
| Figure 4 (`fig:lime`) | `Av_lime_v7.py` | `V7_lime.json`, `V7_lime_fidelity.png` |
| Table 10 (`tab:compact`), Figure 2 (`fig:compact`) and the cross-validation column of Table 11: compact feature subsets with nested member weights (outer-fold-honest feature selection, ranking of `R2_compact.py`, selected columns in the order of the feature table; HS block as submitted) | `diagnostics_r1/exp_E_compact_nested.py` (pre-registered in `diagnostics_r1/PREREG_compact_nested.md`; hard gate E1: K = 149 must equal `exp_A_folds.jsonl` and `exp_A_probs.jsonl` fold by fold) | `diagnostics_r1/exp_E_summary.json`, `diagnostics_r1/exp_E_folds.jsonl`, `diagnostics_r1/exp_E_probs.jsonl`, `diagnostics_r1/exp_E_gate0.json`, `diagnostics_r1/V7_compact_curve_nested.png` |
| Figure 2 (`fig:compact`) as printed (same content as `V7_compact_curve_nested.png`, label moved clear of the legend) | `figures_r1/fig_compact_nested.py` (reads `diagnostics_r1/exp_E_summary.json`) | `diagnostics_r1/V7_compact_curve_nested_fig2.png` |
| Table B1 (`tab:compact30`; fixed 30-feature list and selection frequencies), compact-model cost in Table 16, original values of Table 10 (weights on the scored fold, columns in rank order; Appendix A) | `R2_compact.py` | `V7_compact.json` |
| Former Figure 2 from the original values (not in the manuscript) | `figures_r1/fig_compact_r1.py` (reads `V7_compact.json`) | `V7_compact_curve_r1.png` |
| Table 11 (`tab:fullcompact`), compact-30 late submission (test-set columns; historical, unchanged) | `R2_compact_submission.py` | `submission_compact30.csv`, `V7_compact_submission.json` |
| Canopy-masked feature tables used by Table 9; original values of the canopy analysis (weights on the scored fold; Appendix A) | `R2_canopy_extract.py`, then `R2_canopy_eval.py` | `features_v7_canopy*.csv`, `V7_canopy.json` |
| Table A3 (`tab:triage_insample`), panel (a), and the in-sample selective-prediction cell of Table A2 (`tab:checks`), panel (a): selective prediction of the original analysis | `R2_triage.py` | `V7_triage.json` |
| Table A3, panel (b): confusion matrices of the original analysis | full coverage: `Av_consolidated_eval.py` (`confusion_matrix`); 80% coverage: derived from the per-class counts and metrics in `V7_triage.json`; it is the only integer matrix consistent with them and with the full-coverage matrix (no cell can exceed its full-coverage count; no separate script) | `V7_consolidated_eval.json`, `V7_triage.json` |
| In-sample risk-coverage figure of the original analysis (not in the revised manuscript) | `figures_r1/fig_risk_coverage_r1.py` (reads `V7_triage.json`) | `V7_risk_coverage_r1.png` |
| Table 15 (`tab:proxyspatial`) | `R2_proxy_spatial.py` | `V7_proxy_spatial.json` |
| `tab:srsm` (submitted version only; removed in the revision) | `R2_srsm.py` | `features_v7_srsm.csv`, `V7_srsm.json` |

### Re-evaluations and post-review checks (October and November 2026)

| Manuscript item | Script | Output |
|---|---|---|
| Fixed folds of the 10×5 protocol (6,000 rows: `sample_id`, `label`, `repetition`, `fold`) | `make_folds_10x5.py` | `folds_10x5.csv` (this folder) |
| Table 6 (`tab:main_results`), weighted ensemble with nested weights; Table A2 (`tab:checks`), panel (a), nested α and selective-prediction cut-off; reference values for the gates of `exp_A_corrected_hs_nested.py` | `diagnostics_r1/diag_leakage_hs.py` | `diagnostics_r1/diag_leakage_hs.json` |
| Table 8 (`tab:ablation`), panel (c), and the corrected-HS row of Table 17: seven feature configurations (HS block as submitted or recomputed in the stored cube orientation) under the primary nested protocol, with corrected CIs and the two paired differences fixed in advance | `diagnostics_r1/exp_A_corrected_hs_nested.py` (pre-registered in `diagnostics_r1/PREREG_final_robustness.md`; its summary step checks gates G1–G3 against `diag_nested_perfold.json` and `diag_leakage_hs.json`) | `diagnostics_r1/exp_A_summary.json`, `diagnostics_r1/exp_A_folds.jsonl`, `diagnostics_r1/exp_A_probs.jsonl` |
| Per-fold ensemble scores used by the paired comparisons (Table 6, panel (b); Table A2, panel (b), note); Table 7 (`tab:perclass`), ensemble row (aggregated out-of-fold probabilities with nested weights); Table 12 (`tab:operating`), Health-prior weight α with nested selection; Tables 13 (`tab:triage`) and 14 (`tab:triagecm`), selective prediction with nested cut-offs | `diagnostics_r1/diag_nested_perfold.py` (re-runs the nested part of `diag_leakage_hs.py` for the evaluated pipeline and also stores the score, probabilities and selective-prediction results of every outer fold; it stops unless its means equal those stored in `diag_leakage_hs.json`), then `diagnostics_r1/summarise_nested_rerun.py` (also reads `diag_dl_earlystop.json` and `diag_dl_protoC.json`; fits no model) | `diagnostics_r1/diag_nested_perfold.json`, `diagnostics_r1/diag_nested_perfold_probs.npz`, `diagnostics_r1/nested_rerun_summary.json` |
| Figure 6 (`fig:triage`) | `figures_r1/fig_risk_coverage_r2.py` (reads `nested_rerun_summary.json`) | `V7_risk_coverage_r2.png` |
| Figure 3 (`fig:shap`) and the SHAP values of Section 4.5.1 (re-run of the SHAP step of `V7_assets.py`; agreement over all and over the non-constant features; package versions stored in the output) | `diagnostics_r1/diag_shap_nonconstant.py`, then `figures_r1/fig_shap_r2.py` | `diagnostics_r1/diag_shap_nonconstant.json`, `V7_shap_permodality_r2.png` |
| Tables 6 (panels (a) and (b)), 7 and 16, matched multimodal DL, protocol B (early stopping on an inner 15% split) on all 50 folds with three seeds per fold; Appendix A (repetition-0 check) | `diagnostics_r1/exp_B_dl_protocolB_10x5.py` (GPU; pre-registered in `diagnostics_r1/PREREG_final_robustness.md`) | `diagnostics_r1/exp_B_summary.json`, `diagnostics_r1/exp_B_seedlevel.jsonl`, `diagnostics_r1/exp_B_summary_rep0_gateF2.json` (summary after repetition 0, gate F2) |
| Table 6 (`tab:main_results`), panel (b), row with corrected HS features: deep network (protocol B, mean of the three seeds) minus the ensemble with corrected HS features (configuration S5 of `exp_A`) on the same 50 folds. Additional comparison, not part of the pre-registration (decided on 6 November 2026, after experiments A and B had been run); stored fold scores only, no model fitted; gates G1 to G4 check the inputs against `exp_A_summary.json` and `exp_B_summary.json`, and the wording for each outcome was fixed in the script before the run | `diagnostics_r1/diag_corrected_hs_vs_dl.py` | `diagnostics_r1/diag_corrected_hs_vs_dl.json` |
| Table A2, panel (b), protocol B on the five folds of repetition 0 (first run; reference of gate F2); Table A1 (same rule as submitted, re-run); training time on an A100 (Table 16, note a) | `diagnostics_r1/diag_dl_earlystop.py` (GPU) | `diagnostics_r1/diag_dl_earlystop.json` |
| Table A2, panel (b), protocol C (retrained on the full training fold; five folds of repetition 0) | `diagnostics_r1/diag_dl_protoC.py` (GPU) | `diagnostics_r1/diag_dl_protoC.json` |
| Table 8 (`tab:ablation`, all panels), Table 9 (`tab:canopy`), Figure 5 (`fig:reliability`) and the corresponding rows of Table 17: ablations, canopy masking and calibration under the primary nested protocol, HS block as submitted and corrected | `diagnostics_r1/exp_C_auxiliary_nested.py` (pre-registered in `diagnostics_r1/PREREG_aux_hyperparam.md`; its summary step checks gates C1 and C2 against `exp_A_folds.jsonl` and `diag_nested_perfold.json`; gate R, see Known gaps) | `diagnostics_r1/exp_C_summary.json`, `diagnostics_r1/exp_C_folds.jsonl`, `diagnostics_r1/exp_C_probs.jsonl`, `diagnostics_r1/exp_C_calibration.json`, `diagnostics_r1/V7_reliability_diagram_r3.png`, `diagnostics_r1/exp_C_gate_R.json` |
| Table A2 (`tab:checks`), panel (c): hyperparameters fixed, at library defaults, or selected by nested cross-validation (Reviewer 4, Q2.1) | `diagnostics_r1/exp_D_nested_hyperparams.py` (pre-registered in `diagnostics_r1/PREREG_aux_hyperparam.md`; gate D1 against `exp_A_folds.jsonl`) | `diagnostics_r1/exp_D_summary.json`, `diagnostics_r1/exp_D_folds.jsonl`, `diagnostics_r1/exp_D_gate.json` |
| Section 5.1, "Sensitivity to model selection" (paired differences computed from saved outputs; no model fitted) | `diagnostics_r1/summarise_protocol_sensitivity.py` | `diagnostics_r1/protocol_sensitivity_summary.json` |
| Tables 6, 7 and 16, MLP (256-128-3) on the 149 features | `B2_mlp_v7.py` | `outputs/reviewer_experiments/B2_mlp_v7.json` |
| Table 16 (`tab:compute`), feature-extraction time | `timing_r1/time_feature_extraction.py` | `timing_r1/timing_feature_extraction.json` |

Not produced by code:
- Table 17 (`tab:stages`) summarises the main findings of Tables 6, 8, 9, 10, 12, 13 and 15; no model was fitted for it.
- Figure 1 (`fig:block_diagram`) is drawn with TikZ in the manuscript source (`fig_pipeline.tex`). In the submitted version,
  it and `fig:flowchart` (now removed) were drawn by hand in draw.io and showed an earlier 226-feature version
  of the pipeline.
- Kaggle public and private scores were typed into the manuscript by hand from the competition's Submissions
  and Leaderboard pages. All scored files are late submissions, made after the competition had closed.

`original_v7/wheat_disease_v7_raw_nopca.py` is the original competition script (January 2026,
Windows paths). `feat_v7_lib.py` contains the same feature extractors in reformatted form: on the 600 training
tiles its output equals that of the original script's extractors (checked interactively in October 2026;
no script), and the timing run in `timing_r1/` reproduced `features_v7.csv` with it (max_abs_diff = 0).

## Run order

Steps marked **[raw]** read the image folders. All other steps only read `features_v7.csv`, so they
run in minutes.

1. `V7_assets.py` **[raw]**: builds `features_v7.csv`, which every later step needs.
2. `Av_consolidated_eval.py` (~6 min)
3. `Av_feature_ablation_v7.py` (~6 min), `Av_shap_figure_v7.py`, `B2_mlp_v7.py` (~15 s)
4. `Av_lime_v7.py` (~2 min): needs `V7_permodel.json`.
5. `R2_compact.py`, `R2_triage.py`, `R2_proxy_spatial.py`
6. `R2_canopy_extract.py` **[raw]**, then `R2_canopy_eval.py`; `R2_srsm.py` **[raw]**
7. Test-set submissions: `V7_honest_eval.py` **[raw]** and `R2_compact_submission.py` **[raw]**.
   `R2_compact_submission.py` needs `V7_compact.json` and `<ROOT>/submission_V7_repro.csv`.
8. GPU baselines: `B1_dl_baseline.py` **[raw]** and `legacy_dl/04_baseline_deep_learning.py` **[raw]**.
9. October 2026 re-evaluations and checks: `make_folds_10x5.py`, `B2_mlp_v7.py`,
   `diagnostics_r1/diag_leakage_hs.py` **[raw]**, `timing_r1/time_feature_extraction.py` **[raw]**, and on a GPU
   `diagnostics_r1/diag_dl_earlystop.py` **[raw]** and `diagnostics_r1/diag_dl_protoC.py` **[raw]**.
10. October 2026 re-runs on CPU: `diagnostics_r1/diag_nested_perfold.py` (about 26 min with 2 jobs; set `NJOBS`;
    finished folds are kept in a checkpoint file, so an interrupted run resumes), then
    `diagnostics_r1/summarise_nested_rerun.py` and `figures_r1/fig_risk_coverage_r2.py`;
    `diagnostics_r1/diag_shap_nonconstant.py`, then `figures_r1/fig_shap_r2.py`.
11. Final robustness experiments (protocols, gates and wording fixed in advance in
    `diagnostics_r1/PREREG_final_robustness.md`, git tag `pre_final_robustness`):
    `diagnostics_r1/exp_A_corrected_hs_nested.py <out_dir> run S0 S1 S1b S2 S3 S4 S5` **[raw]**, then `summarise`
    (CPU; about 100 min with 2 jobs, set `NJOBS`); on a GPU, `diagnostics_r1/exp_B_dl_protocolB_10x5.py <out_dir> run 0`
    **[raw]** and `summarise` (gate F2), then `run 1-9` and `summarise` (about 30 min on a Tesla T4). Both scripts keep
    finished folds in a checkpoint file, so an interrupted run resumes.
12. Further checks (protocols, gates and wording fixed in advance in `diagnostics_r1/PREREG_aux_hyperparam.md`, git tag
    `pre_aux_hyperparam`): `diagnostics_r1/exp_C_auxiliary_nested.py <out_dir> run all`, `calib` and `summarise`, and
    `diagnostics_r1/exp_D_nested_hyperparams.py <out_dir> run all` and `summarise` (CPU only; about 12 h in total on two
    CPU threads or about 3 h on eight; `diagnostics_r1/run_round4_chain.sh` runs them in sequence with `NJOBS` set to the
    number of CPU threads). Both scripts keep finished folds in a checkpoint file, so an interrupted run resumes; before a
    restart, `diagnostics_r1/check_round4_checkpoints.py` checks the checkpoint files, and on another machine
    `diagnostics_r1/verify_one_fold.py` recomputes a stored fold and requires identical values.
13. Compact feature subsets with nested member weights (protocol, gates and wording fixed in advance in
    `diagnostics_r1/PREREG_compact_nested.md`, git tag `pre_compact_nested`): `diagnostics_r1/exp_E_compact_nested.py
    <out_dir> gate` (gate E0, run before the lock), `run` and `summarise` (CPU only; about 25 min on eight CPU threads;
    `diagnostics_r1/run_round5_chain.sh` runs them after checking that the script equals the tagged version). The K = 149
    folds run first and the run stops at the first one that differs from the primary analysis.
14. Additional comparison, not part of the pre-registration (decided on 6 November 2026, after steps 11 and 13):
    `diagnostics_r1/diag_corrected_hs_vs_dl.py` (reads the stored fold scores of step 11; a few seconds on a CPU; its
    gates G1 to G4 and the wording for each outcome are given in the script).

## HS block: loader axis order

`feat_v7_lib.load_sample` moves the first axis of a cube to the end when `shape[0] < shape[2]`. The HS cubes are
stored as 32×32×125 (or 126) bands, so the loader turns them into 32×125×32, and `extract_hs_raw_features`
averages over columns and bands. The "mean spectrum" of the HS block is therefore a band-averaged intensity
profile over pixel rows (rows 11–18 after trimming), and 47 of the 70 HS entries are constant. The code is kept
unchanged, with a comment at the loader, so that the features match the evaluated pipeline (manuscript,
Section 3.4.3). `diagnostics_r1/diag_leakage_hs.py` and `diagnostics_r1/exp_A_corrected_hs_nested.py` recompute the HS
block with each cube read in its stored orientation; Table 8, panel (c), reports the latter (primary nested protocol).

## Known gaps

The following items are not reproduced by the scripts in this folder, or differ from what the submitted
manuscript stated.

- **Single-model rows of Table C2 (`tab:otherprot`)** (`V7_permodel.json`) come from an exploratory interactive run whose
  script was not preserved. The saved accuracies are multiples of 1/1800, consistent with three repetitions
  (3×5) rather than ten. `Av_lime_v7.py` reads this file for the LIME ensemble weights.
- **ResNet-18 and naive fusion** come from the older pipeline (`legacy_dl/`). They were evaluated in one 5-fold
  pass (seed 42), with training stopped on the accuracy of the fold that was scored; their folds cannot be
  reconstructed exactly.
- **Cost table.** The submitted `tab:compute` contained timings and parameter counts that no script measured. The
  revised table reports only measured values (`timing_r1/time_feature_extraction.py`, `R2_compact.py`,
  `B2_mlp_v7.py`, `diagnostics_r1/exp_B_dl_protocolB_10x5.py`, `diagnostics_r1/diag_dl_earlystop.py`).
- **Wilcoxon test.** The Wilcoxon signed-rank test cited in the submitted version was never run; it has been
  removed from the manuscript.
- **`tab:competition` of the submitted version.** The 112/300 row was a late submission from an older 311-feature
  pipeline (`A_gen_submission_v2.py`, not included), and the source of the 117/300 row could not be traced. The
  table has been removed from the revised manuscript.
- **Figures 1–2 of the submitted version** were drawn by hand (see "Not produced by code"): Figure 1 has been
  redrawn and Figure 2 removed.
- **Kaggle scores** were typed by hand (see "Not produced by code").
- **Original SHAP run.** The SHAP values of the submitted manuscript (`V7_assets.json`; Appendix A, "Original SHAP
  run") are not reproduced exactly by the released code: `diagnostics_r1/diag_shap_nonconstant.py`, which refits the
  same four models (their definitions are read from `V7_assets.py`) on the same features, gives the same five
  leading features but a mean pairwise Spearman correlation of 0.889 over all 149 features instead of 0.900. The
  cause was not identified, and the package versions of the original run were not recorded. The revised manuscript
  uses the re-run (Figure 3, Section 4.5.1); its package versions are stored in `diag_shap_nonconstant.json`.
- **File checks in Table 1 and Section 3.1** (no coordinate reference system or geotransform in the GeoTIFF files;
  MS digital numbers 0–11,406; 111 training and 48 test HS files with 126 bands; RGB minimum 0 in every channel;
  30 MS tiles with zero-valued pixels) were computed interactively; there is no script.
- **`V7_honest_eval.json` per-class F1** has Rust and Other swapped, because of the `LabelEncoder` ordering. Use
  `V7_consolidated_eval.json` instead.
- **Original ablation and canopy values.** Rerun in the current environment, the original procedures of `V7_assets.py`,
  `Av_feature_ablation_v7.py` and `R2_canopy_eval.py` reproduce the published original values within 0.55 points (gate R of
  `PREREG_aux_hyperparam.md`, tolerance 0.5 points: not met for one configuration, "without HS", 66.78% against 66.23%).
  The original code copied verbatim gives the same values as the new script (`exp_C_gate_R_verbatim_check.py`), so the
  gap comes from the unrecorded June 2026 environment. The manuscript now reports these analyses under the nested
  protocol (`exp_C_auxiliary_nested.py`), whose gates require exact equality with stored per-fold results.
- **Change of machine during the round-4 runs.** `PREREG_aux_hyperparam.md` specifies `NJOBS=2` on the two-core Colab
  machine. After `exp_D` with the HS block as submitted (150 folds) and the first 26 folds of `exp_C`, the run was stopped
  and resumed on an eight-thread Colab CPU (Intel Xeon 2.20 GHz, `NJOBS=8`, Python 3.13.18; the Python patch version of
  the first machine was not recorded, and the versions of the other packages were the same). Per-fold results do not
  depend on `NJOBS`: each fold is computed on its own with fixed seeds and single-threaded BLAS. Before resuming,
  `verify_one_fold.py` recomputed three stored folds on the second machine (one of `exp_C`, two tuned folds of `exp_D`)
  with identical values in every field, and gate D1 (exact equality with `exp_A`) passed for the fixed rows computed on
  each machine.
- **Original compact-feature values.** `R2_compact.py` (June 2026) kept the selected columns in rank order, so its
  K = 149 row (66.9%) differs from the original 10x5 evaluation of the same model (66.8%): with column subsampling in the
  tree members, a column permutation changes the fitted models. `exp_E_compact_nested.py` restores the canonical column
  order after selection, so its K = 149 row is the primary analysis. Rerun in the current environment (gate E0,
  `exp_E_gate0.json`), the original procedure selects exactly the same features as in June and reproduces the published
  Macro-F1 values within 0.17 points (June package versions not recorded).
- **Pre-deadline file and reproduction.** The pre-deadline prediction file `submission_v7.csv` (127 Rust) and the
  reproduced `submission_V7_repro.csv` (129 Rust) disagree on 4 of 300 tiles. The scored late submission is the
  reproduced file.

## Environment

`B2_mlp_v7.py` was run in October 2026 with the package versions pinned in `requirements.txt`. Its sanity check
reproduced the earlier 226-feature MLP number exactly (62.89% ± 3.2). The scripts in `diagnostics_r1/` and
`timing_r1/` and `make_folds_10x5.py` were also run in Google Colab in October 2026; `environment_postreview.txt`
lists the package versions and hardware of the final post-review runs (`exp_A_*`, `exp_B_*`; `exp_C_*` and `exp_D_*` used
the same versions of numpy, scikit-learn, SciPy, LightGBM and XGBoost on Colab CPUs, with Python 3.13.18 on the second
machine; see Known gaps; `exp_E_*` used the same versions with Python 3.13.18 on an eight-thread AMD EPYC 7B12 Colab CPU,
and its gate E1, exact identity of every K = 149 fold with `exp_A` run on an Intel CPU, passed).
`diagnostics_r1/diag_corrected_hs_vs_dl.py` was run in November 2026; its output stores the Python, numpy and SciPy
versions. The other scripts were run
in Google Colab in May–June 2026. Their package versions were not recorded.
