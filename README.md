# Code — Compact Red-Edge Features and Risk-Controlled Triage for Explainable UAV Wheat-Rust Scouting

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
`B1_dl_baseline.py`, `diagnostics_r1/diag_dl_earlystop.py`, `diagnostics_r1/diag_dl_protoC.py` and `legacy_dl/`
need a GPU. Everything else runs on CPU.

Run the scripts from any directory. They import `feat_v7_lib.py` and `r2_lib.py` from this folder.
All evaluations use the same fixed protocol (defined in `r2_lib.py`):
- stratified CV, 5 folds repeated 10 times (seed 42 + r);
- `StandardScaler` fitted inside each fold;
- class order fixed as Health, Other, Rust;
- seed 42;
- the folds of every repetition are listed in `folds_10x5.csv` (written by `make_folds_10x5.py`); the fair
  deep baseline uses repetition 0.

## Which script produces what

| Manuscript item (revised numbering) | Script | Output (`outputs/reviewer_experiments/`) |
|---|---|---|
| 149-feature extraction (Section 3.4; Table 3, `tab:featcount`) | `feat_v7_lib.py` (library), `V7_assets.py` | `features_v7.csv` |
| Table 2 (`tab:ndvi`), tile NDVI by class | computed from the `ms_NDVI` column (no separate script) | `features_v7.csv` |
| Table 6 (`tab:main_results`) and Table A1, weighted ensemble with validation-fold weights (as submitted) | `Av_consolidated_eval.py` | `V7_consolidated_eval.json` |
| Table 6, single-model rows (SVM, ET, RF, LGB, XGB) | *no script, see Known gaps* | `V7_permodel.json` |
| Table A1, fair multimodal DL as submitted (epoch chosen on the scored fold) | `B1_dl_baseline.py` (GPU) | `B1_dl_baseline.json` |
| Tables 6 and 7, ResNet-18 and naive late fusion; Table C1 (`tab:naivedl`) | `legacy_dl/04_baseline_deep_learning.py` (GPU) | `outputs/results/dl_baseline_detailed.json` |
| Table 7 (`tab:perclass`), ensemble row; Table 12 (`tab:operating`); Figure 5 (`fig:reliability`); ECE/Brier | `Av_consolidated_eval.py` | `V7_consolidated_eval.json`, `V7_reliability_diagram.png` |
| Section 4.9 and Table 11 (`tab:fullcompact`), full-model late submission (0.790 private, 0.653 public) | `V7_honest_eval.py` | `submission_v7_repro.csv` |
| Table 8 (`tab:ablation`), modality panel; SHAP values | `V7_assets.py` | `V7_assets.json` |
| Table 8 (`tab:ablation`), feature-group panel | `Av_feature_ablation_v7.py` | `V7_feature_ablation.json` |
| Figure 3 (`fig:shap`) | `figures_r1/fig_shap_r1.py` (reads `V7_assets.json`; the submitted figure was drawn by `Av_shap_figure_v7.py`) | `V7_shap_permodality_r1.png` |
| Figure 4 (`fig:lime`) | `Av_lime_v7.py` | `V7_lime.json`, `V7_lime_fidelity.png` |
| Table 10 (`tab:compact`), Table B1 (`tab:compact30`), compact-model cost in Table 16 | `R2_compact.py` | `V7_compact.json` |
| Figure 2 (`fig:compact`) and the corrected CIs of Table 10 | `figures_r1/fig_compact_r1.py` (reads `V7_compact.json`) | `V7_compact_curve_r1.png` |
| Table 11 (`tab:fullcompact`), compact-30 late submission | `R2_compact_submission.py` | `submission_compact30.csv`, `V7_compact_submission.json` |
| Table 9 (`tab:canopy`) | `R2_canopy_extract.py`, then `R2_canopy_eval.py` | `features_v7_canopy*.csv`, `V7_canopy.json` |
| Table 13 (`tab:triage`) | `R2_triage.py` | `V7_triage.json` |
| Table 14 (`tab:triagecm`) | full coverage: `Av_consolidated_eval.py` (`confusion_matrix`); 80% coverage: derived from the per-class counts and metrics in `V7_triage.json`; it is the only integer matrix consistent with them and with the full-coverage matrix (no cell can exceed its full-coverage count; no separate script) | `V7_consolidated_eval.json`, `V7_triage.json` |
| Figure 6 (`fig:triage`) | `figures_r1/fig_risk_coverage_r1.py` (reads `V7_triage.json`) | `V7_risk_coverage_r1.png` |
| Table 15 (`tab:proxyspatial`) | `R2_proxy_spatial.py` | `V7_proxy_spatial.json` |
| `tab:srsm` (submitted version only; removed in the revision) | `R2_srsm.py` | `features_v7_srsm.csv`, `V7_srsm.json` |

### Re-evaluations and post-review checks (October 2026)

| Manuscript item | Script | Output |
|---|---|---|
| Fixed folds of the 10×5 protocol (6,000 rows: `sample_id`, `label`, `repetition`, `fold`) | `make_folds_10x5.py` | `folds_10x5.csv` (this folder) |
| Table 6 (`tab:main_results`), weighted ensemble with nested weights; Table A2 (`tab:checks`), panels (a) nested α and selective-prediction cut-off and (c) HS block recomputed in the stored cube orientation | `diagnostics_r1/diag_leakage_hs.py` | `diagnostics_r1/diag_leakage_hs.json` |
| Tables 6, 7 and 16, fair multimodal DL, protocol B (early stopping on an inner 15% split); Table A1 (same rule as submitted, re-run); Table A2, panel (b) | `diagnostics_r1/diag_dl_earlystop.py` (GPU) | `diagnostics_r1/diag_dl_earlystop.json` |
| Table 6, fair multimodal DL, protocol C (retrained on the full training fold); Table A2, panel (b) | `diagnostics_r1/diag_dl_protoC.py` (GPU) | `diagnostics_r1/diag_dl_protoC.json` |
| Tables 6, 7 and 16, MLP (256-128-3) on the 149 features | `B2_mlp_v7.py` | `outputs/reviewer_experiments/B2_mlp_v7.json` |
| Table 16 (`tab:compute`), feature-extraction time | `timing_r1/time_feature_extraction.py` | `timing_r1/timing_feature_extraction.json` |

Not produced by code:
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

## HS block: loader axis order

`feat_v7_lib.load_sample` moves the first axis of a cube to the end when `shape[0] < shape[2]`. The HS cubes are
stored as 32×32×125 (or 126) bands, so the loader turns them into 32×125×32, and `extract_hs_raw_features`
averages over columns and bands. The "mean spectrum" of the HS block is therefore a band-averaged intensity
profile over pixel rows (rows 11–18 after trimming), and 47 of the 70 HS entries are constant. The code is kept
unchanged, with a comment at the loader, so that the features match the evaluated pipeline (manuscript,
Section 3.4.3). `diagnostics_r1/diag_leakage_hs.py` recomputes the HS block with each cube read in its stored
orientation (Appendix A).

## Known gaps

The following items are not reproduced by the scripts in this folder, or differ from what the submitted
manuscript stated.

- **Single-model rows of Table 6 (`tab:main_results`)** (`V7_permodel.json`) come from an exploratory interactive run whose
  script was not preserved. The saved accuracies are multiples of 1/1800, consistent with three repetitions
  (3×5) rather than ten. `Av_lime_v7.py` reads this file for the LIME ensemble weights.
- **ResNet-18 and naive fusion** come from the older pipeline (`legacy_dl/`). They were evaluated in one 5-fold
  pass (seed 42), with training stopped on the accuracy of the fold that was scored; their folds cannot be
  reconstructed exactly.
- **Cost table.** The submitted `tab:compute` contained timings and parameter counts that no script measured. The
  revised table reports only measured values (`timing_r1/time_feature_extraction.py`, `R2_compact.py`,
  `B2_mlp_v7.py`, `diagnostics_r1/diag_dl_earlystop.py`).
- **Wilcoxon test.** The Wilcoxon signed-rank test cited in the submitted version was never run; it has been
  removed from the manuscript.
- **`tab:competition` of the submitted version.** The 112/300 row was a late submission from an older 311-feature
  pipeline (`A_gen_submission_v2.py`, not included), and the source of the 117/300 row could not be traced. The
  table has been removed from the revised manuscript.
- **Figures 1–2 of the submitted version** were drawn by hand (see "Not produced by code"): Figure 1 has been
  redrawn and Figure 2 removed.
- **Kaggle scores** were typed by hand (see "Not produced by code").
- **File checks in Table 1 and Section 3.1** (no coordinate reference system or geotransform in the GeoTIFF files;
  MS digital numbers 0–11,406; 111 training and 48 test HS files with 126 bands; RGB minimum 0 in every channel;
  30 MS tiles with zero-valued pixels) were computed interactively; there is no script.
- **`V7_honest_eval.json` per-class F1** has Rust and Other swapped, because of the `LabelEncoder` ordering. Use
  `V7_consolidated_eval.json` instead.
- **Pre-deadline file and reproduction.** The pre-deadline prediction file `submission_v7.csv` (127 Rust) and the
  reproduced `submission_V7_repro.csv` (129 Rust) disagree on 4 of 300 tiles. The scored late submission is the
  reproduced file.

## Environment

`B2_mlp_v7.py` was run in October 2026 with the package versions pinned in `requirements.txt`. Its sanity check
reproduced the earlier 226-feature MLP number exactly (62.89% ± 3.2). The scripts in `diagnostics_r1/` and
`timing_r1/` and `make_folds_10x5.py` were also run in Google Colab in October 2026. The other scripts were run
in Google Colab in May–June 2026. Their package versions were not recorded.
