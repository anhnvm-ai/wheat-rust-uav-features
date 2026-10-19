# Pre-registration: final robustness experiments (19 October 2026)

Written and committed **before** either experiment was run. The scripts committed with this file are the
ones that will be run: `exp_A_corrected_hs_nested.py` and `exp_B_dl_protocolB_10x5.py`. Protocols, gates,
primary metrics and the wording for each possible outcome are fixed here and will not be changed after
seeing results. Any deviation will be reported as such.

## What was already known when this was written
- Primary analysis: ensemble with nested member weights, 149 features with the HS block as submitted,
  repeated 10x5 CV: Macro-F1 66.7% ± 3.8 (per-fold scores in `diag_nested_perfold.json`).
- `diag_leakage_hs.json` already contains the full model with **corrected** HS features and nested weights:
  fold mean 66.4% (per-fold scores not stored). The other corrected-HS rows (HS only, MS only, RGB+MS,
  MS + corrected HS) exist only with equal member weights (published as Table 8(c)).
- Fair multimodal network, protocol B, on the five folds of repetition 0 with three seeds: 65.2% ± 3.5
  (`diag_dl_earlystop.json`; NVIDIA A100). No other repetition has been run.

## Experiment A: corrected HS features under the primary nested protocol
**Questions.** (1) Δ_bug = F1(S5) − F1(S0); (2) Δ_HS = F1(S5) − F1(S3). Only these two paired contrasts are
tested; the other rows are descriptive.

**Configurations.** S0 RGB+MS+HS as submitted (149); S1 corrected HS only (70); S1b HS as submitted only
(70); S2 MS only (49); S3 RGB+MS (79); S4 MS + corrected HS (119); S5 RGB+MS+corrected HS (149).
Column subsets and order as in `diag_leakage_hs.py`.

**Protocol (unchanged from the primary analysis).** 50 outer folds, `StratifiedKFold(5, shuffle=True,
random_state=42+r)`, r = 0…9, tiles sorted by file name (= `folds_10x5.csv`). In each outer fold:
StandardScaler fitted on the outer training part; inner stratified 5-fold CV (seed 42) on the outer training
part; member weight = inner out-of-fold Macro-F1; five members (Table 4 settings) refitted on the outer
training part; the outer fold is scored once. Corrected HS features: cube read as rows × columns × bands,
first 125 bands of the 126-band files, same extractor (`feat_v7_lib.extract_hs_raw_features`).

**Gates (stop and debug if any fails; nothing is interpreted before all pass).**
- G1: S0 per-fold nested Macro-F1 equals `diag_nested_perfold.json` (max |diff| < 1e-9).
- G2: S5 nested fold mean equals `diag_leakage_hs.json` V7_fixedHS_full (|diff| < 1e-9).
- G3: equal-weight fold means equal `diag_leakage_hs.json` for every configuration (|diff| < 1e-9).

**Statistics.** Per configuration: mean ± SD (divisor n) over 50 folds and Nadeau–Bengio corrected 95% CI
(variance (1/J + 1/4)·s², s² with divisor J−1, J = 50, t on 49 df). Paired contrasts: mean of fold-wise
differences, same correction applied to their variance, two-sided p.

**Wording fixed in advance.**
| Outcome | Wording |
|---|---|
| \|S5−S0\| small, CI includes 0 | Correcting the HS representation did not materially change overall performance under the primary nested protocol. |
| S5−S3 CI includes 0, S5 ≤ S3 | Corrected HS features were informative alone but provided no detectable incremental benefit over RGB+MS under the tested protocol. |
| S5−S3 > 0, CI includes 0 | A numerical improvement of X points, although the corrected paired interval included zero. |
| S5−S3 CI entirely > 0 | Corrected HS features significantly improved Macro-F1 over RGB+MS under the tested protocol. |
| S5−S3 CI entirely < 0 | Adding corrected HS features reduced performance. |
| \|S5−S0\| > 3 points | The implementation error affected the main result; the main representation must be reconsidered. |

## Experiment B: fair multimodal network, protocol B, on all 50 folds
**Question.** Paired difference Δ_j = F1_DL,j − F1_ensemble,j over the 50 outer folds shared with the primary
analysis (ensemble per-fold scores from `diag_nested_perfold.json`; not re-run).

**Frozen configuration** (copied from `diag_dl_earlystop.py`, Table 5): HS bands 11–111 (101 channels), MS 5,
RGB 3; modality-specific 2-D encoders, late fusion, 255,203 parameters; Adam (lr 1e-3, weight decay 1e-4),
batch 32, label smoothing 0.1, random flips; at most 70 epochs; early stopping on Macro-F1 of a stratified 15%
split of the outer training part (random_state = seed), patience 12, best epoch restored; channel-wise
standardisation fitted on the training subset; outer fold scored once. No change to architecture or
hyperparameters whatever the result. Protocol C is not re-run (it stays a five-fold sensitivity analysis).

**Seeds and determinism.** Seeds 42, 43, 44 for every outer fold (150 fits); seeds reset for every
(repetition, fold, seed); cuDNN deterministic. The published repetition-0 runs (A100; interleaved with
protocol A) cannot be reproduced bit for bit; repetition 0 is re-run so that all 50 folds share one
environment, and the published values are kept for comparison.

**Primary fold score (author's choice):** mean of the three per-seed Macro-F1 values (single-network
performance, as in the published Table 6). **Secondary:** Macro-F1 of the seed-averaged probabilities.
Both are reported; if they lead to different conclusions, this is stated.

**Gates.** F1: DL folds equal `folds_10x5.csv` for all 50 folds (checked before training; passed on
19 Oct 2026 before this commit). F2: repetition-0 mean of the primary score within 3 points of the published
65.2%; otherwise stop and debug before running repetitions 1–9.

**Statistics.** As in Experiment A (J = 50, 49 df), for DL alone and for the paired difference. Aggregated
out-of-fold class-wise values: probabilities averaged over seeds, then over the ten repetitions.

**Wording fixed in advance (Δ = DL − ensemble).**
| Outcome | Wording |
|---|---|
| CI entirely < 0 | The paired difference favoured the feature ensemble under the tested protocol; this is specific to this small dataset and architecture and does not establish a general advantage of engineered features. |
| CI includes 0 | The interval includes zero; the experiment establishes neither superiority nor equivalence. |
| Δ > 0, CI includes 0 | The deep baseline was numerically higher by X points, but the corrected paired interval included zero. |
| CI entirely > 0 | The fair multimodal network outperformed the feature ensemble under the full repeated evaluation; the value of the feature pipeline lies in its explicit representation, CPU-only fitting, compact feature subset and modality analysis. |
