"""Additional comparison, NOT part of the pre-registration (added 6 November 2026).

Paired difference between the matched multimodal deep network (protocol B; fold score = mean Macro-F1 of the three
training seeds, the primary score of experiment B) and the feature ensemble with corrected HS features (configuration
S5 of experiment A, nested member weights) on the same 50 outer folds of the repeated 10x5 protocol.

PREREG_final_robustness.md tested only S5-S0 and S5-S3 in experiment A, and in experiment B the deep network against
the ensemble with the HS block as submitted. This further contrast was decided after both experiments had been run;
it is reported whatever its outcome and is described as not pre-specified. It uses the stored fold scores only
(exp_A_folds.jsonl, exp_B_seedlevel.jsonl); no model is re-run.

Statistics: Nadeau-Bengio corrected paired t, as in exp_B_dl_protocolB_10x5.py (J = 50, n_test/n_train = 1/4,
SD with divisor J - 1, t on 49 df, two-sided p).

Gates (all must pass before the contrast is computed):
  G1  the 50 (r, k) folds of S5 equal the 50 folds of the deep network, each with seeds 42, 43 and 44;
  G2  the S5 fold mean equals exp_A_summary.json (|diff| < 1e-12);
  G3  the deep-network fold scores equal exp_B_summary.json per_fold.primary (|diff| < 1e-12);
  G4  DL minus S0 (S0 from exp_A_folds.jsonl) reproduces paired_primary_DL_minus_ensemble of exp_B_summary.json
      (mean, CI, p; |diff| < 1e-9).

Wording fixed before the run (Delta = DL - S5):
  CI includes 0       "The same qualitative conclusion was obtained in an additional comparison in which the ensemble
                       used the corrected HS features."
  CI entirely < 0     "In an additional comparison in which the ensemble used the corrected HS features, the paired
                       difference favoured the feature ensemble; because this comparison was not pre-specified, it does
                       not replace the primary comparison."
  CI entirely > 0     "In an additional comparison in which the ensemble used the corrected HS features, the matched deep
                       network scored higher; the conclusion of the primary comparison therefore depends on the HS
                       representation."

Run: python3 diag_corrected_hs_vs_dl.py   (writes diag_corrected_hs_vs_dl.json next to this file)
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
WORDING = {
    'ci_includes_0': "The same qualitative conclusion was obtained in an additional comparison in which the ensemble "
                     "used the corrected HS features.",
    'ci_below_0': "In an additional comparison in which the ensemble used the corrected HS features, the paired "
                  "difference favoured the feature ensemble; because this comparison was not pre-specified, it does "
                  "not replace the primary comparison.",
    'ci_above_0': "In an additional comparison in which the ensemble used the corrected HS features, the matched deep "
                  "network scored higher; the conclusion of the primary comparison therefore depends on the HS "
                  "representation.",
}


def corrected(v, J, ratio=0.25):
    """Identical to corrected() in exp_B_dl_protocolB_10x5.py."""
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se; t = m / se
    return {'mean': float(m), 'sd_ddof0': float(v.std()), 'ci95': [float(m - h), float(m + h)],
            't': float(t), 'p_two_sided': float(2 * stats.t.sf(abs(t), J - 1))}


def main():
    A = [json.loads(l) for l in open(HERE / 'exp_A_folds.jsonl')]
    s5 = {(d['r'], d['k']): d['f1_nested'] for d in A if d['config'] == 'S5'}
    s0 = {(d['r'], d['k']): d['f1_nested'] for d in A if d['config'] == 'S0'}
    by = {}
    for line in open(HERE / 'exp_B_seedlevel.jsonl'):
        d = json.loads(line); by.setdefault((d['r'], d['k']), {})[d['seed']] = d['outer_macroF1']
    sumA = json.load(open(HERE / 'exp_A_summary.json')); sumB = json.load(open(HERE / 'exp_B_summary.json'))
    gates = {}

    keys = sorted(by)
    gates['G1'] = bool(len(keys) == 50 and sorted(s5) == keys and sorted(s0) == keys
                       and all(sorted(by[k]) == list(SEEDS) for k in keys))
    if not gates['G1']:
        raise SystemExit(f'G1 failed: {gates}')
    J = len(keys)
    e5 = np.array([s5[k] for k in keys]); e0 = np.array([s0[k] for k in keys])
    dl = np.array([np.mean([by[k][s] for s in SEEDS]) for k in keys])

    ref5 = sumA['configs']['S5']['nested']; ref5 = ref5['mean'] if isinstance(ref5, dict) else ref5
    gates['G2'] = bool(abs(e5.mean() - ref5) < 1e-12)
    pf = {(d['r'], d['k']): d['primary'] for d in sumB['per_fold']}
    gates['G3'] = bool(sorted(pf) == keys and max(abs(dl[i] - pf[k]) for i, k in enumerate(keys)) < 1e-12)
    rec = corrected(dl - e0, J); ref = sumB['paired_primary_DL_minus_ensemble']
    gates['G4'] = bool(abs(rec['mean'] - ref['mean']) < 1e-9 and abs(rec['p_two_sided'] - ref['p_two_sided']) < 1e-9
                       and all(abs(a - b) < 1e-9 for a, b in zip(rec['ci95'], ref['ci95'])))
    if not all(gates.values()):
        raise SystemExit(f'gate failed, contrast not computed: {gates}')

    res = corrected(dl - e5, J) | {'folds_DL_higher': int((dl > e5).sum())}
    lo, hi = res['ci95']
    outcome = 'ci_below_0' if hi < 0 else 'ci_above_0' if lo > 0 else 'ci_includes_0'
    out = {'note': 'Additional comparison, not part of the pre-registration (decided 6 November 2026, after experiments '
                   'A and B had been run); stored fold scores only, no model re-run.',
           'n_folds': J, 'gates': gates,
           'deep_network_primary': corrected(dl, J), 'ensemble_corrected_hs_S5': corrected(e5, J),
           'paired_DL_minus_S5': res, 'outcome': outcome, 'wording': WORDING[outcome],
           'check_DL_minus_S0_reproduced': rec,
           'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
           'per_fold': [{'r': k[0], 'k': k[1], 'deep_network': float(a), 'ensemble_S5': float(b)}
                        for k, a, b in zip(keys, dl, e5)]}
    json.dump(out, open(HERE / 'diag_corrected_hs_vs_dl.json', 'w'), indent=1)
    print('gates', gates)
    print(f"DL {out['deep_network_primary']['mean']*100:.1f}; ensemble S5 {out['ensemble_corrected_hs_S5']['mean']*100:.1f}; "
          f"DL-S5 {res['mean']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] p={res['p_two_sided']:.3f}; "
          f"folds DL higher {res['folds_DL_higher']}/{J}; outcome {outcome}")


if __name__ == '__main__':
    main()
