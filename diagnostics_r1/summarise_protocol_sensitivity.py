"""Sensitivity of the estimates to selection on the scored fold (Section 5.1). Fits no model; reads saved outputs only.

Deep network (five folds of repetition 0, seeds 42-44; diag_dl_earlystop.json and diag_dl_protoC.json):
  A = training epoch with the highest Macro-F1 on the scored fold, network trained on the full training fold;
  C = number of epochs chosen on an inner 15% split, network retrained on the full training fold;
  B = early stopping on the inner split, network trained on the remaining 85%.
  Fold score = mean over the three seeds; paired differences A - C and A - B with the corrected t interval (J = 5).
Ensemble (50 folds of the 10x5 protocol; diag_nested_perfold.json):
  member weights = Macro-F1 on the scored fold ('f1_leaky') versus inner-CV Macro-F1 ('f1_nested');
  paired difference with the corrected t interval (J = 50); number of folds with identical scores; weight ratios.
Usage: python3 summarise_protocol_sensitivity.py   (writes protocol_sensitivity_summary.json next to this file)"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

D = Path(__file__).resolve().parent


def corrected(d, J, ratio=0.25):
    d = np.asarray(d, float); m = d.mean(); s = d.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se
    return {'mean': float(m), 'ci95': [float(m - h), float(m + h)], 'p_two_sided': float(2 * stats.t.sf(abs(m / se), J - 1)), 'J': J}


E = json.load(open(D / 'diag_dl_earlystop.json')); C = json.load(open(D / 'diag_dl_protoC.json'))
a = np.array(E['A_bestepoch_on_valfold']['per_fold']).reshape(3, 5)       # seed-major: 3 seeds x 5 folds
b = np.array(E['B_inner_earlystop']['per_fold']).reshape(3, 5)
c = np.array(C['C_inner_epochs_retrain_full']['per_fold']).reshape(3, 5)
out = {'deep_network_rep0': {
    'means': {'A_scored_fold_epoch': float(a.mean()), 'B_inner_earlystop': float(b.mean()), 'C_inner_epochs_retrain': float(c.mean())},
    'A_minus_C': corrected((a - c).mean(0), 5) | {'folds_A_higher': int((a.mean(0) > c.mean(0)).sum())},
    'A_minus_B': corrected((a - b).mean(0), 5) | {'folds_A_higher': int((a.mean(0) > b.mean(0)).sum())}}}
pf = sorted(json.load(open(D / 'diag_nested_perfold.json'))['per_fold'], key=lambda d: (d['r'], d['k']))
L = np.array([d['f1_leaky'] for d in pf]); N = np.array([d['f1_nested'] for d in pf])
w = np.array([[d['weights_nested'][k] for k in sorted(d['weights_nested'])] for d in pf])
out['ensemble_50folds'] = {'means': {'scored_fold_weights': float(L.mean()), 'nested_weights': float(N.mean())},
                           'scored_minus_nested': corrected(L - N, 50) | {'folds_higher': int((L > N).sum()), 'folds_identical': int((L == N).sum()),
                                                                          'folds_lower': int((L < N).sum())},
                           'nested_weight_ratio_max_over_min': [float((w.max(1) / w.min(1)).min()), float((w.max(1) / w.min(1)).max())]}
json.dump(out, open(D / 'protocol_sensitivity_summary.json', 'w'), indent=1)
q = out['deep_network_rep0']['A_minus_C']; e = out['ensemble_50folds']['scored_minus_nested']
print(f"DL A-C {q['mean']*100:+.2f} [{q['ci95'][0]*100:+.2f}, {q['ci95'][1]*100:+.2f}] p={q['p_two_sided']:.3f} ({q['folds_A_higher']}/5)")
print(f"Ensemble {e['mean']*100:+.2f} [{e['ci95'][0]*100:+.2f}, {e['ci95'][1]*100:+.2f}] p={e['p_two_sided']:.3f}; identical on {e['folds_identical']}/50")
