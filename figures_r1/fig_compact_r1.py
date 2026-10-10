"""Compact-feature curve of the revised manuscript (fig:compact), redrawn during the revision (R4.Q3.2).

Reads only outputs/reviewer_experiments/V7_compact.json (curve written by R2_compact.py); nothing is
recomputed except the confidence band: the band is the Nadeau-Bengio corrected 95% CI of the mean
Macro-F1 over the 50 folds (as in Table 10 of the manuscript), replacing the bootstrap band of
R2_compact.py, which treated the folds as independent.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")
OUT = ROOT / "outputs/reviewer_experiments"
c = json.load(open(OUT / "V7_compact.json"))['curve']
ks = sorted(int(k) for k in c)
J = 50                                      # 10 repetitions x 5 folds
half_factor = stats.t.ppf(0.975, J - 1) * np.sqrt(1 / J + 1 / 4)   # n_test / n_train = 1/4


def ci(k):
    m = c[str(k)]['macroF1']
    sd = c[str(k)]['std'] * np.sqrt(J / (J - 1))   # saved SD uses ddof=0
    return 100 * (m - half_factor * sd), 100 * (m + half_factor * sd)


ms = [100 * c[str(k)]['macroF1'] for k in ks]
lo = [ci(k)[0] for k in ks]
hi = [ci(k)[1] for k in ks]
full = 100 * c['149']['macroF1']

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.fill_between(ks, lo, hi, alpha=0.18, color='#2c7fb8', label='Corrected 95% CI of the mean')
ax.plot(ks, ms, 'o-', color='#2c7fb8', lw=2, label='Macro-F1 (repeated 10x5 CV)')
ax.axhline(full, ls='--', color='gray', lw=1, label=f'Full 149-feature model = {full:.1f}%')
ax.axvline(30, ls=':', color='#d95f0e', lw=1.5)
ax.annotate(f"K = 30: {ms[ks.index(30)]:.1f}%\n({c['30']['retention_pct']:.1f}% of K = 149)",
            xy=(30, ms[ks.index(30)]), xytext=(42, min(lo) + 0.6), fontsize=9, color='#d95f0e',
            arrowprops=dict(arrowstyle='->', color='#d95f0e', lw=1))
ax.set_xlabel("Number of features (K, top-ranked inside each training fold)")
ax.set_ylabel("Macro-F1 (%)")
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "V7_compact_curve_r1.png", dpi=150)
plt.close()
print("Saved V7_compact_curve_r1.png:", [(k, round(m, 1), round(a, 1), round(b, 1)) for k, m, a, b in zip(ks, ms, lo, hi)])
