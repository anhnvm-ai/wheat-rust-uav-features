"""Figure 2 of the revised manuscript (fig:compact): compact feature subsets with nested member weights.

Reads only diagnostics_r1/exp_E_summary.json (written by exp_E_compact_nested.py summarise); nothing is recomputed. Same
content as the figure drawn by the summarise step (V7_compact_curve_nested.png), with the K = 30 label moved so that the
legend does not cover it. Band: Nadeau-Bengio corrected 95% CI of the mean Macro-F1 over the 50 outer folds.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path("/content/drive/MyDrive/Kaggle_Prepared/code/diagnostics_r1")
J = json.load(open(D / "exp_E_summary.json"))
assert J['gates']['E1_K149_equals_primary']['pass']
ks = J['K_grid']
ms = [100 * J['rows'][str(k)]['mean'] for k in ks]
lo = [100 * J['rows'][str(k)]['ci95'][0] for k in ks]
hi = [100 * J['rows'][str(k)]['ci95'][1] for k in ks]
full = ms[-1]
d30 = 100 * J['paired_vs_149']['30']['mean']

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.fill_between(ks, lo, hi, alpha=0.18, color='#2c7fb8', label='Corrected 95% CI of the mean')
ax.plot(ks, ms, 'o-', color='#2c7fb8', lw=2, label='Macro-F1 (nested member weights, repeated 10x5 CV)')
ax.axhline(full, ls='--', color='gray', lw=1, label=f'Full 149-feature model = {full:.1f}%')
ax.axvline(30, ls=':', color='#d95f0e', lw=1.5)
ax.annotate(f"K = 30: {ms[ks.index(30)]:.1f}%\n({d30:+.1f} points vs. K = 149)",
            xy=(30, ms[ks.index(30)]), xytext=(45, min(lo) + 3.0), fontsize=9, color='#d95f0e',
            arrowprops=dict(arrowstyle='->', color='#d95f0e', lw=1))
ax.set_xlabel("Number of features (K, top-ranked inside each outer training part)")
ax.set_ylabel("Macro-F1 (%)")
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(D / "V7_compact_curve_nested_fig2.png", dpi=150)
plt.close()
print("Saved V7_compact_curve_nested_fig2.png:", [(k, round(m, 1), round(a, 1), round(b, 1)) for k, m, a, b in zip(ks, ms, lo, hi)])
