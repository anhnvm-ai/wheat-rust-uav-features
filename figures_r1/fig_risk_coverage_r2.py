"""Risk-coverage figure of the revised manuscript (fig:triage) with nested cut-offs (revision, October 2026).

Reads only code/diagnostics_r1/nested_rerun_summary.json (selective_prediction_nested, written by
summarise_nested_rerun.py from the outputs of diag_nested_perfold.py); nothing is refitted. In each of the
50 outer folds of the 10x5 protocol the cut-off for a target coverage is the quantile of the maximum class
probabilities of the inner out-of-fold predictions, applied once to the outer fold (single ensemble with
nested weights per fold). Points are means over the 50 outer folds, plotted at the realised coverage.
The in-sample figure of the original analysis is drawn by fig_risk_coverage_r1.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")
OUT = ROOT / "outputs/reviewer_experiments"
t = json.load(open(ROOT / "code/diagnostics_r1/nested_rerun_summary.json"))['selective_prediction_nested']
targets = sorted(t, key=lambda c: -float(c))
cov = [100 * t[c]['cov'] for c in targets]

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(cov, [100 * t[c]['f1'] for c in targets], 'o-', label='Macro-F1')
ax.plot(cov, [100 * t[c]['acc'] for c in targets], 's-', label='Accuracy')
ax.plot(cov, [100 * t[c]['rust_miss'] for c in targets], '^-', label='Rust miss rate')
x80 = 100 * t['0.80']['cov']
ax.axvline(x80, color='grey', ls=':', lw=1.2)
ax.text(x80 - 0.8, 45, 'illustrative\noperating point\n(target 80%)', fontsize=9, color='grey')  # x axis is reversed
ax.set_xlim(102, 48)
ax.set_xlabel('Realised coverage (% of tiles classified automatically;\nrest deferred for manual review)')
ax.set_ylabel('% (mean over 50 outer folds)')
ax.set_title('Risk-coverage with nested cut-offs', fontsize=10)
ax.legend(loc='center left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "V7_risk_coverage_r2.png", dpi=200)
plt.close()
print("Saved V7_risk_coverage_r2.png:", [(c, round(100 * t[c]['cov'], 1), round(100 * t[c]['f1'], 1),
                                         round(100 * t[c]['acc'], 1), round(100 * t[c]['rust_miss'], 1)) for c in targets])
