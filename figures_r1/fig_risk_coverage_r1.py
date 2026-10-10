"""Risk-coverage figure of the revised manuscript (fig:triage), redrawn during the revision (R3.5).

Reads only outputs/reviewer_experiments/V7_triage.json (risk_coverage_table, written by
R2_triage.py); nothing is recomputed. In-sample: aggregated out-of-fold probabilities, cut-off
set on the same 600 predictions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")
OUT = ROOT / "outputs/reviewer_experiments"
t = json.load(open(OUT / "V7_triage.json"))['risk_coverage_table']
rows = sorted(t.values(), key=lambda r: -r['coverage'])
cov = [100 * r['coverage'] for r in rows]

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(cov, [100 * r['macroF1'] for r in rows], 'o-', label='Macro-F1')
ax.plot(cov, [100 * r['accuracy'] for r in rows], 's-', label='Accuracy')
ax.plot(cov, [100 * r['rust_miss_rate'] for r in rows], '^-', label='Rust miss rate')
ax.axvline(80, color='grey', ls=':', lw=1.2)
ax.text(79.2, 45, 'illustrative\noperating point', fontsize=9, color='grey')  # x axis is reversed
ax.set_xlim(102, 48)
ax.set_xlabel('Coverage (% of tiles classified automatically;\nrest deferred for manual review)')
ax.set_ylabel('%')
ax.set_title('In-sample risk-coverage (aggregated OOF)', fontsize=10)
ax.legend(loc='center left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "V7_risk_coverage_r1.png", dpi=200)
plt.close()
print("Saved V7_risk_coverage_r1.png:", [(round(c), round(100 * r['macroF1'], 1), round(100 * r['accuracy'], 1),
                                         round(100 * r['rust_miss_rate'], 1)) for c, r in zip(cov, rows)])
