"""SHAP figure of the revised manuscript (fig:shap) from the October 2026 re-run of the SHAP step
(diagnostics_r1/diag_shap_nonconstant.py; four tree-based members refitted on all 600 tiles, HS block as
submitted). The original run (V7_assets.json, drawn by fig_shap_r1.py) could not be reproduced exactly with
the released code, so the manuscript now uses this re-run; its original values are given in Appendix A.
No model is fitted here."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")
OUT = ROOT / "outputs/reviewer_experiments"
d = json.load(open(ROOT / "code/diagnostics_r1/diag_shap_nonconstant.json"))
ens = np.array(d['ensemble_importance']); feat = d['feature_names']
order = np.argsort(ens)[::-1][:15]
names = [feat[i] for i in order][::-1]
vals = [float(ens[i]) for i in order][::-1]
pm = d['per_modality_shares_pct']
col = {'ms': '#2c7fb8', 'rgb': '#d95f0e', 'hs': '#31a354'}
bar_c = [col.get(n.split('_')[0], '#888') for n in names]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].barh(range(len(names)), vals, color=bar_c)
ax[0].set_yticks(range(len(names)))
ax[0].set_yticklabels(names, fontsize=9)
ax[0].set_xlabel("Normalised mean |SHAP| (four tree-based members)")
ax[0].set_title("(a) Top-15 features", fontsize=11)
blocks = ['MS', 'RGB', 'HS block\n(as submitted)']
share = [pm['MS'], pm['RGB'], pm['HS']]
ax[1].bar(blocks, share, color=[col['ms'], col['rgb'], col['hs']])
for i, v in enumerate(share):
    ax[1].text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=11)
ax[1].set_ylabel("Share of aggregated |SHAP| (%)")
ax[1].set_ylim(0, 90)
ax[1].set_title("(b) Share by feature block", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "V7_shap_permodality_r2.png", dpi=160)
plt.close()
print("Saved V7_shap_permodality_r2.png | shares:", pm, "| top-15:", names[::-1])
