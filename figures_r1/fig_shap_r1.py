"""SHAP figure of the revised manuscript (fig:shap), relabelled during the revision (R1.5, R2.M1).

Reads only outputs/reviewer_experiments/V7_assets.json (no model is fitted); same data as
Av_shap_figure_v7.py, with labels that state what the values are: normalised mean |SHAP| of the
four tree-based members (the SVM is not included), shares by feature block, HS block as submitted.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")
OUT = ROOT / "outputs/reviewer_experiments"
a = json.load(open(OUT / "V7_assets.json"))
top = a['shap_top20'][:15]
names = [t[0] for t in top][::-1]
vals = [t[1] for t in top][::-1]
pm = a['per_modality_shap_pct']
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
plt.savefig(OUT / "V7_shap_permodality_r1.png", dpi=160)
plt.close()
print("Saved V7_shap_permodality_r1.png | shares:", pm, "| top-5:", [n for n, _ in a['shap_top20'][:5]])
