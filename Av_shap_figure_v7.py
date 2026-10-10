"""Publication-ready per-modality + top-feature SHAP figure for V7 (from V7_assets.json)."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=Path("/content/drive/MyDrive/Kaggle_Prepared/outputs/reviewer_experiments")
a=json.load(open(OUT/"V7_assets.json"))
top=a['shap_top20'][:15]; names=[t[0] for t in top][::-1]; vals=[t[1] for t in top][::-1]
pm=a['per_modality_shap_pct']; sp=a['shap_mean_spearman']
col={'ms':'#2c7fb8','rgb':'#d95f0e','hs':'#31a354'}
bar_c=[col.get(n.split('_')[0],'#888') for n in names]
fig,ax=plt.subplots(1,2,figsize=(12,5))
ax[0].barh(range(len(names)),vals,color=bar_c); ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names,fontsize=9)
ax[0].set_xlabel("Mean |SHAP| (ensemble-averaged)"); ax[0].set_title(f"(a) Top-15 features  (rank stability Spearman={sp:.2f})",fontsize=11)
mods=['MS','RGB','HS']; share=[pm['MS'],pm['RGB'],pm['HS']]
ax[1].bar(mods,share,color=[col['ms'],col['rgb'],col['hs']])
for i,v in enumerate(share): ax[1].text(i,v+1,f"{v:.0f}%",ha='center',fontsize=11)
ax[1].set_ylabel("Share of total |SHAP| (%)"); ax[1].set_ylim(0,90); ax[1].set_title("(b) Per-modality importance",fontsize=11)
plt.tight_layout(); plt.savefig(OUT/"V7_shap_permodality.png",dpi=160); plt.close()
print("Saved V7_shap_permodality.png  | per-modality:",pm," Spearman:",round(sp,3))
print("Top-5:",[(n,round(v,3)) for n,v in a['shap_top20'][:5]])
