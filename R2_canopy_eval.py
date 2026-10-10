"""
R2_canopy_eval.py  —  Item 2 evaluation: full-tile vs canopy-aware ROI features.

Compares three configs under the identical locked V7 protocol (repeated 10x5 CV, in-fold scaler,
macro-F1-weighted ensemble):
  A  Full-tile 149  (outputs/.../features_v7.csv)             -- baseline
  B  Canopy-masked 149  (features_v7_canopy.csv)              -- same features, canopy support only
  C  Canopy-masked 149 + 12 ROI-context  (features_v7_canopy_plus.csv)
Reports macro-F1, CI, per-class F1 for each + the delta. Honest: if ROI does not help, we report
it as a (useful) negative result rather than hiding it.

Output: outputs/reviewer_experiments/V7_canopy.json + V7_canopy.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import r2_lib as L

OUT = Path("/content/drive/MyDrive/Kaggle_Prepared/outputs/reviewer_experiments")
CONFIGS = [('A_full_tile', 'features_v7.csv'),
           ('B_canopy_masked', 'features_v7_canopy.csv'),
           ('C_canopy_plus', 'features_v7_canopy_plus.csv')]

res = {}
for name, csv in CONFIGS:
    X, y, fn, cols = L.load_features(OUT / csv)
    print(f"\n[{name}] X={X.shape}")
    cv = L.repeated_cv(X, y, reps=L.REPS, verbose=True, label=name)
    pc = L.per_class_stats(y, cv['oof'].argmax(1))
    res[name] = {'n_features': int(X.shape[1]), 'macroF1_mean': cv['mean'], 'macroF1_std': cv['std'],
                 'ci95': cv['ci95'], 'oof_macroF1': cv['oof_macroF1'],
                 'per_class_F1': {c: pc[c]['F1'] for c in L.CLASSES}}

base = res['A_full_tile']['macroF1_mean']
for k in res: res[k]['delta_vs_full_pp'] = float((res[k]['macroF1_mean'] - base) * 100)
out = {'item': 'canopy_aware_roi', 'mask': 'Otsu-on-NDVI canopy mask, stats over canopy pixels',
       'protocol': 'repeated 10x5 CV, locked V7 ensemble', 'configs': res}
json.dump(out, open(OUT / "V7_canopy.json", 'w'), indent=2)

# figure
names = list(res.keys()); vals = [res[n]['macroF1_mean'] * 100 for n in names]
errs = [res[n]['macroF1_std'] * 100 for n in names]
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(names, vals, yerr=errs, capsize=4, color=['#7f7f7f', '#2c7fb8', '#31a354'], alpha=0.85)
ax.axhline(base * 100, ls='--', color='gray', lw=1)
for b, n in zip(bars, names):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
            f"{res[n]['macroF1_mean']*100:.1f}\n({res[n]['delta_vs_full_pp']:+.1f})", ha='center', fontsize=8)
ax.set_ylabel("Macro-F1 (%)"); ax.set_title("Canopy-aware ROI vs full-tile features")
ax.grid(alpha=0.3, axis='y'); ax.set_ylim(min(vals) - 4, max(vals) + 4)
plt.tight_layout(); plt.savefig(OUT / "V7_canopy.png", dpi=150); plt.close()

print("\n" + "=" * 70)
for n in names:
    r = res[n]
    print(f"{n:18s} {r['n_features']:3d}f  macroF1={r['macroF1_mean']*100:.2f}%  "
          f"(Δ{r['delta_vs_full_pp']:+.2f}pp)  perclassF1={ {c: round(v,3) for c,v in r['per_class_F1'].items()} }")
print("Saved V7_canopy.json + V7_canopy.png")
print("=" * 70)
