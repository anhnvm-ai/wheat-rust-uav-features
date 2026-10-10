"""
R2_compact.py  —  Item 1: Compact Feature Selection (SAT round 2).

Claim target: a compact 30-feature model retains ~all of the 149-feature macro-F1 while
being markedly cheaper to train / serve  ->  supports the "lightweight, deployable" narrative.

Honest protocol (no selection leakage):
  * performance-vs-K curve: feature ranking is done IN-FOLD (LightGBM gain on the training
    fold only), top-K selected, full 5-model ensemble evaluated OOF, repeated 10x5.
  * cost: training time, per-sample inference latency, serialized model size at K=149 vs K=30.
  * stability: top-30 set across all 50 folds (mean pairwise Jaccard + per-feature frequency).
  * global ranking (full-data importance) only to NAME the features in the manuscript.

Output: outputs/reviewer_experiments/V7_compact.json  +  V7_compact_curve.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, time, io, pickle
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import r2_lib as L

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
X, y, fn, cols = L.load_features(OUT / "features_v7.csv")
cols = np.array(cols)
K_GRID = [10, 20, 30, 50, 70, 100, 149]
REPS = 10
print(f"X={X.shape}  K grid={K_GRID}  reps={REPS}")


def rank_infold(Xtr, ytr, topk=None):
    """LightGBM gain-importance ranking on a (standardized) training fold. Returns column order."""
    sc = StandardScaler(); a = sc.fit_transform(Xtr)
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=20, max_depth=5,
                           min_child_samples=15, subsample=0.7, colsample_bytree=0.7,
                           reg_alpha=1.0, reg_lambda=1.0, random_state=L.SEED, verbose=-1, n_jobs=1)
    m.fit(a, ytr)
    order = np.argsort(m.feature_importances_)[::-1]
    return order if topk is None else order[:topk]


# ---- performance-vs-K curve (one ranking per fold, reused for all K) + top-30 stability ----
print("\n[1] performance-vs-K curve (in-fold ranking, repeated CV)...")
t0 = time.time()
foldf1 = {k: [] for k in K_GRID}
top30_sets = []
for r in range(REPS):
    skf = StratifiedKFold(5, shuffle=True, random_state=L.SEED + r)
    for tr, va in skf.split(X, y):
        order = rank_infold(X[tr], y[tr])
        top30_sets.append(set(cols[order[:30]]))
        sc = StandardScaler(); a_full = sc.fit_transform(X[tr]); b_full = sc.transform(X[va])
        for k in K_GRID:
            sel = order[:k]
            a, b = a_full[:, sel], b_full[:, sel]
            mf1, mp = {}, {}
            for nm, md in L.make_models().items():
                md.fit(a, y[tr]); p = md.predict_proba(b)
                mf1[nm] = f1_score(y[va], p.argmax(1), average='macro'); mp[nm] = p
            ens = sum(mf1[n] * mp[n] for n in mf1) / sum(mf1.values())
            foldf1[k].append(f1_score(y[va], ens.argmax(1), average='macro'))
    print(f"  rep {r+1}/{REPS} done ({time.time()-t0:.0f}s)")

curve = {}
rng = np.random.RandomState(L.SEED)
for k in K_GRID:
    arr = np.array(foldf1[k]); boot = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(5000)]
    curve[k] = {'macroF1': float(arr.mean()), 'std': float(arr.std()),
                'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
full = curve[149]['macroF1']
for k in K_GRID:
    curve[k]['retention_pct'] = float(100 * curve[k]['macroF1'] / full)

# ---- top-30 stability across the 50 folds ----
freq = {}
for s in top30_sets:
    for f_ in s: freq[f_] = freq.get(f_, 0) + 1
freq = dict(sorted(freq.items(), key=lambda kv: -kv[1]))
jac = []
for i in range(len(top30_sets)):
    for j in range(i + 1, len(top30_sets)):
        a, b = top30_sets[i], top30_sets[j]
        jac.append(len(a & b) / len(a | b))
stability = {'mean_pairwise_jaccard': float(np.mean(jac)),
             'n_features_ever_in_top30': len(freq),
             'features_in_all_folds': int(sum(1 for v in freq.values() if v == len(top30_sets))),
             'frequency_top40': {k: v for k, v in list(freq.items())[:40]}}

# ---- global ranking (full data) to name the compact feature set ----
print("\n[2] global ranking + cost benchmark...")
order_global = rank_infold(X, y)
top30_global = cols[order_global[:30]].tolist()
top20_global = cols[order_global[:20]].tolist()


def benchmark(sel_cols):
    """Fit the 5-model ensemble on full data with given columns; return cost metrics."""
    Xs = StandardScaler().fit_transform(X[:, sel_cols])
    fitted = {}; t = time.time()
    for nm, md in L.make_models().items():
        md.fit(Xs, y); fitted[nm] = md
    fit_s = time.time() - t
    # inference latency: average wall time per sample for full ensemble predict_proba
    t = time.time()
    for nm, md in fitted.items(): md.predict_proba(Xs)
    infer_total = time.time() - t
    size = sum(len(pickle.dumps(md)) for md in fitted.values())
    return {'n_features': int(len(sel_cols)), 'train_seconds': float(fit_s),
            'infer_ms_per_sample': float(1000 * infer_total / len(y)),
            'model_size_kb': float(size / 1024)}


cost = {'full_149': benchmark(np.arange(X.shape[1])),
        'compact_30': benchmark(order_global[:30]),
        'compact_20': benchmark(order_global[:20])}

res = {'item': 'compact_feature_selection', 'protocol': 'in-fold LightGBM-gain ranking, top-K, '
       'repeated 10x5 CV macro-F1-weighted ensemble (locked V7 protocol)',
       'n_total_features': int(X.shape[1]), 'K_grid': K_GRID,
       'curve': {str(k): v for k, v in curve.items()},
       'full_macroF1': full,
       'compact_30_macroF1': curve[30]['macroF1'], 'compact_30_retention_pct': curve[30]['retention_pct'],
       'compact_20_macroF1': curve[20]['macroF1'], 'compact_20_retention_pct': curve[20]['retention_pct'],
       'stability': stability,
       'global_top30': top30_global, 'global_top20': top20_global,
       'cost': cost}
json.dump(res, open(OUT / "V7_compact.json", 'w'), indent=2)

# ---- figure: macro-F1 vs K with CI band + cost annotation ----
ks = K_GRID; ms = [curve[k]['macroF1'] * 100 for k in ks]
lo = [curve[k]['ci95'][0] * 100 for k in ks]; hi = [curve[k]['ci95'][1] * 100 for k in ks]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.fill_between(ks, lo, hi, alpha=0.18, color='#2c7fb8')
ax.plot(ks, ms, 'o-', color='#2c7fb8', lw=2, label='Macro-F1 (repeated 10x5 CV)')
ax.axhline(full * 100, ls='--', color='gray', lw=1, label=f'Full 149-feat = {full*100:.1f}%')
ax.axvline(30, ls=':', color='#d95f0e', lw=1.5)
ax.annotate(f"30 feat: {curve[30]['macroF1']*100:.1f}%\n({curve[30]['retention_pct']:.1f}% retained)\n"
            f"{cost['compact_30']['model_size_kb']:.0f} KB, {cost['compact_30']['infer_ms_per_sample']:.2f} ms/sample",
            xy=(30, curve[30]['macroF1'] * 100), xytext=(45, full * 100 - 4),
            fontsize=8, color='#d95f0e', arrowprops=dict(arrowstyle='->', color='#d95f0e'))
ax.set_xlabel("Number of features (K, top-ranked)"); ax.set_ylabel("Macro-F1 (%)")
ax.set_title("Compact feature selection: performance is flat from ~30 features")
ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "V7_compact_curve.png", dpi=150); plt.close()

print("\n" + "=" * 70)
print(f"Full 149-feat macro-F1 = {full*100:.2f}%")
for k in K_GRID:
    print(f"  K={k:3d}: {curve[k]['macroF1']*100:5.2f}%  (retain {curve[k]['retention_pct']:5.1f}%)  CI[{curve[k]['ci95'][0]*100:.1f},{curve[k]['ci95'][1]*100:.1f}]")
print(f"\nTop-30 stability: mean pairwise Jaccard={stability['mean_pairwise_jaccard']:.2f}, "
      f"{stability['features_in_all_folds']} feats in ALL 50 folds")
print("Cost  full149 -> compact30:")
print(f"  train  {cost['full_149']['train_seconds']:.2f}s -> {cost['compact_30']['train_seconds']:.2f}s")
print(f"  infer  {cost['full_149']['infer_ms_per_sample']:.3f} -> {cost['compact_30']['infer_ms_per_sample']:.3f} ms/sample")
print(f"  size   {cost['full_149']['model_size_kb']:.0f} -> {cost['compact_30']['model_size_kb']:.0f} KB")
print("Saved V7_compact.json + V7_compact_curve.png")
print("=" * 70)
