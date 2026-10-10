"""
R2_proxy_spatial.py  —  Item 4: Proxy Spatial Validation (SAT round 2).

The competition tiles carry no GPS / plot id (filenames are just <class>_hyper_<n>), so a true
spatial-block CV is impossible. As an honest proxy we group spectrally-similar tiles (K-means on
their spectral signature) and run LEAVE-ONE-CLUSTER-OUT CV: a held-out cluster is spectrally
distinct from everything the model trained on, the same way a spatially-separated field block
would be. The gap vs ordinary stratified CV upper-bounds optimism from spectral autocorrelation.

We aggregate predictions over held-out clusters into a single OOF set (each sample predicted once)
so class imbalance inside a cluster never breaks macro-F1. Reported alongside the stratified
baseline; any drop goes straight into Limitations.

Output: outputs/reviewer_experiments/V7_proxy_spatial.json  +  V7_proxy_spatial.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import r2_lib as L

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
X, y, fn, cols = L.load_features(OUT / "features_v7.csv")
cols = np.array(cols)

# spectral signature for clustering = MS band means + HS sampled spectrum (label-agnostic content)
spec_cols = [i for i, c in enumerate(cols)
             if c in ('ms_blue_mean', 'ms_green_mean', 'ms_red_mean', 'ms_rededge_mean', 'ms_nir_mean')
             or c.startswith('hs_spec_')]
S = StandardScaler().fit_transform(X[:, spec_cols])
print(f"X={X.shape}  spectral-signature dims={len(spec_cols)}")


def grouped_oof_macroF1(groups):
    """Leave-one-group-out: each sample predicted once (group held out). Returns OOF macro-F1."""
    oof = np.zeros((len(y), 3))
    for g in np.unique(groups):
        tr = groups != g; va = groups == g
        if tr.sum() < 50 or len(np.unique(y[tr])) < 3:  # need all 3 classes to train sanely
            return None
        sc = StandardScaler(); a = sc.fit_transform(X[tr]); b = sc.transform(X[va])
        mf1, mp = {}, {}
        for nm, md in L.make_models().items():
            md.fit(a, y[tr]); p = md.predict_proba(b)
            mf1[nm] = f1_score(y[tr], md.predict(a), average='macro'); mp[nm] = p
        ens = sum(mf1[n] * mp[n] for n in mf1) / sum(mf1.values())
        oof[va] = ens
    return float(f1_score(y, oof.argmax(1), average='macro'))


# ---- baseline: ordinary stratified repeated CV (the locked headline protocol) ----
print("\n[baseline] stratified repeated 10x5 CV...")
base = L.repeated_cv(X, y, reps=L.REPS, verbose=True, label='strat')
base_f1 = base['oof_macroF1']

# ---- proxy spatial: leave-one-cluster-out at several granularities ----
results = {}
for K in [5, 8, 10, 12]:
    km = KMeans(n_clusters=K, random_state=L.SEED, n_init=10).fit(S)
    g = km.labels_
    sizes = np.bincount(g).tolist()
    # per-cluster class composition (to expose imbalance honestly)
    comp = {int(c): np.bincount(y[g == c], minlength=3).tolist() for c in range(K)}
    f1 = grouped_oof_macroF1(g)
    results[str(K)] = {'n_clusters': K, 'cluster_sizes': sizes,
                       'cluster_class_composition_HOR': comp,
                       'leave_cluster_out_oof_macroF1': f1,
                       'drop_vs_stratified': (base_f1 - f1) if f1 is not None else None}
    print(f"  K={K:2d} clusters: leave-cluster-out OOF macro-F1 = "
          f"{(f1*100 if f1 else float('nan')):.2f}%  (drop {((base_f1-f1)*100 if f1 else float('nan')):+.2f} pp)")

valid = {k: v for k, v in results.items() if v['leave_cluster_out_oof_macroF1'] is not None}
mean_proxy = float(np.mean([v['leave_cluster_out_oof_macroF1'] for v in valid.values()]))
res = {'item': 'proxy_spatial_validation',
       'rationale': 'no GPS/plot id available; K-means spectral clusters as spatial-block proxy, '
                    'leave-one-cluster-out OOF macro-F1',
       'spectral_signature_dims': int(len(spec_cols)),
       'stratified_oof_macroF1': base_f1,
       'stratified_cv_mean': base['mean'], 'stratified_cv_ci95': base['ci95'],
       'leave_cluster_out': results,
       'mean_leave_cluster_out_macroF1': mean_proxy,
       'mean_drop_pp': float((base_f1 - mean_proxy) * 100)}
json.dump(res, open(OUT / "V7_proxy_spatial.json", 'w'), indent=2)

# ---- figure ----
Ks = [int(k) for k in valid]; vals = [valid[str(k)]['leave_cluster_out_oof_macroF1'] * 100 for k in Ks]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.axhline(base_f1 * 100, ls='--', color='#2c7fb8', lw=2, label=f'Stratified CV = {base_f1*100:.1f}%')
ax.bar([str(k) for k in Ks], vals, color='#d95f0e', alpha=0.8, label='Leave-cluster-out (proxy spatial)')
for k, v in zip(Ks, vals): ax.text(str(k), v + 0.3, f"{v:.1f}", ha='center', fontsize=8)
ax.set_xlabel("Number of spectral clusters (K)"); ax.set_ylabel("OOF Macro-F1 (%)")
ax.set_title("Proxy spatial validation: spectrally-distinct held-out blocks")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(min(vals) - 4, base_f1 * 100 + 3)
plt.tight_layout(); plt.savefig(OUT / "V7_proxy_spatial.png", dpi=150); plt.close()

print("\n" + "=" * 70)
print(f"Stratified CV macro-F1   = {base_f1*100:.2f}%")
print(f"Proxy spatial (mean)     = {mean_proxy*100:.2f}%   (drop {res['mean_drop_pp']:+.2f} pp)")
print("Saved V7_proxy_spatial.json + V7_proxy_spatial.png")
print("=" * 70)
