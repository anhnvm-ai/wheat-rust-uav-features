"""
R2_srsm.py  —  Item 5 (optional): Self-supervised Red-edge Spectral Mining on HS.

Question: can a richer, canopy-restricted mining of the 125-band HS cube beat the current
70-dim HS summary, and does it add anything on top of MS? Honest test of HS value.

SRSM-HS feature set (leakage-free):
  * dense canopy-mean reflectance + 1st/2nd derivatives along the cleaned spectrum
  * red-edge descriptors (REIP, max slope, area, continuum-removed depth) at relative positions
  * NIR-plateau AUC/slope, global spectral stats, per-group spectral CV
  * Spectral Angle Mapper distance + correlation to K=6 UNSUPERVISED spectral prototypes
    (KMeans on canopy-mean spectra — no labels used, so it is self-supervised)
No HS wavelength file is shipped, so red-edge/NIR regions use relative band positions (same
convention as feat_v7_lib). Prototypes are label-free, fit once on the 600 training spectra.

Compares 4 configs under locked protocol: HS-summary / SRSM-HS / MS-only / MS + SRSM-HS.
Output: outputs/.../features_v7_srsm.csv, V7_srsm.json, V7_srsm.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2, tifffile, json
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import r2_lib as L

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
EPS = 1e-8; N_PROTO = 6


def load(sid, base):
    try:
        ms = tifffile.imread(str(base / "MS" / f"{sid}.tif")).astype(np.float32)
        if ms.ndim == 3 and ms.shape[0] < ms.shape[2]: ms = np.moveaxis(ms, 0, -1)
    except Exception: ms = None
    try:
        hs = tifffile.imread(str(base / "HS" / f"{sid}.tif")).astype(np.float32)
        if hs.ndim == 3 and hs.shape[0] < hs.shape[2]: hs = np.moveaxis(hs, 0, -1)
    except Exception: hs = None
    return ms, hs


def canopy_mask_hs(ms, hs):
    if ms is None or hs is None: return np.ones((hs.shape[0], hs.shape[1]), bool) if hs is not None else None
    red, nir = ms[:, :, 2], ms[:, :, 4]; ndvi = (nir - red) / (nir + red + EPS)
    nd8 = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(nd8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ndvi > (t / 255 * 2 - 1)
    if m.sum() < 16: m = np.ones_like(m, bool)
    return cv2.resize(m.astype(np.uint8), (hs.shape[1], hs.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def mean_spectrum(hs, m):
    flat = hs[m]; ms_ = flat.mean(0)
    cs, ce = 10, hs.shape[-1] - 14
    mc = ms_[cs:ce] / 5000.0
    return mc


def srsm_features(hs, m, protos):
    if hs is None: return np.zeros(40 + 20 + 10 + 4 + 3 + 6 + 5 + 2 * N_PROTO)
    mc = mean_spectrum(hs, m); nc = len(mc)
    d1 = np.gradient(mc); d2 = np.gradient(d1)
    f = []
    f += list(np.interp(np.linspace(0, nc - 1, 40), np.arange(nc), mc))           # 40 dense reflectance
    f += list(np.interp(np.linspace(0, nc - 1, 20), np.arange(nc), d1))            # 20 first deriv
    f += list(np.interp(np.linspace(0, nc - 1, 10), np.arange(nc), d2))            # 10 second deriv
    rs, re_ = int(nc * 0.35), int(nc * 0.55); rr = mc[rs:re_]; rd = np.gradient(rr) if len(rr) > 1 else np.zeros(1)
    reip = (rs + np.argmax(rd)) / nc; reslope = float(np.max(rd)); rearea = float(np.trapz(rr))
    cont = np.linspace(rr[0], rr[-1], len(rr)); depth = float(np.max(cont - rr)) if len(rr) else 0.0
    f += [reip, reslope, rearea, depth]                                            # 4 red-edge
    ns, ne = int(nc * 0.6), int(nc * 0.9); nr = mc[ns:ne]
    f += [float(np.trapz(nr)), float(np.mean(nr)), float(np.gradient(nr).mean() if len(nr) > 1 else 0)]  # 3 NIR
    f += [float(mc.mean()), float(mc.std()), float(mc.min()), float(mc.max()), float(np.median(mc)), float(mc.max() - mc.min())]  # 6
    g = np.array_split(mc, 5)
    f += [float(x.std() / (x.mean() + EPS)) for x in g]                            # 5 group CV
    for c in protos:                                                               # SAM + corr to prototypes
        sam = float(np.arccos(np.clip(np.dot(mc, c) / (np.linalg.norm(mc) * np.linalg.norm(c) + EPS), -1, 1)))
        corr = float(np.corrcoef(mc, c)[0, 1]) if np.std(mc) > 0 else 0.0
        f += [sam, corr]
    return np.nan_to_num(np.array(f), nan=0, posinf=0, neginf=0)


# pass 1: collect canopy-mean spectra, fit unsupervised prototypes
ids = sorted(p.stem for p in (ROOT / "train" / "HS").glob("*.tif"))
print(f"Pass 1: {len(ids)} tiles -> canopy-mean spectra, fit {N_PROTO} prototypes")
spectra = []; cache = {}
for sid in ids:
    ms, hs = load(sid, ROOT / "train"); m = canopy_mask_hs(ms, hs)
    cache[sid] = (hs, m); spectra.append(mean_spectrum(hs, m))
spectra = np.array(spectra)
protos = KMeans(n_clusters=N_PROTO, random_state=L.SEED, n_init=10).fit(spectra).cluster_centers_

# pass 2: SRSM features
print("Pass 2: SRSM feature extraction")
Xsrsm = []; y = []
for sid in ids:
    hs, m = cache[sid]; Xsrsm.append(srsm_features(hs, m, protos)); y.append(sid.split('_')[0])
Xsrsm = np.nan_to_num(np.array(Xsrsm));
srsm_names = [f'srsm_{i}' for i in range(Xsrsm.shape[1])]
dfs = pd.DataFrame(Xsrsm, columns=srsm_names); dfs['filename'] = ids; dfs['label'] = y
dfs.to_csv(OUT / "features_v7_srsm.csv", index=False)
print(f"SRSM-HS dims = {Xsrsm.shape[1]}")

# build configs from existing v7 columns + SRSM
v7 = pd.read_csv(OUT / "features_v7.csv")
yv = v7['label'].map(L.IDX).values
ms_cols = [c for c in v7.columns if c.startswith('ms_')]
hs_cols = [c for c in v7.columns if c.startswith('hs_')]
Xms = np.nan_to_num(v7[ms_cols].values.astype(float)); Xhs = np.nan_to_num(v7[hs_cols].values.astype(float))
assert list(v7['filename']) == ids, "row order mismatch"

configs = {'HS_summary_70': Xhs, 'SRSM_HS': Xsrsm, 'MS_only': Xms,
           'MS_plus_SRSM': np.hstack([Xms, Xsrsm])}
res = {}
for nm, Xc in configs.items():
    print(f"\n[{nm}] X={Xc.shape}")
    cv = L.repeated_cv(Xc, yv, reps=L.REPS, verbose=True, label=nm)
    res[nm] = {'n_features': int(Xc.shape[1]), 'macroF1_mean': cv['mean'],
               'macroF1_std': cv['std'], 'ci95': cv['ci95'], 'oof_macroF1': cv['oof_macroF1']}

out = {'item': 'srsm', 'n_prototypes': N_PROTO, 'srsm_dims': int(Xsrsm.shape[1]),
       'note': 'relative band positions (no wavelength file); prototypes label-free', 'configs': res,
       'srsm_vs_hs_summary_pp': float((res['SRSM_HS']['macroF1_mean'] - res['HS_summary_70']['macroF1_mean']) * 100),
       'ms_plus_srsm_vs_ms_pp': float((res['MS_plus_SRSM']['macroF1_mean'] - res['MS_only']['macroF1_mean']) * 100)}
json.dump(out, open(OUT / "V7_srsm.json", 'w'), indent=2)

names = list(res.keys()); vals = [res[n]['macroF1_mean'] * 100 for n in names]; errs = [res[n]['macroF1_std'] * 100 for n in names]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names, vals, yerr=errs, capsize=4, color=['#7f7f7f', '#2c7fb8', '#31a354', '#d95f0e'], alpha=0.85)
for i, (v, e) in enumerate(zip(vals, errs)): ax.text(i, v + e + 0.3, f"{v:.1f}", ha='center', fontsize=8)
ax.set_ylabel("Macro-F1 (%)"); ax.set_title("SRSM red-edge mining vs HS summary / MS")
ax.grid(alpha=0.3, axis='y'); plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(OUT / "V7_srsm.png", dpi=150); plt.close()

print("\n" + "=" * 70)
for n in names: print(f"{n:16s} {res[n]['n_features']:3d}f  macroF1={res[n]['macroF1_mean']*100:.2f}% +/- {res[n]['macroF1_std']*100:.2f}")
print(f"SRSM-HS vs HS-summary: {out['srsm_vs_hs_summary_pp']:+.2f} pp")
print(f"MS+SRSM vs MS-only:    {out['ms_plus_srsm_vs_ms_pp']:+.2f} pp")
print("Saved V7_srsm.json + V7_srsm.png")
print("=" * 70)
