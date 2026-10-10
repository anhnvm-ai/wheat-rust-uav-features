"""
R2_canopy_extract.py  —  Item 2: Canopy-aware ROI feature extraction (SAT round 2).

Builds a per-tile canopy mask (Otsu threshold on MS-NDVI; vegetation kept, soil/background
dropped) and recomputes the EXACT V7 149-feature vector over canopy pixels only, so the
comparison against the full-tile features is one-to-one (same dimensions, only the support
changes). Also emits ~12 extra canopy-context features (coverage, NDVI/NDRE canopy stats,
NDVI-map texture, red-edge slope) for an "ROI+" configuration.

Mirrors feat_v7_lib.py extractor maths verbatim; the only change is that pixel statistics and
band means are taken over the mask. Mask computed at MS/RGB resolution (64x64) and resized
nearest to HS (32x32).

Outputs:
  outputs/reviewer_experiments/features_v7_canopy.csv        (masked 149-feat, drop-in)
  outputs/reviewer_experiments/features_v7_canopy_plus.csv   (masked 149 + 12 ROI-context)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2, tifffile
from pathlib import Path
from feat_v7_lib import feature_names

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
EPS = 1e-8
MIN_PIX = 16  # if canopy too small, fall back to whole tile


def load_sample(sid, base):
    rgb = cv2.imread(str(base / "RGB" / f"{sid}.png"))
    if rgb is not None: rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    try:
        ms = tifffile.imread(str(base / "MS" / f"{sid}.tif"))
        if ms.ndim == 3 and ms.shape[0] < ms.shape[2]: ms = np.moveaxis(ms, 0, -1)
    except Exception: ms = None
    try:
        hs = tifffile.imread(str(base / "HS" / f"{sid}.tif"))
        if hs.ndim == 3 and hs.shape[0] < hs.shape[2]: hs = np.moveaxis(hs, 0, -1)
    except Exception: hs = None
    return rgb, ms, hs


def canopy_mask(ms):
    """Otsu-on-NDVI canopy mask at MS resolution (HxW bool). Falls back to all-True if degenerate."""
    if ms is None or ms.shape[-1] < 5: return None
    msf = ms.astype(np.float32); red, nir = msf[:, :, 2], msf[:, :, 4]
    ndvi = (nir - red) / (nir + red + EPS)
    nd8 = np.clip((ndvi + 1) / 2 * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(nd8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = t / 255 * 2 - 1
    m = ndvi > thr
    if m.sum() < MIN_PIX or m.mean() > 0.98:  # degenerate -> use all
        m = np.ones_like(m, bool)
    return m


def _stats6(ch, m, qs):
    v = ch[m]
    return [float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)),
            float(np.percentile(v, qs[0])), float(np.percentile(v, qs[1]))]


def rgb_masked(rgb, m):
    if rgb is None: return np.zeros(30)
    rgb = rgb.astype(np.float32); f = []
    for c in range(3): f += _stats6(rgb[:, :, c], m, (25, 75))
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mr, mg, mb = np.mean(r[m]), np.mean(g[m]), np.mean(b[m])
    f.append((2 * mg - mr - mb) / 255.0); f.append((1.4 * mr - mg) / 255.0)
    f += [mr / (mg + EPS), mr / (mb + EPS), mg / (mb + EPS), (mr - mg) / (mr + mg + EPS), (mg - mb) / (mg + mb + EPS)]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2); f += [float(np.mean(gm[m])), float(np.std(gm[m])), float(np.max(gm[m]))]
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    f += [float(np.mean(hsv[:, :, 0][m])), float(np.std(hsv[:, :, 0][m])),
          float(np.mean(hsv[:, :, 1][m])), float(np.std(hsv[:, :, 1][m]))]
    return np.array(f[:30])


def ms_masked(ms, m):
    if ms is None: return np.zeros(50)
    ms = ms.astype(np.float32); nb = ms.shape[-1]; f = []
    for b in range(min(nb, 5)): f += _stats6(ms[:, :, b], m, (10, 90))
    if nb >= 5:
        blue, green, red, re, nir = [np.mean(ms[:, :, i][m]) for i in range(5)]
        ndvi = (nir - red) / (nir + red + EPS); gndvi = (nir - green) / (nir + green + EPS)
        ndre = (nir - re) / (nir + re + EPS); sr = nir / (red + EPS)
        mtci = (nir - re) / (re - red + EPS); ci_green = nir / (green + EPS) - 1; ci_re = nir / (re + EPS) - 1
        pri = (green - red) / (green + red + EPS); ndwi = (green - nir) / (green + nir + EPS)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS); savi = ((nir - red) / (nir + red + 0.5)) * 1.5
        f += [ndvi, gndvi, ndre, sr, mtci, ci_green, ci_re, pri, ndwi, evi, savi,
              nir / (red + EPS), nir / (green + EPS), re / (red + EPS)]
        for b in range(5):
            v = ms[:, :, b][m]; f.append(float(np.std(v) / (np.mean(v) + EPS)))
    else: f += [0] * 19
    return np.array(f[:50])


def hs_masked(hs, m_hs):
    if hs is None: return np.zeros(70)
    hs = hs.astype(np.float32); nb = hs.shape[-1]; f = []
    flat = hs[m_hs]                                   # (Npix, nb)
    mean_spectrum = flat.mean(0); std_spectrum = flat.std(0)
    if nb > 24: cs, ce = 10, nb - 14; mean_clean = mean_spectrum[cs:ce]; std_clean = std_spectrum[cs:ce]
    else: cs = 0; mean_clean = mean_spectrum; std_clean = std_spectrum
    nc = len(mean_clean)
    if nc >= 20: idx = np.linspace(0, nc - 1, 20).astype(int); f += list(mean_clean[idx] / 5000.0)
    else: f += list(mean_clean / 5000.0) + [0] * (20 - nc)
    f += [np.mean(mean_clean) / 5000.0, np.std(mean_clean) / 5000.0, np.min(mean_clean) / 5000.0,
          np.max(mean_clean) / 5000.0, (np.max(mean_clean) - np.min(mean_clean)) / 5000.0, np.median(mean_clean) / 5000.0]
    if nc > 1: f += [np.argmax(mean_clean) / nc, np.argmin(mean_clean) / nc]
    else: f += [0, 0]
    if nc > 2:
        d = np.gradient(mean_clean); f += [np.mean(d) / 5000.0, np.std(d) / 5000.0, np.max(d) / 5000.0, np.argmax(d) / nc]
    else: f += [0, 0, 0, 0]
    if nc >= 50:
        rs, re_ = int(nc * 0.35), int(nc * 0.50); rr = mean_clean[rs:re_]
        if len(rr) > 1: rd = np.gradient(rr); f += [(rs + np.argmax(rd)) / nc, np.max(rd) / 5000.0]
        else: f += [0, 0]
        ci, ni = int(nc * 0.30), int(nc * 0.65)
        f.append((mean_clean[ni] - mean_clean[ci]) / (mean_clean[ni] + mean_clean[ci] + EPS) if ci < nc and ni < nc else 0)
    else: f += [0, 0, 0]
    if nc > 10:
        cors = []
        for i in range(0, nc - 10, 10):
            c = np.corrcoef(mean_clean[i:i + 5], mean_clean[i + 5:i + 10])[0, 1]
            if not np.isnan(c): cors.append(c)
        f.append(np.mean(cors) if cors else 0)
    else: f.append(0)
    if nc >= 6:
        tot = np.trapz(mean_clean); t1 = np.trapz(mean_clean[:nc // 3]); t2 = np.trapz(mean_clean[nc // 3:2 * nc // 3]); t3 = np.trapz(mean_clean[2 * nc // 3:])
        f += [t1 / (tot + EPS), t2 / (tot + EPS), t3 / (tot + EPS)]
    else: f += [0, 0, 0]
    if nc >= 10:
        si = np.linspace(0, nc - 1, 10).astype(int)
        for ix in si[:5]:
            bidx = min(ix + cs, hs.shape[-1] - 1); v = hs[:, :, bidx][m_hs]; f.append(float(np.std(v) / (np.mean(v) + EPS)))
    else: f += [0] * 5
    while len(f) < 70: f.append(0)
    return np.array(f[:70])


def canopy_extras(ms, m):
    """~12 ROI-context features (only meaningful with a real mask)."""
    if ms is None: return np.zeros(12)
    msf = ms.astype(np.float32); blue, green, red, re, nir = [msf[:, :, i] for i in range(5)]
    ndvi = (nir - red) / (nir + red + EPS); ndre = (nir - re) / (nir + re + EPS)
    nv, nr = ndvi[m], ndre[m]
    grad = cv2.Sobel(ndvi.astype(np.float32), cv2.CV_32F, 1, 0, 3); grady = cv2.Sobel(ndvi.astype(np.float32), cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(grad ** 2 + grady ** 2)[m]
    nirv = nir[m]
    return np.array([
        float(m.mean()),                                            # canopy coverage fraction
        float(nv.mean()), float(nv.std()), float(np.percentile(nv, 25)), float(np.percentile(nv, 75)),
        float(nr.mean()), float(nr.std()),
        float(gmag.mean()), float(gmag.std()),                      # NDVI-map texture (within canopy)
        float(nirv.std() / (nirv.mean() + EPS)),                    # NIR canopy CV
        float((re[m].mean() - red[m].mean())),                      # red-edge step
        float(np.percentile(nv, 90) - np.percentile(nv, 10)),       # NDVI spread
    ])


def build(base, is_val=False):
    base = Path(base)
    ids = sorted(p.stem for p in (base / "HS").glob("*.tif"))
    rows149, rowsplus, labels, names_out = [], [], [], []
    for sid in ids:
        rgb, ms, hs = load_sample(sid, base)
        m = canopy_mask(ms)
        if m is None: m = np.ones((64, 64), bool)
        m_hs = cv2.resize(m.astype(np.uint8), (hs.shape[1], hs.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) if hs is not None else None
        if m_hs is not None and m_hs.sum() < 4: m_hs = np.ones((hs.shape[0], hs.shape[1]), bool)
        v149 = np.concatenate([rgb_masked(rgb, m), ms_masked(ms, m), hs_masked(hs, m_hs)])
        extra = canopy_extras(ms, m)
        rows149.append(v149); rowsplus.append(np.concatenate([v149, extra]))
        labels.append(sid.split('_')[0] if not is_val else '')
        names_out.append(sid)
    return (np.nan_to_num(np.array(rows149), nan=0, posinf=0, neginf=0),
            np.nan_to_num(np.array(rowsplus), nan=0, posinf=0, neginf=0),
            labels, names_out)


print("Extracting canopy-aware features for TRAIN (600 tiles)...")
X149, Xplus, y, ids = build(ROOT / "train", is_val=False)
names149 = feature_names()
extra_names = ['cz_frac', 'cz_ndvi_mean', 'cz_ndvi_std', 'cz_ndvi_p25', 'cz_ndvi_p75',
               'cz_ndre_mean', 'cz_ndre_std', 'cz_ndvi_grad_mean', 'cz_ndvi_grad_std',
               'cz_nir_cv', 'cz_re_step', 'cz_ndvi_spread']
df149 = pd.DataFrame(X149, columns=names149); df149['filename'] = ids; df149['label'] = y
df149.to_csv(OUT / "features_v7_canopy.csv", index=False)
dfp = pd.DataFrame(Xplus, columns=names149 + extra_names); dfp['filename'] = ids; dfp['label'] = y
dfp.to_csv(OUT / "features_v7_canopy_plus.csv", index=False)
print(f"Saved features_v7_canopy.csv {df149.shape}  and  features_v7_canopy_plus.csv {dfp.shape}")
print(f"Mean canopy coverage = {Xplus[:, 149].mean():.3f}  (col cz_frac)")
