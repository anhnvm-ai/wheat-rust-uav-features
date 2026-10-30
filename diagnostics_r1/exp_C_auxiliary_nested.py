"""Experiment C (pre-registered 29 Oct 2026; protocol in PREREG_aux_hyperparam.md): the auxiliary analyses of Table 8
(panels (a) and (b)) and Table 9 re-run under the primary nested protocol, with the HS block as submitted (primary) and
with corrected HS features (secondary); and Figure 5 (calibration) recomputed from the saved nested out-of-fold probabilities.

Protocol (identical to the primary analysis and to exp_A_corrected_hs_nested.py): 50 outer folds,
StratifiedKFold(5, shuffle=True, random_state=42+r), r = 0..9, tiles in the order of the feature table (= folds_10x5.csv);
StandardScaler fitted on the training part; member weight = Macro-F1 of the member in an inner stratified 5-fold CV
(seed 42) on the outer training part; five members with the Table 4 settings refitted on the outer training part; the outer
fold is scored once. Column subsets keep the column order of the feature table, so the full model of every panel is the
primary analysis (S0, HS as submitted) or experiment A's S5 (corrected HS).

Configurations
  Table 8(a): A_full_sub, A_noHS, A_noRGB_sub, A_noMS_sub, A_MS, A_RGB, A_HS_sub | A_full_cor, A_noRGB_cor, A_noMS_cor, A_HS_cor
  Table 8(b): B1_raw_sub, B2_veg_sub, B3_shape_sub | B1_raw_cor, B2_veg_cor, B3_shape_cor  (step 4 = A_full_*)
              (groups as in Av_feature_ablation_v7.py: RAW 102, + MS indices and ratios 116, + further HS-block entries 132)
  Table 9   : K_mask_sub (features_v7_canopy.csv), K_ctx_sub (features_v7_canopy_plus.csv)
              K_mask_cor, K_ctx_cor (same files, HS columns replaced by the corrected HS features of the full tile)
              (full-tile rows = A_full_sub and A_full_cor)
Usage:  python3 exp_C_auxiliary_nested.py <out_dir> gate            (reproduces the published values; run before the lock)
        python3 exp_C_auxiliary_nested.py <out_dir> run [IDs|all]    (NJOBS env, default 6)
        python3 exp_C_auxiliary_nested.py <out_dir> calib            (Figure 5 from exp_A_probs.jsonl; fits no model)
        python3 exp_C_auxiliary_nested.py <out_dir> summarise
Finished folds are appended to exp_C_folds.jsonl, so an interrupted run resumes."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, time, platform, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/drive/MyDrive/Kaggle_Prepared/code")
import r2_lib as L
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from scipy import stats

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1]); MODE = sys.argv[2]
RX = ROOT / "outputs/reviewer_experiments"
CKPT = OUT / "exp_C_folds.jsonl"; PROBS = OUT / "exp_C_probs.jsonl"
X, y, fn, cols = L.load_features(RX / "features_v7.csv")
cols = np.array(cols).astype(str)
hs_i = np.where(np.char.startswith(cols, "hs_"))[0]
ms_i = np.where(np.char.startswith(cols, "ms_"))[0]
rgb_i = np.where(np.char.startswith(cols, "rgb_"))[0]

# feature groups, copied from Av_feature_ablation_v7.py (column names); used in the order of the feature table
VEG = ['ms_NDVI', 'ms_GNDVI', 'ms_NDRE', 'ms_SR', 'ms_MTCI', 'ms_CIgreen', 'ms_CIre', 'ms_PRI', 'ms_NDWI', 'ms_EVI', 'ms_SAVI',
       'ms_NIR/Red', 'ms_NIR/Green', 'ms_RE/Red']
HS_SHAPE = ['hs_d1_mean', 'hs_d1_std', 'hs_d1_max', 'hs_d1_maxpos', 'hs_REIP', 'hs_RE_slope', 'hs_NDVI', 'hs_band_corr', 'hs_AUC1',
            'hs_AUC2', 'hs_AUC3'] + [c for c in cols if c.startswith('hs_CV')]
RGB_COLOR = ['rgb_ExG', 'rgb_ExR', 'rgb_R/G', 'rgb_R/B', 'rgb_G/B', 'rgb_nd_RG', 'rgb_nd_GB', 'rgb_grad_mean', 'rgb_grad_std',
             'rgb_grad_max', 'rgb_hue_mean', 'rgb_hue_std']
MS_CV = [c for c in cols if c.startswith('ms_CV')]
RAW = [c for c in cols if c not in set(VEG + HS_SHAPE + RGB_COLOR + MS_CV)]
def idx(names): s = set(names); return np.array([i for i, c in enumerate(cols) if c in s])
G_RAW, G_VEG, G_SHAPE = idx(RAW), idx(RAW + VEG), idx(RAW + VEG + HS_SHAPE)
assert (len(G_RAW), len(G_VEG), len(G_SHAPE)) == (102, 116, 132)


def corrected_hs():
    HSF = np.load(OUT / "exp_A_corrected_hs_features.npy")          # built by exp_A_corrected_hs_nested.py
    assert HSF.shape == (len(y), len(hs_i)); return HSF


def canopy(name, hs_fix=None):
    Xk, yk, fk, ck = L.load_features(RX / name)
    assert (fk == fn).all() and (yk == y).all() and list(ck[:len(cols)]) == list(cols)
    if hs_fix is not None:
        Xk = Xk.copy(); Xk[:, hs_i] = hs_fix
    return Xk


def config(cid):
    Xs = X
    if cid.endswith('_cor'):
        Xs = X.copy(); Xs[:, hs_i] = corrected_hs()
    base = cid.rsplit('_', 1)[0] if cid.endswith(('_sub', '_cor')) else cid
    return {'A_full': lambda: Xs, 'A_noHS': lambda: Xs[:, np.r_[rgb_i, ms_i]], 'A_noRGB': lambda: Xs[:, np.r_[ms_i, hs_i]],
            'A_noMS': lambda: Xs[:, np.r_[rgb_i, hs_i]], 'A_MS': lambda: Xs[:, ms_i], 'A_RGB': lambda: Xs[:, rgb_i],
            'A_HS': lambda: Xs[:, hs_i], 'B1_raw': lambda: Xs[:, G_RAW], 'B2_veg': lambda: Xs[:, G_VEG],
            'B3_shape': lambda: Xs[:, G_SHAPE],
            'K_mask': lambda: canopy("features_v7_canopy.csv", Xs[:, hs_i] if cid.endswith('_cor') else None),
            'K_ctx': lambda: canopy("features_v7_canopy_plus.csv", Xs[:, hs_i] if cid.endswith('_cor') else None)}[base]()


IDS = ['A_full_sub', 'A_noHS', 'A_noRGB_sub', 'A_noMS_sub', 'A_MS', 'A_RGB', 'A_HS_sub',
       'A_full_cor', 'A_noRGB_cor', 'A_noMS_cor', 'A_HS_cor',
       'B1_raw_sub', 'B2_veg_sub', 'B3_shape_sub', 'B1_raw_cor', 'B2_veg_cor', 'B3_shape_cor',
       'K_mask_sub', 'K_ctx_sub', 'K_mask_cor', 'K_ctx_cor']


def mf1(yt, p): return f1_score(yt, p, average='macro')


def fit_predict(Xa, ya, Xb):
    sc = StandardScaler().fit(Xa); a, b = sc.transform(Xa), sc.transform(Xb)
    out = {}
    for nm, md in L.make_models().items():
        md.fit(a, ya); out[nm] = md.predict_proba(b)
    return out


def classwise(yt, pred):
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2]).astype(float)
    pr = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    rc = np.diag(cm) / cm.sum(1)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(3), where=(pr + rc) > 0)
    return {'precision': pr.tolist(), 'recall': rc.tolist(), 'f1': f.tolist(), 'cm': cm.astype(int).tolist()}


def run_fold(cid, Xc, r, k, tr, va):
    P = fit_predict(Xc[tr], y[tr], Xc[va]); yv = y[va]
    equal = sum(P.values()) / len(P)
    wl = {n: mf1(yv, P[n].argmax(1)) for n in P}                        # weights on the scored fold (original rule)
    leaky = sum(wl[n] * P[n] for n in P) / sum(wl.values())
    Pin = {n: np.zeros((len(tr), 3)) for n in P}
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(Xc[tr], y[tr]):
        pi = fit_predict(Xc[tr][itr], y[tr][itr], Xc[tr][iva])
        for n in pi: Pin[n][iva] = pi[n]
    wn = {n: mf1(y[tr], Pin[n].argmax(1)) for n in Pin}                  # inner-OOF Macro-F1 (training labels only)
    ens = sum(wn[n] * P[n] for n in P) / sum(wn.values()); pk = ens.argmax(1)
    rec = {'config': cid, 'r': r, 'k': k, 'n_feat': int(Xc.shape[1]), 'weights_nested': wn,
           'f1_nested': mf1(yv, pk), 'acc_nested': accuracy_score(yv, pk), 'classwise_nested': classwise(yv, pk),
           'f1_equal': mf1(yv, equal.argmax(1)), 'f1_leaky': mf1(yv, leaky.argmax(1))}
    return rec, {'config': cid, 'r': r, 'k': k, 'va': [int(i) for i in va], 'p_nested': ens.round(8).tolist()}


def folds_ok():
    ref = pd.read_csv(ROOT / "code/folds_10x5.csv"); bad = 0
    for r in range(L.REPS):
        for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(X, y)):
            bad += set(ref[(ref.repetition == r) & (ref.fold == k)].sample_id) != set(fn[va])
    return bad == 0


def load_done():
    done = {}
    if CKPT.exists():
        for line in open(CKPT):
            d = json.loads(line); done[(d['config'], d['r'], d['k'])] = d
    return done


def run(ids):
    assert folds_ok(), "folds differ from folds_10x5.csv"
    done = load_done(); t0 = time.time(); tasks = []
    for cid in ids:
        Xc = config(cid)
        for r in range(L.REPS):
            for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(Xc, y)):
                if (cid, r, k) not in done: tasks.append((cid, Xc, r, k, tr, va))
    print(f"{len(done)} folds from checkpoint, {len(tasks)} to run", flush=True)
    with open(CKPT, "a") as fh, open(PROBS, "a") as fp:
        for rec, pr in Parallel(n_jobs=int(os.environ.get("NJOBS", "6")), return_as="generator_unordered")(
                delayed(run_fold)(*t) for t in tasks):
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush(); fp.write(json.dumps(pr) + "\n"); fp.flush()
            print(f"{rec['config']} r={rec['r']} k={rec['k']} nested {rec['f1_nested']*100:.1f} ({time.time()-t0:.0f}s)", flush=True)


# ---------------- gate R: reproduce the published (original-protocol) values before the lock ----------------
def leaky_fold(Xc, tr, va):
    P = fit_predict(Xc[tr], y[tr], Xc[va]); wl = {n: mf1(y[va], P[n].argmax(1)) for n in P}
    return mf1(y[va], (sum(wl[n] * P[n] for n in P) / sum(wl.values())).argmax(1))


def leaky_cv(Xc, reps):
    t = [(tr, va) for r in range(reps) for tr, va in StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(Xc, y)]
    return float(np.mean(Parallel(n_jobs=int(os.environ.get("NJOBS", "6")))(delayed(leaky_fold)(Xc, tr, va) for tr, va in t)))


def gate():
    A = json.load(open(RX / "V7_assets.json"))['modality_ablation']; B = json.load(open(RX / "V7_feature_ablation.json"))
    K = json.load(open(RX / "V7_canopy.json"))['configs']
    df = pd.read_csv(RX / "features_v7.csv"); allc = [c for c in df.columns if c not in ('filename', 'label')]
    def orig(names): return np.nan_to_num(df[names].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    R = [c for c in allc if c in set(RAW)]                       # RAW in the column order of the table
    checks = {   # (matrix in the column order of the original script, reps, published value)
        'Full(HS+MS+RGB)': (X, 3, A['Full(HS+MS+RGB)']['f1']), 'without HS': (X[:, np.r_[rgb_i, ms_i]], 3, A['without HS']['f1']),
        'without MS': (X[:, np.r_[rgb_i, hs_i]], 3, A['without MS']['f1']), 'without RGB': (X[:, np.r_[ms_i, hs_i]], 3, A['without RGB']['f1']),
        'HS only': (X[:, hs_i], 3, A['HS only']['f1']), 'MS only': (X[:, ms_i], 3, A['MS only']['f1']),
        'RGB only': (X[:, rgb_i], 3, A['RGB only']['f1']),
        'groups RAW': (orig(R), 3, B[0]['macroF1']), 'groups +VEG': (orig(R + VEG), 3, B[1]['macroF1']),
        'groups +HS shape': (orig(R + VEG + HS_SHAPE), 3, B[2]['macroF1']),
        'groups full': (orig(R + VEG + HS_SHAPE + RGB_COLOR + MS_CV), 3, B[3]['macroF1']),
        'canopy full tile': (X, 10, K['A_full_tile']['macroF1_mean']),
        'canopy masked': (canopy("features_v7_canopy.csv"), 10, K['B_canopy_masked']['macroF1_mean']),
        'canopy + context': (canopy("features_v7_canopy_plus.csv"), 10, K['C_canopy_plus']['macroF1_mean'])}
    out = {}
    for name, (Xc, reps, pub) in checks.items():
        v = leaky_cv(Xc, reps); out[name] = {'reproduced': v, 'published': pub, 'abs_diff_points': abs(v - pub) * 100}
        print(f"{name:18s} reproduced {v*100:.3f}  published {pub*100:.3f}  |diff| {abs(v-pub)*100:.3f}", flush=True)
    m = max(d['abs_diff_points'] for d in out.values())
    res = {'gate_R_reproduction': out, 'max_abs_diff_points': m, 'tolerance_points': 0.5, 'pass': bool(m <= 0.5),
           'folds_equal_folds_10x5': folds_ok(), 'group_sizes': [len(G_RAW), len(G_VEG), len(G_SHAPE), len(cols)]}
    json.dump(res, open(OUT / "exp_C_gate_R.json", "w"), indent=1); print(json.dumps({k: res[k] for k in res if k != 'gate_R_reproduction'}))


# ---------------- Figure 5: calibration of the nested out-of-fold probabilities (no model is fitted) ----------------
def ece(probs, labels, bins=10):                               # as in Av_consolidated_eval.py
    conf = probs.max(1); acc = (probs.argmax(1) == labels).astype(float); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1.0 / bins)
        if m.sum() > 0: e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(probs, labels): return float(np.mean(np.sum((probs - np.eye(3)[labels]) ** 2, 1)))


def isotonic_split_swap(P):                                     # as in Av_consolidated_eval.py
    cal = np.zeros_like(P)
    for fi, ai in StratifiedKFold(2, shuffle=True, random_state=L.SEED).split(P, y):
        cc = np.zeros((len(ai), 3))
        for k in range(3):
            cc[:, k] = IsotonicRegression(out_of_bounds='clip').fit(P[fi, k], (y[fi] == k).astype(float)).predict(P[ai, k])
        cc = np.clip(cc, 1e-6, None); cal[ai] = cc / cc.sum(1, keepdims=True)
    return cal


def calib():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = {}
    for cfg in ('S0', 'S5'):
        agg = np.zeros((len(y), 3)); n = 0
        for line in open(OUT / "exp_A_probs.jsonl"):
            d = json.loads(line)
            if d['config'] == cfg: agg[np.array(d['va'])] += np.array(d['p_nested']); n += 1
        assert n == 50; P = agg / L.REPS; cal = isotonic_split_swap(P); conf = P.max(1); ok = P.argmax(1) == y
        res[cfg] = {'raw_ECE': ece(P, y), 'raw_Brier': brier(P, y), 'isotonic_ECE': ece(cal, y), 'isotonic_Brier': brier(cal, y),
                    'conf_correct': float(conf[ok].mean()), 'conf_wrong': float(conf[~ok].mean()),
                    'agg_oof_macroF1': float(mf1(y, P.argmax(1)))}
        if cfg == 'S0':
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for axi, (Q, nm, e) in zip(ax, [(P, 'Raw ensemble (nested weights)', res[cfg]['raw_ECE']),
                                           (cal, 'Isotonic-calibrated', res[cfg]['isotonic_ECE'])]):
                cq = Q.max(1); aq = (Q.argmax(1) == y).astype(float); xs, ys = [], []
                for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
                    m = (cq > lo) & (cq <= hi)
                    if m.sum() > 0: xs.append((lo + hi) / 2); ys.append(aq[m].mean())
                axi.plot([0, 1], [0, 1], 'k--', lw=1); axi.plot(xs, ys, 'o-', color='#2c7fb8')
                axi.set_title(f"{nm}\nECE={e:.3f}"); axi.set_xlabel("Confidence"); axi.set_ylabel("Accuracy")
                axi.set_xlim(0, 1); axi.set_ylim(0, 1)
            plt.tight_layout(); plt.savefig(OUT / "V7_reliability_diagram_r3.png", dpi=150); plt.close()
    json.dump(res, open(OUT / "exp_C_calibration.json", "w"), indent=1); print(json.dumps(res, indent=1))


# ---------------- summary ----------------
def corrected_ci(v, J=50, ratio=0.25):
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se; t = m / se if se > 0 else np.nan
    return {'mean': float(m), 'sd_ddof0': float(v.std()), 'ci95': [float(m - h), float(m + h)], 't': float(t),
            'p_two_sided': float(2 * stats.t.sf(abs(t), J - 1)) if se > 0 else 1.0}


def summarise():
    done = load_done(); ids = [c for c in IDS if any(k[0] == c for k in done)]
    per = {c: sorted([d for (cc, _, _), d in done.items() if cc == c], key=lambda d: (d['r'], d['k'])) for c in ids}
    for c in ids: assert len(per[c]) == 50, (c, len(per[c]))
    f1 = {c: np.array([d['f1_nested'] for d in per[c]]) for c in ids}
    probs = {}
    for line in open(PROBS):
        d = json.loads(line); probs.setdefault(d['config'], []).append(d)
    out = {'configs': {}, 'paired': {}, 'gates': {}}
    for c in ids:
        agg = np.zeros((len(y), 3))
        for d in probs[c]: agg[np.array(d['va'])] += np.array(d['p_nested'])
        pa = (agg / L.REPS).argmax(1)
        out['configs'][c] = {'n_feat': per[c][0]['n_feat'], 'nested': corrected_ci(f1[c]),
                             'leaky_fold_mean': float(np.mean([d['f1_leaky'] for d in per[c]])),
                             'equal_fold_mean': float(np.mean([d['f1_equal'] for d in per[c]])),
                             'agg_oof_macroF1': float(mf1(y, pa)), 'agg_oof_classwise': classwise(y, pa)}
    pairs = []
    for v in ('sub', 'cor'):
        full = f'A_full_{v}'
        for c in ('A_noHS', f'A_noRGB_{v}', f'A_noMS_{v}', 'A_MS', 'A_RGB', f'A_HS_{v}'): pairs.append((c, full))
        pairs += [(f'B2_veg_{v}', f'B1_raw_{v}'), (f'B3_shape_{v}', f'B2_veg_{v}'), (full, f'B3_shape_{v}')]
        pairs += [(f'K_mask_{v}', full), (f'K_ctx_{v}', f'K_mask_{v}'), (f'K_ctx_{v}', full)]
    for a, b in pairs:
        if a in f1 and b in f1:
            out['paired'][f'{a} - {b}'] = corrected_ci(f1[a] - f1[b]) | {'folds_a_higher': int((f1[a] > f1[b]).sum())}
    # gates C1 (overlap with experiment A, exact) and C2 (validation-fold weights of the full model, exact)
    A = {}
    for line in open(OUT / "exp_A_folds.jsonl"):
        d = json.loads(line); A[(d['config'], d['r'], d['k'])] = d
    ov = {'A_full_sub': 'S0', 'A_noHS': 'S3', 'A_MS': 'S2', 'A_HS_sub': 'S1b', 'A_full_cor': 'S5', 'A_noRGB_cor': 'S4', 'A_HS_cor': 'S1'}
    g1 = {}
    for c, s0 in ov.items():
        if c in f1:
            ref = np.array([A[(s0, d['r'], d['k'])]['f1_nested'] for d in per[c]])
            refe = np.array([A[(s0, d['r'], d['k'])]['f1_equal'] for d in per[c]]); eq = np.array([d['f1_equal'] for d in per[c]])
            m = float(max(np.abs(f1[c] - ref).max(), np.abs(eq - refe).max())); g1[c] = {'exp_A': s0, 'max_abs_diff': m, 'pass': m < 1e-9}
    out['gates']['C1_overlap_equals_exp_A'] = g1
    if 'A_full_sub' in f1:
        ref = {(d['r'], d['k']): d['f1_leaky'] for d in json.load(open(OUT / "diag_nested_perfold.json"))['per_fold']}
        m = float(max(abs(d['f1_leaky'] - ref[(d['r'], d['k'])]) for d in per['A_full_sub']))
        out['gates']['C2_scored_fold_weights_equal_stored'] = {'max_abs_diff': m, 'pass': m < 1e-9}
    gr = OUT / "exp_C_gate_R.json"
    if gr.exists(): out['gates']['R_reproduction_before_lock'] = {k: v for k, v in json.load(open(gr)).items() if k != 'gate_R_reproduction'}
    import sklearn, scipy, lightgbm, xgboost
    out['versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                       'scipy': scipy.__version__, 'lightgbm': lightgbm.__version__, 'xgboost': xgboost.__version__}
    json.dump(out, open(OUT / "exp_C_summary.json", "w"), indent=1)
    for c in ids:
        q = out['configs'][c]['nested']
        print(f"{c:13s} {q['mean']*100:5.1f} ± {q['sd_ddof0']*100:.1f} [{q['ci95'][0]*100:.1f}, {q['ci95'][1]*100:.1f}]")
    for k_, q in out['paired'].items():
        print(f"{k_:26s} {q['mean']*100:+.2f} [{q['ci95'][0]*100:+.2f}, {q['ci95'][1]*100:+.2f}] p={q['p_two_sided']:.3f}")
    print(json.dumps(out['gates'], indent=1))


if MODE == 'gate':
    gate()
elif MODE == 'run':
    run(IDS if sys.argv[3:] in ([], ['all']) else sys.argv[3:])
elif MODE == 'calib':
    calib()
elif MODE == 'summarise':
    summarise()
else:
    sys.exit("mode: gate | run [IDs|all] | calib | summarise")
