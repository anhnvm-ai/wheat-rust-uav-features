"""Experiment A (final robustness check, pre-registered 19 Oct 2026; protocol in PREREG_final_robustness.md):
corrected HS features under the primary nested 10x5 protocol.

Same folds, models, seeds and nested member weighting as the primary analysis (diag_nested_perfold.py);
only the feature representation changes. Corrected HS features are built as in diag_leakage_hs.py
(cube read as rows x columns x bands; first 125 bands of the 126-band files; same extractor), and the
feature subsets use the same column order as diag_leakage_hs.py.

Configurations: S0 RGB+MS+HS as submitted (149) | S1 corrected HS only (70) | S1b HS as submitted only (70)
| S2 MS only (49) | S3 RGB+MS (79) | S4 MS+corrected HS (119) | S5 RGB+MS+corrected HS (149).

Gates (stop if any fails; see PREREG): G1 S0 per-fold nested Macro-F1 equals diag_nested_perfold.json;
G2 S5 nested fold mean equals diag_leakage_hs.json (V7_fixedHS_full); G3 equal-weight fold means equal
diag_leakage_hs.json for every configuration stored there.

Usage:  python3 exp_A_corrected_hs_nested.py <out_dir> run S0        (one or more configuration IDs)
        python3 exp_A_corrected_hs_nested.py <out_dir> summarise
Finished folds are appended to exp_A_folds.jsonl, so an interrupted run resumes."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, time, platform, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/drive/MyDrive/Kaggle_Prepared/code")
import r2_lib as L
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy import stats

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1]); MODE = sys.argv[2]
CKPT = OUT / "exp_A_folds.jsonl"; PROBS = OUT / "exp_A_probs.jsonl"
X, y, fn, cols = L.load_features(ROOT / "outputs/reviewer_experiments/features_v7.csv")
cols = np.array(cols).astype(str)
hs_i = np.where(np.char.startswith(cols, "hs_"))[0]
ms_i = np.where(np.char.startswith(cols, "ms_"))[0]
rgb_i = np.where(np.char.startswith(cols, "rgb_"))[0]


def corrected_X():
    import tifffile, feat_v7_lib as F
    cache = OUT / "exp_A_corrected_hs_features.npy"
    if cache.exists():
        HSF = np.load(cache)
    else:
        HSF = np.array([F.extract_hs_raw_features(
            tifffile.imread(ROOT / "train" / "HS" / f"{s}.tif").astype(np.float32)[..., :125]) for s in fn])
        HSF = np.nan_to_num(HSF, nan=0, posinf=0, neginf=0); np.save(cache, HSF)
    Xf = X.copy(); Xf[:, hs_i] = HSF
    return Xf


def config(cid, Xfix):
    return {'S0': lambda: X, 'S1': lambda: Xfix[:, hs_i], 'S1b': lambda: X[:, hs_i], 'S2': lambda: X[:, ms_i],
            'S3': lambda: X[:, np.r_[rgb_i, ms_i]], 'S4': lambda: Xfix[:, np.r_[ms_i, hs_i]], 'S5': lambda: Xfix}[cid]()


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
    Pin = {n: np.zeros((len(tr), 3)) for n in P}
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(Xc[tr], y[tr]):
        pi = fit_predict(Xc[tr][itr], y[tr][itr], Xc[tr][iva])
        for n in pi: Pin[n][iva] = pi[n]
    wn = {n: mf1(y[tr], Pin[n].argmax(1)) for n in Pin}                  # inner-OOF Macro-F1 (training labels only)
    ens = sum(wn[n] * P[n] for n in P) / sum(wn.values()); pk = ens.argmax(1)
    rec = {'config': cid, 'r': r, 'k': k, 'n_feat': int(Xc.shape[1]), 'weights_nested': wn,
           'f1_nested': mf1(yv, pk), 'acc_nested': accuracy_score(yv, pk), 'classwise_nested': classwise(yv, pk),
           'f1_equal': mf1(yv, equal.argmax(1)), 'acc_equal': accuracy_score(yv, equal.argmax(1))}
    return rec, {'config': cid, 'r': r, 'k': k, 'va': [int(i) for i in va], 'p_nested': ens.round(8).tolist()}


def load_done():
    done = {}
    if CKPT.exists():
        for line in open(CKPT):
            d = json.loads(line); done[(d['config'], d['r'], d['k'])] = d
    return done


def run(ids):
    Xfix = corrected_X() if any(c in ('S1', 'S4', 'S5') for c in ids) else None
    done = load_done(); t0 = time.time()
    tasks = []
    for cid in ids:
        Xc = config(cid, Xfix)
        for r in range(L.REPS):
            for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(Xc, y)):
                if (cid, r, k) not in done: tasks.append((cid, Xc, r, k, tr, va))
    print(f"{len(done)} folds from checkpoint, {len(tasks)} to run", flush=True)
    with open(CKPT, "a") as fh, open(PROBS, "a") as fp:
        for rec, pr in Parallel(n_jobs=int(os.environ.get("NJOBS", "2")), return_as="generator_unordered")(
                delayed(run_fold)(*t) for t in tasks):
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            fp.write(json.dumps(pr) + "\n"); fp.flush()
            print(f"{rec['config']} r={rec['r']} k={rec['k']} nested {rec['f1_nested']*100:.1f} ({time.time()-t0:.0f}s)", flush=True)


def corrected_ci(v, J=50, ratio=0.25):
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se; t = m / se if se > 0 else np.nan
    return {'mean': float(m), 'sd_ddof0': float(v.std()), 'ci95': [float(m - h), float(m + h)],
            't': float(t), 'p_two_sided': float(2 * stats.t.sf(abs(t), J - 1))}


def summarise():
    done = load_done(); ids = sorted({c for c, _, _ in done})
    per = {c: sorted([d for (cc, _, _), d in done.items() if cc == c], key=lambda d: (d['r'], d['k'])) for c in ids}
    for c in ids: assert len(per[c]) == 50, (c, len(per[c]))
    f1 = {c: np.array([d['f1_nested'] for d in per[c]]) for c in ids}
    out = {'configs': {}, 'paired': {}, 'gates': {}}
    probs = {}
    for line in open(PROBS):
        d = json.loads(line); probs.setdefault(d['config'], []).append(d)
    for c in ids:
        agg = np.zeros((len(y), 3))
        for d in probs[c]: agg[np.array(d['va'])] += np.array(d['p_nested'])
        agg /= L.REPS; pa = agg.argmax(1)
        out['configs'][c] = {'n_feat': per[c][0]['n_feat'], 'nested': corrected_ci(f1[c]),
                             'equal_fold_mean': float(np.mean([d['f1_equal'] for d in per[c]])),
                             'agg_oof_macroF1': float(mf1(y, pa)), 'agg_oof_acc': float(accuracy_score(y, pa)),
                             'agg_oof_classwise': classwise(y, pa)}
    for a, b in (('S5', 'S0'), ('S5', 'S3')):
        if a in f1 and b in f1: out['paired'][f'{a}-{b}'] = corrected_ci(f1[a] - f1[b]) | {'folds_a_higher': int((f1[a] > f1[b]).sum())}
    # gates
    if 'S0' in f1:
        ref = json.load(open(ROOT / "code/diagnostics_r1/diag_nested_perfold.json"))['per_fold']
        refv = np.array([d['f1_nested'] for d in sorted(ref, key=lambda d: (d['r'], d['k']))])
        out['gates']['G1_S0_perfold_equals_stored'] = {'max_abs_diff': float(np.abs(f1['S0'] - refv).max()),
                                                       'pass': bool(np.abs(f1['S0'] - refv).max() < 1e-9)}
    st = json.load(open(ROOT / "code/diagnostics_r1/diag_leakage_hs.json"))
    if 'S5' in f1:
        d = abs(f1['S5'].mean() - st['V7_fixedHS_full']['nested']['fold_mean'])
        out['gates']['G2_S5_nested_mean_equals_stored'] = {'abs_diff': float(d), 'pass': bool(d < 1e-9)}
    key = {'S0': 'V7_buggyHS_full', 'S5': 'V7_fixedHS_full', 'S1': 'HS_only_fixed', 'S1b': 'HS_only_buggy',
           'S2': 'MS_only', 'S3': 'RGB+MS (no HS)', 'S4': 'MS+HS_fixed'}
    g3 = {}
    for c in ids:
        d = abs(out['configs'][c]['equal_fold_mean'] - st[key[c]]['equal']['fold_mean'])
        g3[c] = {'abs_diff': float(d), 'pass': bool(d < 1e-9)}
    out['gates']['G3_equal_weight_means_equal_stored'] = g3
    import sklearn, scipy, lightgbm, xgboost
    out['versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                       'scipy': scipy.__version__, 'lightgbm': lightgbm.__version__, 'xgboost': xgboost.__version__}
    json.dump(out, open(OUT / "exp_A_summary.json", "w"), indent=1)
    for c in ids:
        q = out['configs'][c]['nested']
        print(f"{c:4s} nested {q['mean']*100:.1f} ± {q['sd_ddof0']*100:.1f} [{q['ci95'][0]*100:.1f}, {q['ci95'][1]*100:.1f}]"
              f"  equal {out['configs'][c]['equal_fold_mean']*100:.1f}")
    for k_, q in out['paired'].items():
        print(f"{k_}: {q['mean']*100:+.2f} [{q['ci95'][0]*100:+.2f}, {q['ci95'][1]*100:+.2f}] p={q['p_two_sided']:.3f}")
    print(json.dumps(out['gates'], indent=1))


if MODE == 'run':
    run(sys.argv[3:])
elif MODE == 'summarise':
    summarise()
else:
    sys.exit("mode: run <IDs> | summarise")
