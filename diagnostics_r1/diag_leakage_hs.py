"""Diagnostic (scratchpad only): how much do the reviewer-flagged issues move the numbers?
D1a  ensemble weights: leaky (val-fold F1, current) vs equal vs nested (inner-CV F1 on training fold)
D1b  alpha operating point + 80%-coverage triage threshold chosen on inner OOF only (nested), single-model
D1c  HS axis bug: same pipeline with HS cubes read as (H,W,bands)
Same folds as r2_lib (StratifiedKFold(5, shuffle, seed 42+r), r=0..9), same 5 models."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, time, warnings
import numpy as np, tifffile
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/drive/MyDrive/Kaggle_Prepared/code")
import r2_lib as L, feat_v7_lib as F
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1])
H, O, R = 0, 1, 2
X, y, fn, cols = L.load_features(ROOT / "outputs/reviewer_experiments/features_v7.csv")
cols = np.array(cols)
hs_i = np.where(np.char.startswith(cols.astype(str), "hs_"))[0]
ms_i = np.where(np.char.startswith(cols.astype(str), "ms_"))[0]
rgb_i = np.where(np.char.startswith(cols.astype(str), "rgb_"))[0]

# ---- corrected HS features: cube is (H, W, bands); keep first 125 bands for the 126-band files
t0 = time.time()
HSF = np.array([F.extract_hs_raw_features(
    tifffile.imread(ROOT / "train" / "HS" / f"{s}.tif").astype(np.float32)[..., :125]) for s in fn])
HSF = np.nan_to_num(HSF, nan=0, posinf=0, neginf=0)
Xfix = X.copy(); Xfix[:, hs_i] = HSF
print(f"corrected HS features built in {time.time()-t0:.0f}s; non-constant HS cols: buggy "
      f"{int((X[:, hs_i].std(0) > 0).sum())} -> fixed {int((HSF.std(0) > 0).sum())}", flush=True)

def mf1(yt, p): return f1_score(yt, p, average='macro')

def alpha_pick(P, yt):
    """paper's rule: smallest alpha (1.00..2.00 step .05) with Health recall >= 0.47, else best macro-F1"""
    grid = np.arange(1.0, 2.0001, 0.05)
    def m(a):
        q = P.copy(); q[:, H] *= a; pr = q.argmax(1)
        return mf1(yt, pr), recall_score(yt, pr, labels=[H], average='macro')
    for a in grid:
        if m(a)[1] >= 0.47: return float(a)
    return float(max(grid, key=lambda a: m(a)[0]))

def fit_predict(Xa, ya, Xb):
    sc = StandardScaler().fit(Xa); a, b = sc.transform(Xa), sc.transform(Xb)
    out = {}
    for nm, md in L.make_models().items():
        md.fit(a, ya); out[nm] = md.predict_proba(b)
    return out

def run_fold(Xc, r, k, tr, va, nested):
    P = fit_predict(Xc[tr], y[tr], Xc[va]); yv = y[va]
    wl = {n: mf1(yv, P[n].argmax(1)) for n in P}                       # leaky: val-fold labels
    leaky = sum(wl[n] * P[n] for n in P) / sum(wl.values())
    equal = sum(P.values()) / len(P)
    res = {'r': r, 'k': k, 'va': va, 'p_leaky': leaky, 'p_equal': equal,
           'f1_leaky': mf1(yv, leaky.argmax(1)), 'f1_equal': mf1(yv, equal.argmax(1)),
           'acc_equal': accuracy_score(yv, equal.argmax(1))}
    if nested:
        Pin = {n: np.zeros((len(tr), 3)) for n in P}
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(Xc[tr], y[tr]):
            pi = fit_predict(Xc[tr][itr], y[tr][itr], Xc[tr][iva])
            for n in pi: Pin[n][iva] = pi[n]
        wn = {n: mf1(y[tr], Pin[n].argmax(1)) for n in Pin}            # nested: training-fold labels only
        ens_in = sum(wn[n] * Pin[n] for n in Pin) / sum(wn.values())
        ens = sum(wn[n] * P[n] for n in P) / sum(wn.values())
        a = alpha_pick(ens_in, y[tr]); q = ens.copy(); q[:, H] *= a; pa = q.argmax(1)
        t = np.quantile(ens_in.max(1), 0.20)                              # 80%-coverage threshold from inner OOF
        keep = ens.max(1) >= t; pk = ens.argmax(1)
        tt = np.quantile(ens.max(1), 0.20); keep_t = ens.max(1) >= tt     # transductive (on the eval fold itself)
        def tri(kp):
            yk, pkk = yv[kp], pk[kp]
            rr = recall_score(yk, pkk, labels=[R], average='macro') if (yk == R).any() else np.nan
            return {'cov': float(kp.mean()), 'acc': accuracy_score(yk, pkk), 'f1': mf1(yk, pkk), 'rust_miss': 1 - rr}
        res.update({'p_nested': ens, 'f1_nested': mf1(yv, ens.argmax(1)),
                    'acc_nested': accuracy_score(yv, ens.argmax(1)),
                    'alpha': a, 'f1_alpha': mf1(yv, pa),
                    'hrec_alpha': recall_score(yv, pa, labels=[H], average='macro'),
                    'hrec_base': recall_score(yv, ens.argmax(1), labels=[H], average='macro'),
                    'rustmiss_full': 1 - recall_score(yv, pk, labels=[R], average='macro'),
                    'tri_nested': tri(keep), 'tri_transductive': tri(keep_t)})
    return res

def ece(P, yt, bins=10):
    c = P.max(1); acc = (P.argmax(1) == yt).astype(float); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.sum(): e += m.mean() * abs(acc[m].mean() - c[m].mean())
    return float(e)

def evaluate(name, Xc, nested):
    t = time.time()
    tasks = [(r, k, tr, va) for r in range(L.REPS)
             for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(Xc, y))]
    res = Parallel(n_jobs=12)(delayed(run_fold)(Xc, r, k, tr, va, nested) for r, k, tr, va in tasks)
    out = {'n_feat': int(Xc.shape[1])}
    for key in ['leaky', 'equal'] + (['nested'] if nested else []):
        f = np.array([d['f1_' + key] for d in res])
        agg = np.zeros((len(y), 3)); pooled_p, pooled_y = [], []
        for d in res: agg[d['va']] += d['p_' + key]; pooled_p.append(d['p_' + key]); pooled_y.append(y[d['va']])
        agg /= L.REPS
        out[key] = {'fold_mean': f.mean(), 'fold_std': f.std(),
                    'agg_oof_f1': mf1(y, agg.argmax(1)), 'agg_oof_acc': accuracy_score(y, agg.argmax(1)),
                    'ece_agg_oof': ece(agg, y), 'ece_single_model': ece(np.vstack(pooled_p), np.concatenate(pooled_y))}
    if nested:
        g = lambda k_: float(np.mean([d[k_] for d in res]))
        out['alpha_nested'] = {'alpha_mean': g('alpha'), 'alpha_range': [min(d['alpha'] for d in res), max(d['alpha'] for d in res)],
                               'f1_base': g('f1_nested'), 'f1_alpha': g('f1_alpha'), 'hrec_base': g('hrec_base'), 'hrec_alpha': g('hrec_alpha')}
        for tk in ['tri_nested', 'tri_transductive']:
            out[tk] = {m: float(np.nanmean([d[tk][m] for d in res])) for m in ['cov', 'acc', 'f1', 'rust_miss']}
        out['full_cov'] = {'acc': g('acc_nested'), 'f1': g('f1_nested'), 'rust_miss': float(np.nanmean([d['rustmiss_full'] for d in res]))}
    print(f"[{name}] done in {time.time()-t:.0f}s: " + ", ".join(
        f"{k} {out[k]['fold_mean']*100:.1f}" for k in ['leaky', 'equal', 'nested'] if k in out), flush=True)
    return out

results = {}
results['V7_buggyHS_full'] = evaluate('V7 full (current, buggy HS)', X, nested=True)
results['V7_fixedHS_full'] = evaluate('V7 full (HS fixed)', Xfix, nested=True)
cfg = {'MS_only': X[:, ms_i], 'RGB+MS (no HS)': X[:, np.r_[rgb_i, ms_i]],
       'HS_only_buggy': X[:, hs_i], 'HS_only_fixed': Xfix[:, hs_i], 'MS+HS_fixed': Xfix[:, np.r_[ms_i, hs_i]]}
for nm, Xc in cfg.items(): results[nm] = evaluate(nm, Xc, nested=False)

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(clean(results), open(OUT / "diag_leakage_hs.json", "w"), indent=2)
print("saved", OUT / "diag_leakage_hs.json")
