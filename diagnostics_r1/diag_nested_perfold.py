"""Re-run of the nested part of diag_leakage_hs.py for the evaluated pipeline (149 features, HS block as
submitted), keeping what that script did not store (revision, October 2026):
  - the Macro-F1 of every outer fold (nested, validation-fold and equal weights), for a paired comparison
    with the fair deep baseline on the folds of repetition 0;
  - selective prediction with the cut-off chosen on inner-fold predictions at several target coverages,
    with the confusion matrix of the retained tiles.
Same folds, models, seeds and rules as diag_leakage_hs.py. Before anything is used, the means of this
run are checked against diag_leakage_hs.json (V7_buggyHS_full). Usage: python3 diag_nested_perfold.py <out_dir>"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, time, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/drive/MyDrive/Kaggle_Prepared/code")
import r2_lib as L
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1])
H, O, R = 0, 1, 2
COVERAGES = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
X, y, fn, cols = L.load_features(ROOT / "outputs/reviewer_experiments/features_v7.csv")


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


def tri(yv, pk, kp):
    yk, pkk = yv[kp], pk[kp]
    rr = recall_score(yk, pkk, labels=[R], average='macro') if (yk == R).any() else np.nan
    return {'cov': float(kp.mean()), 'acc': accuracy_score(yk, pkk), 'f1': mf1(yk, pkk), 'rust_miss': 1 - rr,
            'cm': confusion_matrix(yk, pkk, labels=[H, O, R]).tolist()}


def run_fold(r, k, tr, va):
    P = fit_predict(X[tr], y[tr], X[va]); yv = y[va]
    wl = {n: mf1(yv, P[n].argmax(1)) for n in P}
    leaky = sum(wl[n] * P[n] for n in P) / sum(wl.values())
    equal = sum(P.values()) / len(P)
    Pin = {n: np.zeros((len(tr), 3)) for n in P}
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(X[tr], y[tr]):
        pi = fit_predict(X[tr][itr], y[tr][itr], X[tr][iva])
        for n in pi: Pin[n][iva] = pi[n]
    wn = {n: mf1(y[tr], Pin[n].argmax(1)) for n in Pin}
    ens_in = sum(wn[n] * Pin[n] for n in Pin) / sum(wn.values())
    ens = sum(wn[n] * P[n] for n in P) / sum(wn.values())
    a = alpha_pick(ens_in, y[tr]); q = ens.copy(); q[:, H] *= a; pa = q.argmax(1)
    pk = ens.argmax(1)
    t80 = np.quantile(ens_in.max(1), 0.20); keep = ens.max(1) >= t80                 # as in diag_leakage_hs.py
    tt = np.quantile(ens.max(1), 0.20); keep_t = ens.max(1) >= tt
    sweep = {}
    for c in COVERAGES:                                                             # cut-off from inner predictions
        kp = np.ones(len(va), bool) if c >= 1.0 else ens.max(1) >= np.quantile(ens_in.max(1), 1 - c)
        sweep[f'{c:.2f}'] = tri(yv, pk, kp)
    return {'r': r, 'k': k, 'va': va.tolist(), 'p_nested': ens.tolist(),
            'f1_leaky': mf1(yv, leaky.argmax(1)), 'f1_equal': mf1(yv, equal.argmax(1)),
            'f1_nested': mf1(yv, pk), 'acc_nested': accuracy_score(yv, pk),
            'weights_nested': wn, 'alpha': a, 'f1_alpha': mf1(yv, pa),
            'hrec_alpha': recall_score(yv, pa, labels=[H], average='macro'),
            'hrec_base': recall_score(yv, pk, labels=[H], average='macro'),
            'rustmiss_full': 1 - recall_score(yv, pk, labels=[R], average='macro'),
            'tri_nested': tri(yv, pk, keep), 'tri_transductive': tri(yv, pk, keep_t), 'sweep': sweep}


t0 = time.time()
tasks = [(r, k, tr, va) for r in range(L.REPS)
         for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(X, y))]
# each finished fold is appended to a JSONL checkpoint, so an interrupted run resumes where it stopped
CKPT = OUT / "diag_nested_perfold_folds.jsonl"
done = {}
if CKPT.exists():
    for line in open(CKPT):
        d = json.loads(line); done[(d['r'], d['k'])] = d
todo = [t for t in tasks if (t[0], t[1]) not in done]
print(f"{len(done)} folds from checkpoint, {len(todo)} to run", flush=True)
with open(CKPT, "a") as fh:
    for d in Parallel(n_jobs=int(os.environ.get("NJOBS", "2")), return_as="generator_unordered")(
            delayed(run_fold)(*t) for t in todo):
        fh.write(json.dumps(d, default=float) + "\n"); fh.flush(); done[(d['r'], d['k'])] = d
        print(f"fold r={d['r']} k={d['k']} done ({len(done)}/50, {time.time()-t0:.0f}s)", flush=True)
res = sorted(done.values(), key=lambda d: (d['r'], d['k']))
assert len(res) == 50
g = lambda key: float(np.mean([d[key] for d in res]))
summary = {
    'fold_mean': {w: g('f1_' + w) for w in ['nested', 'leaky', 'equal']},
    'fold_std_ddof0': {w: float(np.std([d['f1_' + w] for d in res])) for w in ['nested', 'leaky', 'equal']},
    'alpha_nested': {'alpha_mean': g('alpha'), 'f1_base': g('f1_nested'), 'f1_alpha': g('f1_alpha'),
                     'hrec_base': g('hrec_base'), 'hrec_alpha': g('hrec_alpha')},
    'tri_nested': {m: float(np.nanmean([d['tri_nested'][m] for d in res])) for m in ['cov', 'acc', 'f1', 'rust_miss']},
    'tri_transductive': {m: float(np.nanmean([d['tri_transductive'][m] for d in res])) for m in ['cov', 'acc', 'f1', 'rust_miss']},
    'full_cov': {'acc': g('acc_nested'), 'f1': g('f1_nested'), 'rust_miss': g('rustmiss_full')},
    'sweep_fold_means': {c: {m: float(np.nanmean([d['sweep'][c][m] for d in res])) for m in ['cov', 'acc', 'f1', 'rust_miss']}
                         for c in res[0]['sweep']},
    'sweep_cm_pooled_50_folds': {c: np.sum([d['sweep'][c]['cm'] for d in res], 0).tolist() for c in res[0]['sweep']},
    'seconds': time.time() - t0,
}
# reproduction check against the stored diagnostic (must agree before any number is used)
ref = json.load(open(ROOT / "code/diagnostics_r1/diag_leakage_hs.json"))['V7_buggyHS_full']
pairs = [('nested fold mean', summary['fold_mean']['nested'], ref['nested']['fold_mean']),
         ('validation-fold fold mean', summary['fold_mean']['leaky'], ref['leaky']['fold_mean']),
         ('equal fold mean', summary['fold_mean']['equal'], ref['equal']['fold_mean']),
         ('alpha hrec_base', summary['alpha_nested']['hrec_base'], ref['alpha_nested']['hrec_base']),
         ('alpha hrec_alpha', summary['alpha_nested']['hrec_alpha'], ref['alpha_nested']['hrec_alpha']),
         ('tri_nested cov', summary['tri_nested']['cov'], ref['tri_nested']['cov']),
         ('tri_nested acc', summary['tri_nested']['acc'], ref['tri_nested']['acc']),
         ('tri_nested rust_miss', summary['tri_nested']['rust_miss'], ref['tri_nested']['rust_miss']),
         ('tri_transductive rust_miss', summary['tri_transductive']['rust_miss'], ref['tri_transductive']['rust_miss']),
         ('full_cov rust_miss', summary['full_cov']['rust_miss'], ref['full_cov']['rust_miss'])]
check = {name: {'rerun': a, 'stored': b, 'abs_diff': abs(a - b)} for name, a, b in pairs}
summary['reproduction_check'] = check
summary['reproduced'] = bool(max(v['abs_diff'] for v in check.values()) < 1e-9)
json.dump({'summary': summary, 'per_fold': [{k_: v for k_, v in d.items() if k_ not in ('va', 'p_nested')} for d in res]},
          open(OUT / "diag_nested_perfold.json", "w"), indent=1, default=float)
np.savez_compressed(OUT / "diag_nested_perfold_probs.npz",
                    r=np.array([d['r'] for d in res]), k=np.array([d['k'] for d in res]),
                    va=np.array([d['va'] for d in res], dtype=object), p=np.array([d['p_nested'] for d in res], dtype=object))
for name, v in check.items():
    print(f"{name:28s} rerun {v['rerun']:.6f}  stored {v['stored']:.6f}  diff {v['abs_diff']:.2e}")
print("REPRODUCED" if summary['reproduced'] else "NOT REPRODUCED", f"({summary['seconds']:.0f}s)")
