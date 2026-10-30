"""Experiment D (pre-registered 29 Oct 2026; protocol in PREREG_aux_hyperparam.md): sensitivity of the ensemble estimate to
hyperparameter selection (Reviewer 4, Comment Q2.1). Three rows, each with nested member weights, on the 50 outer folds of
the primary analysis:
  fixed   : the Table 4 settings (= the primary analysis; gate D1 requires it to equal experiment A per fold);
  default : library defaults (only random_state, n_jobs and logging set);
  tuned   : for every member, the setting with the highest Macro-F1 in an inner stratified 5-fold CV (seed 42) on the outer
            training part is chosen from a fixed grid; the member weight is that inner out-of-fold Macro-F1; the chosen
            settings are refitted on the outer training part and the outer fold is scored once.
Grid: LightGBM, XGBoost, RandomForest, ExtraTrees: max_depth {3, 5, 7} x n_estimators {100, 200, 300}, other settings as in
Table 4; SVM (RBF): C {0.3, 1, 3} x gamma {0.5, 1, 2} x gamma_scale, gamma_scale = 1 / (n_features * Var(X_train)), the
'scale' rule of scikit-learn. Ties are broken by the first setting in the grid order above.
HS block as submitted (primary, 'sub') and corrected HS features (secondary, 'cor').
Usage:  python3 exp_D_nested_hyperparams.py <out_dir> gate                (folds and the SVM gamma rule; fits one SVM)
        python3 exp_D_nested_hyperparams.py <out_dir> run [sub|cor|all]   (NJOBS env, default 6)
        python3 exp_D_nested_hyperparams.py <out_dir> summarise
Finished folds are appended to exp_D_folds.jsonl, so an interrupted run resumes."""
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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import lightgbm as lgb, xgboost as xgb
from scipy import stats

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1]); MODE = sys.argv[2]
CKPT = OUT / "exp_D_folds.jsonl"; SEED = L.SEED
X, y, fn, cols = L.load_features(ROOT / "outputs/reviewer_experiments/features_v7.csv")
hs_i = np.where(np.char.startswith(np.array(cols).astype(str), "hs_"))[0]
LEARNERS = ['LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees', 'SVM']
TREE_GRID = [(d, n) for d in (3, 5, 7) for n in (100, 200, 300)]
SVM_GRID = [(c, g) for c in (0.3, 1.0, 3.0) for g in (0.5, 1.0, 2.0)]
GRID = {nm: (SVM_GRID if nm == 'SVM' else TREE_GRID) for nm in LEARNERS}


def matrix(hs):
    if hs == 'sub': return X
    Xc = X.copy(); Xc[:, hs_i] = np.load(OUT / "exp_A_corrected_hs_features.npy"); return Xc


def gamma_scale(Xfit): return 1.0 / (Xfit.shape[1] * Xfit.var())


def make(nm, cfg, Xfit):
    if nm == 'SVM':
        c, g = cfg; return SVC(C=c, kernel='rbf', gamma=g * gamma_scale(Xfit), probability=True, random_state=SEED)
    d, n = cfg
    if nm == 'LightGBM':
        return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=20, max_depth=d, min_child_samples=15,
                                  subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0, random_state=SEED,
                                  verbose=-1, n_jobs=1)
    if nm == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=n, learning_rate=0.05, max_depth=d, min_child_weight=8, subsample=0.7,
                                 colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0, random_state=SEED,
                                 eval_metric='mlogloss', verbosity=0, n_jobs=1)
    cls = RandomForestClassifier if nm == 'RandomForest' else ExtraTreesClassifier
    return cls(n_estimators=n, max_depth=d, min_samples_split=8, min_samples_leaf=4, max_features=0.5, random_state=SEED, n_jobs=1)


def defaults():
    return {'LightGBM': lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1),
            'XGBoost': xgb.XGBClassifier(random_state=SEED, eval_metric='mlogloss', verbosity=0, n_jobs=1),
            'RandomForest': RandomForestClassifier(random_state=SEED, n_jobs=1),
            'ExtraTrees': ExtraTreesClassifier(random_state=SEED, n_jobs=1),
            'SVM': SVC(probability=True, random_state=SEED)}


def mf1(yt, p): return f1_score(yt, p, average='macro')


def fit_predict(factory, Xa, ya, Xb):
    sc = StandardScaler().fit(Xa); a, b = sc.transform(Xa), sc.transform(Xb); out = {}
    for nm, md in factory().items():
        md.fit(a, ya); out[nm] = md.predict_proba(b)
    return out


def nested_fixed_rule(factory, Xc, tr, va):          # as experiment A: inner-OOF Macro-F1 weights, fixed settings
    P = fit_predict(factory, Xc[tr], y[tr], Xc[va])
    Pin = {n: np.zeros((len(tr), 3)) for n in P}
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc[tr], y[tr]):
        pi = fit_predict(factory, Xc[tr][itr], y[tr][itr], Xc[tr][iva])
        for n in pi: Pin[n][iva] = pi[n]
    w = {n: mf1(y[tr], Pin[n].argmax(1)) for n in Pin}
    ens = sum(w[n] * P[n] for n in P) / sum(w.values()); return ens, w, None, None


def nested_tuned(Xc, tr, va):
    yt = y[tr]; oof = {(nm, cfg): np.zeros((len(tr), 3)) for nm in LEARNERS for cfg in GRID[nm]}
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc[tr], yt):
        sc = StandardScaler().fit(Xc[tr][itr]); a = sc.transform(Xc[tr][itr]); b = sc.transform(Xc[tr][iva])
        for nm in LEARNERS:
            for cfg in GRID[nm]:
                md = make(nm, cfg, a); md.fit(a, yt[itr]); oof[(nm, cfg)][iva] = md.predict_proba(b)
    score = {key: mf1(yt, p.argmax(1)) for key, p in oof.items()}
    sel = {nm: max(GRID[nm], key=lambda c: score[(nm, c)]) for nm in LEARNERS}   # max() keeps the first of tied settings
    w = {nm: score[(nm, sel[nm])] for nm in LEARNERS}
    sc = StandardScaler().fit(Xc[tr]); a, b = sc.transform(Xc[tr]), sc.transform(Xc[va]); P = {}
    for nm in LEARNERS:
        md = make(nm, sel[nm], a); md.fit(a, yt); P[nm] = md.predict_proba(b)
    ens = sum(w[n] * P[n] for n in P) / sum(w.values())
    return ens, w, {nm: list(sel[nm]) for nm in LEARNERS}, {f"{nm}|{c[0]}|{c[1]}": s for (nm, c), s in score.items()}


def run_fold(row, hs, Xc, r, k, tr, va):
    t0 = time.time()
    if row == 'fixed': ens, w, sel, inner = nested_fixed_rule(L.make_models, Xc, tr, va)
    elif row == 'default': ens, w, sel, inner = nested_fixed_rule(defaults, Xc, tr, va)
    else: ens, w, sel, inner = nested_tuned(Xc, tr, va)
    pk = ens.argmax(1)
    return {'row': row, 'hs': hs, 'r': r, 'k': k, 'f1': mf1(y[va], pk), 'acc': accuracy_score(y[va], pk), 'weights': w,
            'selected': sel, 'inner_scores': inner, 'seconds': time.time() - t0}


def folds_ok():
    ref = pd.read_csv(ROOT / "code/folds_10x5.csv"); bad = 0
    for r in range(L.REPS):
        for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y)):
            bad += set(ref[(ref.repetition == r) & (ref.fold == k)].sample_id) != set(fn[va])
    return bad == 0


def gate():
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    a = StandardScaler().fit_transform(X[tr]); s = SVC(gamma='scale', probability=True, random_state=SEED).fit(a, y[tr])
    res = {'folds_equal_folds_10x5': folds_ok(), 'svm_gamma_scale_rule': {'sklearn': float(s._gamma), 'ours': gamma_scale(a)},
           'grid_sizes': {nm: len(GRID[nm]) for nm in LEARNERS}}
    res['pass'] = bool(res['folds_equal_folds_10x5'] and abs(s._gamma - gamma_scale(a)) < 1e-15)
    json.dump(res, open(OUT / "exp_D_gate.json", "w"), indent=1); print(json.dumps(res, indent=1))


def load_done():
    done = {}
    if CKPT.exists():
        for line in open(CKPT):
            d = json.loads(line); done[(d['row'], d['hs'], d['r'], d['k'])] = d
    return done


def run(hs_list):
    assert folds_ok(), "folds differ from folds_10x5.csv"
    done = load_done(); tasks = []; t0 = time.time()
    for hs in hs_list:
        Xc = matrix(hs)
        for row in ('fixed', 'default', 'tuned'):
            for r in range(L.REPS):
                for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(Xc, y)):
                    if (row, hs, r, k) not in done: tasks.append((row, hs, Xc, r, k, tr, va))
    tasks.sort(key=lambda t: t[0] != 'tuned')                       # long tasks first
    print(f"{len(done)} folds from checkpoint, {len(tasks)} to run", flush=True)
    with open(CKPT, "a") as fh:
        for rec in Parallel(n_jobs=int(os.environ.get("NJOBS", "6")), return_as="generator_unordered")(delayed(run_fold)(*t) for t in tasks):
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(f"{rec['row']}/{rec['hs']} r={rec['r']} k={rec['k']} {rec['f1']*100:.1f} ({time.time()-t0:.0f}s)", flush=True)


def corrected_ci(v, J=50, ratio=0.25):
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se; t = m / se if se > 0 else np.nan
    return {'mean': float(m), 'sd_ddof0': float(v.std()), 'ci95': [float(m - h), float(m + h)], 't': float(t),
            'p_two_sided': float(2 * stats.t.sf(abs(t), J - 1)) if se > 0 else 1.0}


def summarise():
    done = load_done(); out = {'rows': {}, 'paired': {}, 'selection_counts': {}, 'gates': {}}
    keys = sorted({(r, k) for (_, _, r, k) in done}); f = {}
    for (row, hs) in sorted({(a, b) for (a, b, _, _) in done}):
        v = [done[(row, hs, r, k)] for (r, k) in keys if (row, hs, r, k) in done]
        assert len(v) == 50, (row, hs, len(v))
        f[(row, hs)] = np.array([d['f1'] for d in v]); out['rows'][f'{row}_{hs}'] = corrected_ci(f[(row, hs)])
        if row == 'tuned':
            cnt = {}
            for d in v:
                for nm, c in d['selected'].items(): cnt.setdefault(nm, {}).setdefault(f"{c[0]}|{c[1]}", 0); cnt[nm][f"{c[0]}|{c[1]}"] += 1
            out['selection_counts'][hs] = cnt
    for hs in ('sub', 'cor'):
        for row in ('tuned', 'default'):
            if (row, hs) in f and ('fixed', hs) in f:
                d = f[(row, hs)] - f[('fixed', hs)]
                out['paired'][f'{row}_{hs} - fixed_{hs}'] = corrected_ci(d) | {'folds_higher': int((d > 0).sum())}
    B = {(d['r'], d['k']): d['primary'] for d in json.load(open(OUT / "exp_B_summary.json"))['per_fold']}
    dl = np.array([B[k] for k in keys])
    for row in ('fixed', 'tuned'):
        if (row, 'sub') in f:
            d = dl - f[(row, 'sub')]
            out['paired'][f'DL - {row}_sub'] = corrected_ci(d) | {'folds_DL_higher': int((d > 0).sum())}
    A = {}
    for line in open(OUT / "exp_A_folds.jsonl"):
        d = json.loads(line); A[(d['config'], d['r'], d['k'])] = d['f1_nested']
    for hs, s0 in (('sub', 'S0'), ('cor', 'S5')):
        if ('fixed', hs) in f:
            m = float(np.abs(f[('fixed', hs)] - np.array([A[(s0, r, k)] for (r, k) in keys])).max())
            out['gates'][f'D1_fixed_{hs}_equals_exp_A_{s0}'] = {'max_abs_diff': m, 'pass': m < 1e-9}
    if (OUT / "exp_D_gate.json").exists(): out['gates']['D2_before_lock'] = json.load(open(OUT / "exp_D_gate.json"))
    import sklearn, scipy
    out['versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                       'scipy': scipy.__version__, 'lightgbm': lgb.__version__, 'xgboost': xgb.__version__}
    out['seconds_per_fold_mean'] = {f'{row}_{hs}': float(np.mean([done[(row, hs, r, k)]['seconds'] for (r, k) in keys]))
                                    for (row, hs) in f}
    json.dump(out, open(OUT / "exp_D_summary.json", "w"), indent=1)
    for k_, q in out['rows'].items(): print(f"{k_:12s} {q['mean']*100:.1f} ± {q['sd_ddof0']*100:.1f} [{q['ci95'][0]*100:.1f}, {q['ci95'][1]*100:.1f}]")
    for k_, q in out['paired'].items(): print(f"{k_:22s} {q['mean']*100:+.2f} [{q['ci95'][0]*100:+.2f}, {q['ci95'][1]*100:+.2f}] p={q['p_two_sided']:.3f}")
    print(json.dumps(out['gates'], indent=1))


if MODE == 'gate':
    gate()
elif MODE == 'run':
    a = sys.argv[3] if len(sys.argv) > 3 else 'all'; run(['sub', 'cor'] if a == 'all' else [a])
elif MODE == 'summarise':
    summarise()
else:
    sys.exit("mode: gate | run [sub|cor|all] | summarise")
