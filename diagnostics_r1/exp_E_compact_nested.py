"""Experiment E (pre-registered 31 Oct 2026; protocol in PREREG_compact_nested.md): the compact-feature analysis of Table 10
re-run with nested member weights, HS block as submitted (outer-fold-honest feature selection with nested member weighting).

Protocol: the 50 outer folds of the primary analysis (StratifiedKFold(5, shuffle=True, random_state=42+r), r = 0..9, tiles in
the order of the feature table = folds_10x5.csv). In each outer training part the 149 features are ranked once by LightGBM gain
(rank_infold, copied verbatim from R2_compact.py) and the top-K feature identities are selected, K in 10, 20, 30, 50, 70, 100,
149 (the grid of R2_compact.py). After the top-K identities are selected, the columns are restored to their canonical order in
the 149-feature table before model fitting (implementation correction: R2_compact.py kept them in rank order, so feature order
was an unintended experimental factor). The ensemble is then evaluated as in the primary analysis: StandardScaler fitted on the
outer training part; member weight = Macro-F1 of the member in an inner stratified 5-fold CV (seed 42) of the outer training
part, on the selected features; five members with the Table 4 settings refitted on the outer training part; the outer fold is
scored once. Feature selection is performed once within each outer training part, before the inner CV used to estimate the
weights, so the outer fold is held out from both, while the inner weight estimates are conditional on a subset selected with
the whole outer training part.

Gate E1 (hard): K = 149 selects every column in canonical order, i.e. the primary-analysis input matrix. Each K = 149 fold must
equal experiment A's S0 fold exactly (selected columns, nested weights, Macro-F1, accuracy, confusion matrix, predicted
probabilities; Macro-F1 with validation-fold weights equal to diag_nested_perfold.json). The run stops at the first mismatch.

Usage:  python3 exp_E_compact_nested.py <out_dir> gate        (gate E0, before the lock: R2_compact.py's procedure rerun)
        python3 exp_E_compact_nested.py <out_dir> run         (NJOBS env, default: number of CPU threads)
        python3 exp_E_compact_nested.py <out_dir> summarise   (exp_E_summary.json and Figure 2, V7_compact_curve_nested.png)
Finished folds are appended to exp_E_folds.jsonl and exp_E_probs.jsonl, so an interrupted run resumes. No score is printed
by run."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, time, platform, warnings
from collections import Counter
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/drive/MyDrive/Kaggle_Prepared/code")
import r2_lib as L
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy import stats
import lightgbm as lgb

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1]); MODE = sys.argv[2]
RX = ROOT / "outputs/reviewer_experiments"
CKPT = OUT / "exp_E_folds.jsonl"; PROBS = OUT / "exp_E_probs.jsonl"
X, y, fn, cols = L.load_features(RX / "features_v7.csv")
cols = np.array(cols).astype(str)
K_GRID = [10, 20, 30, 50, 70, 100, 149]
PRIMARY_MEAN = 0.6672084104320107                                   # primary analysis (experiment A, S0)
NJOBS = int(os.environ.get("NJOBS", os.cpu_count() or 1))


def rank_infold(Xtr, ytr, topk=None):
    """LightGBM gain-importance ranking on a (standardized) training fold. Returns column order."""
    sc = StandardScaler(); a = sc.fit_transform(Xtr)
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=20, max_depth=5,
                           min_child_samples=15, subsample=0.7, colsample_bytree=0.7,
                           reg_alpha=1.0, reg_lambda=1.0, random_state=L.SEED, verbose=-1, n_jobs=1)
    m.fit(a, ytr)
    order = np.argsort(m.feature_importances_)[::-1]
    return order if topk is None else order[:topk]


def mf1(yt, p): return f1_score(yt, p, average='macro')


def fit_predict(Xa, ya, Xb):                                         # as in the primary analysis and exp_C
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


def folds():
    return [(r, k, tr, va) for r in range(L.REPS)
            for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=L.SEED + r).split(X, y))]


def folds_ok():
    ref = pd.read_csv(ROOT / "code/folds_10x5.csv"); bad = 0
    for r, k, tr, va in folds():
        bad += set(ref[(ref.repetition == r) & (ref.fold == k)].sample_id) != set(fn[va])
    return bad == 0


def run_fold(K, r, k, tr, va):
    t0 = time.time()
    sel = np.sort(rank_infold(X[tr], y[tr])[:K])                     # top-K identities, canonical column order
    Xc = X[:, sel]
    P = fit_predict(Xc[tr], y[tr], Xc[va]); yv = y[va]
    equal = sum(P.values()) / len(P)
    wl = {n: mf1(yv, P[n].argmax(1)) for n in P}                      # original rule (weights on the scored fold), reference only
    leaky = sum(wl[n] * P[n] for n in P) / sum(wl.values())
    Pin = {n: np.zeros((len(tr), 3)) for n in P}
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(Xc[tr], y[tr]):
        pi = fit_predict(Xc[tr][itr], y[tr][itr], Xc[tr][iva])
        for n in pi: Pin[n][iva] = pi[n]
    wn = {n: mf1(y[tr], Pin[n].argmax(1)) for n in Pin}                # inner-OOF Macro-F1 (training labels only)
    ens = sum(wn[n] * P[n] for n in P) / sum(wn.values()); pk = ens.argmax(1)
    rec = {'K': K, 'r': r, 'k': k, 'sel': [int(i) for i in sel], 'weights_nested': wn,
           'f1_nested': mf1(yv, pk), 'acc_nested': accuracy_score(yv, pk), 'classwise_nested': classwise(yv, pk),
           'f1_equal': mf1(yv, equal.argmax(1)), 'acc_equal': accuracy_score(yv, equal.argmax(1)),
           'f1_leaky': mf1(yv, leaky.argmax(1)), 'seconds': time.time() - t0}
    return rec, {'K': K, 'r': r, 'k': k, 'va': [int(i) for i in va], 'p_nested': ens.round(8).tolist()}


# ---------------- gate E1: K = 149 must reproduce the primary analysis exactly ----------------
_REF = {}
def reference():
    if not _REF:
        _REF['A'] = {(d['r'], d['k']): d for d in map(json.loads, open(OUT / "exp_A_folds.jsonl")) if d['config'] == 'S0'}
        _REF['P'] = {(d['r'], d['k']): d for d in map(json.loads, open(OUT / "exp_A_probs.jsonl")) if d['config'] == 'S0'}
        _REF['L'] = {(d['r'], d['k']): d['f1_leaky'] for d in json.load(open(OUT / "diag_nested_perfold.json"))['per_fold']}
    return _REF


def gate_e1(rec, pr):
    R = reference(); key = (rec['r'], rec['k']); a, ap = R['A'][key], R['P'][key]; bad = []
    if rec['sel'] != list(range(X.shape[1])): bad.append('input matrix')
    for f in ('f1_nested', 'f1_equal', 'acc_nested', 'acc_equal'):
        if abs(rec[f] - a[f]) >= 1e-9: bad.append(f)
    if set(rec['weights_nested']) != set(a['weights_nested']) or \
            any(abs(rec['weights_nested'][n] - a['weights_nested'][n]) >= 1e-9 for n in a['weights_nested']): bad.append('weights')
    if rec['classwise_nested']['cm'] != a['classwise_nested']['cm']: bad.append('confusion matrix')
    if pr['va'] != ap['va'] or np.abs(np.array(pr['p_nested']) - np.array(ap['p_nested'])).max() > 1e-12: bad.append('predictions')
    if abs(rec['f1_leaky'] - R['L'][key]) >= 1e-9: bad.append('Macro-F1 with validation-fold weights')
    return bad


def load(path):
    """Tolerant reader: skips unreadable lines (a hard stop can truncate the last line); a later duplicate replaces an earlier one."""
    done, bad = {}, 0
    if path.exists():
        for line in open(path):
            try: d = json.loads(line); done[(d['K'], d['r'], d['k'])] = d
            except (json.JSONDecodeError, KeyError): bad += 1
    return done, bad


def checkpoint():
    done, b1 = load(CKPT); probs, b2 = load(PROBS); keep = set(done) & set(probs)
    if b1 or b2 or len(keep) != len(done) or len(keep) != len(probs):
        print(f"note: {b1 + b2} unreadable checkpoint lines, {len(set(done) ^ set(probs))} folds in only one file; "
              "these folds are recomputed", flush=True)
    return {k_: done[k_] for k_ in keep}, {k_: probs[k_] for k_ in keep}


def run():
    assert folds_ok(), "folds differ from folds_10x5.csv"
    done, probs = checkpoint(); t0 = time.time()
    for key in sorted(k_ for k_ in done if k_[0] == 149):
        bad = gate_e1(done[key], probs[key])
        if bad: print(f"GATE E1 FAILED (checkpoint) at r={key[1]} k={key[2]}: {bad}", flush=True); sys.exit(2)
    tasks = [(K, r, k, tr, va) for K in sorted(K_GRID, reverse=True) for (r, k, tr, va) in folds() if (K, r, k) not in done]
    print(f"{len(done)} folds from checkpoint, {len(tasks)} to run, NJOBS={NJOBS}", flush=True)
    with open(CKPT, "a") as fh, open(PROBS, "a") as fp:
        for rec, pr in Parallel(n_jobs=NJOBS, return_as="generator_unordered")(delayed(run_fold)(*t) for t in tasks):
            if rec['K'] == 149:
                bad = gate_e1(rec, pr)
                if bad:
                    print(f"GATE E1 FAILED at r={rec['r']} k={rec['k']}: {bad}; run stopped, Table 10 results not to be "
                          "used until the cause is found", flush=True)
                    sys.exit(2)
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush(); fp.write(json.dumps(pr) + "\n"); fp.flush()
            print(f"K={rec['K']} r={rec['r']} k={rec['k']} done ({time.time() - t0:.0f}s)", flush=True)


# ---------------- gate E0 (before the lock): the original procedure of R2_compact.py, rerun ----------------
def original_fold(r, k, tr, va):
    order = rank_infold(X[tr], y[tr])
    sc = StandardScaler(); a_full = sc.fit_transform(X[tr]); b_full = sc.transform(X[va])
    out = {'r': r, 'k': k, 'top30': sorted(cols[order[:30]].tolist())}
    for K in K_GRID:                                                  # as in R2_compact.py: rank order, scored-fold weights
        sel = order[:K]; a, b = a_full[:, sel], b_full[:, sel]; mf, mp = {}, {}
        for nm, md in L.make_models().items():
            md.fit(a, y[tr]); p = md.predict_proba(b); mf[nm] = mf1(y[va], p.argmax(1)); mp[nm] = p
        ens = sum(mf[n] * mp[n] for n in mf) / sum(mf.values())
        out[str(K)] = mf1(y[va], ens.argmax(1))
    return out


def gate():
    res = {'folds_equal_folds_10x5': folds_ok()}
    F = folds()
    per = Parallel(n_jobs=NJOBS)(delayed(original_fold)(r, k, tr, va) for r, k, tr, va in F)
    jun = json.load(open(RX / "V7_compact.json"))
    cur = {str(K): float(np.mean([d[str(K)] for d in per])) for K in K_GRID}
    res['original_procedure_current_environment'] = cur
    res['published_june_2026'] = {str(K): jun['curve'][str(K)]['macroF1'] for K in K_GRID}
    res['max_abs_diff_points'] = float(max(abs(cur[str(K)] - res['published_june_2026'][str(K)]) * 100 for K in K_GRID))
    freq, stab = stability([set(d['top30']) for d in per])
    jf = jun['stability']['frequency_top40']
    res['top30_frequency_current'] = freq
    res['top30_stability_current'] = stab
    res['top30_stability_june'] = {k_: jun['stability'][k_] for k_ in stab}
    res['top30_frequency_june_top40_equal'] = all(freq.get(f, 0) == v for f, v in jf.items())
    r0, k0, tr0, va0 = F[0]
    rec, pr = run_fold(149, r0, k0, tr0, va0)
    res['E1_smoke_K149_r0k0_mismatches'] = gate_e1(rec, pr)
    import sklearn, scipy, xgboost
    res['versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                       'scipy': scipy.__version__, 'lightgbm': lgb.__version__, 'xgboost': xgboost.__version__}
    json.dump(res, open(OUT / "exp_E_gate0.json", "w"), indent=1)
    print(json.dumps({k_: v for k_, v in res.items() if k_ not in ('top30_frequency_current',)}, indent=1))


# ---------------- summary, gate E1 over all folds, Figure 2 ----------------
def stability(sets):                                                 # as in R2_compact.py
    freq = Counter(f for st in sets for f in st)
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))), {
        'mean_pairwise_jaccard': float(np.mean(jac)), 'n_features_ever_in_top30': len(freq),
        'features_in_all_folds': int(sum(v == len(sets) for v in freq.values()))}


def corrected_ci(v, J=50, ratio=0.25):
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se; t = m / se if se > 0 else np.nan
    return {'mean': float(m), 'sd_ddof0': float(v.std()), 'ci95': [float(m - h), float(m + h)], 't': float(t),
            'p_two_sided': float(2 * stats.t.sf(abs(t), J - 1)) if se > 0 else 1.0}


def summarise():
    done, probs = checkpoint(); keys = [(r, k) for r, k, _, _ in folds()]
    for K in K_GRID:
        miss = [rk for rk in keys if (K, *rk) not in done]
        assert not miss, (K, len(miss))
    f1 = {K: np.array([done[(K, r, k)]['f1_nested'] for r, k in keys]) for K in K_GRID}
    out = {'K_grid': K_GRID, 'rows': {}, 'paired_vs_149': {}, 'gates': {}}
    for K in K_GRID:
        out['rows'][str(K)] = corrected_ci(f1[K]) | {
            'leaky_fold_mean': float(np.mean([done[(K, r, k)]['f1_leaky'] for r, k in keys])),
            'equal_fold_mean': float(np.mean([done[(K, r, k)]['f1_equal'] for r, k in keys]))}
        if K != 149:
            d = f1[K] - f1[149]
            out['paired_vs_149'][str(K)] = corrected_ci(d) | {'folds_K_higher': int((d > 0).sum()), 'folds_equal': int((d == 0).sum())}
    mism = {f"{r},{k}": b for r, k in keys if (b := gate_e1(done[(149, r, k)], probs[(149, r, k)]))}
    m149 = float(f1[149].mean())
    out['gates']['E1_K149_equals_primary'] = {'folds_with_mismatch': mism, 'mean': m149, 'primary_mean': PRIMARY_MEAN,
                                              'pass': (not mism) and abs(m149 - PRIMARY_MEAN) < 1e-12}
    out['gates']['E2_folds_equal_folds_10x5'] = {'pass': folds_ok()}
    g0 = OUT / "exp_E_gate0.json"
    if g0.exists(): out['gates']['E0_before_lock'] = {k_: v for k_, v in json.load(open(g0)).items() if k_ != 'top30_frequency_current'}
    out['top30_frequency'], out['top30_stability'] = stability([set(cols[done[(30, r, k)]['sel']]) for r, k in keys])
    import sklearn, scipy, xgboost
    out['versions'] = {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                       'scipy': scipy.__version__, 'lightgbm': lgb.__version__, 'xgboost': xgboost.__version__}
    out['seconds_per_fold_mean'] = {str(K): float(np.mean([done[(K, r, k)]['seconds'] for r, k in keys])) for K in K_GRID}
    json.dump(out, open(OUT / "exp_E_summary.json", "w"), indent=1)
    print(json.dumps(out['gates'], indent=1))
    if out['gates']['E1_K149_equals_primary']['pass']:
        figure(out)
    else:
        print("GATE E1 FAILED: no figure; Table 10 results not to be used until the cause is found")


def figure(out):                                                     # style of figures_r1/fig_compact_r1.py
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = K_GRID; ms = [100 * out['rows'][str(K)]['mean'] for K in ks]
    lo = [100 * out['rows'][str(K)]['ci95'][0] for K in ks]; hi = [100 * out['rows'][str(K)]['ci95'][1] for K in ks]
    full = ms[-1]; d30 = out['paired_vs_149']['30']
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(ks, lo, hi, alpha=0.18, color='#2c7fb8', label='Corrected 95% CI of the mean')
    ax.plot(ks, ms, 'o-', color='#2c7fb8', lw=2, label='Macro-F1 (nested member weights, repeated 10x5 CV)')
    ax.axhline(full, ls='--', color='gray', lw=1, label=f'Full 149-feature model = {full:.1f}%')
    ax.axvline(30, ls=':', color='#d95f0e', lw=1.5)
    ax.annotate(f"K = 30: {ms[ks.index(30)]:.1f}%\n(difference to K = 149: {100 * d30['mean']:+.1f} points)",
                xy=(30, ms[ks.index(30)]), xytext=(42, min(lo) + 0.6), fontsize=9, color='#d95f0e',
                arrowprops=dict(arrowstyle='->', color='#d95f0e', lw=1))
    ax.set_xlabel("Number of features (K, top-ranked inside each outer training part)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "V7_compact_curve_nested.png", dpi=150)
    plt.close()


if MODE == 'gate':
    gate()
elif MODE == 'run':
    run()
elif MODE == 'summarise':
    summarise()
else:
    sys.exit("mode: gate | run | summarise")
