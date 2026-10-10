"""
B2 — MLP baseline on the V7 149 engineered features (replaces the old 226-feature MLP row).

Same MLP as the earlier manuscript (105_paper_pipeline_rerun.py, STEP 3):
  sklearn MLPClassifier(hidden=(256,128), relu, adam, alpha=1e-4, max_iter=500,
  early_stopping=True, validation_fraction=0.15, random_state=SEED)
Only two things change, so the row is consistent with the rest of tab:main_results:
  - features: V7 149 (features_v7.csv) instead of the old 226 (features_226.csv)
  - protocol: the locked repeated 10x5 stratified CV of r2_lib (same folds as the ensemble),
    StandardScaler fit in-fold, fixed class order ['Health','Other','Rust'], SEED=42.
Sanity check first: the same MLP on features_226.csv with the old single 5-fold split must
reproduce the old manuscript number (62.89% +/- 3.2).
Also measures single-core CPU cost on all 600 samples (fit time, inference, #params) for tab:compute.
Output: outputs/reviewer_experiments/B2_mlp_v7.json
"""
import warnings, json, time, pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import r2_lib as L

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', max_iter=500,
                         early_stopping=True, validation_fraction=0.15,
                         random_state=L.SEED, alpha=1e-4)


def mlp_cv(X, y, reps):
    """reps x 5 stratified CV with the r2_lib folds (seed SEED+r). Returns per-fold F1/acc, avg OOF probs."""
    ff, fa, nit = [], [], []; oof_sum = np.zeros((len(y), 3)); t0 = time.time()
    for r in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=L.SEED + r)   # identical to r2_lib.one_cv
        for tr, va in skf.split(X, y):
            sc = StandardScaler(); a = sc.fit_transform(X[tr]); b = sc.transform(X[va])
            m = make_mlp().fit(a, y[tr]); p = m.predict_proba(b)
            ff.append(f1_score(y[va], p.argmax(1), average='macro'))
            fa.append(accuracy_score(y[va], p.argmax(1)))
            nit.append(m.n_iter_); oof_sum[va] += p
        print(f"  rep {r+1}/{reps}: F1 {np.mean(ff[-5:])*100:.2f}%  ({time.time()-t0:.0f}s)")
    return np.array(ff), np.array(fa), np.array(nit), oof_sum / reps


with threadpool_limits(1):   # single CPU core, like the ensemble members (n_jobs=1)
    # ---- 1. sanity: reproduce the old 226-feature number (old protocol = one 5-fold pass, seed 42)
    print("Sanity check: MLP on features_226.csv, single 5-fold (old protocol)")
    X226, y226, _, _ = L.load_features(OUT / "features_226.csv")
    f226, a226, _, _ = mlp_cv(X226, y226, reps=1)
    print(f"  -> F1 {f226.mean()*100:.2f}% +/- {f226.std()*100:.1f}  (old manuscript: 62.89% +/- 3.2)")

    # ---- 2. main: V7 149 features, locked repeated 10x5 protocol
    X, y, _, cols = L.load_features(OUT / "features_v7.csv")
    assert X.shape == (600, 149), X.shape
    print(f"\nMLP on V7 features {X.shape}, repeated {L.REPS}x5 CV")
    ff, fa, nit, oof = mlp_cv(X, y, reps=L.REPS)
    rng = np.random.RandomState(L.SEED)   # same bootstrap as r2_lib.repeated_cv
    boot = [rng.choice(ff, len(ff), replace=True).mean() for _ in range(10000)]
    ci = np.percentile(boot, [2.5, 97.5])
    pred = oof.argmax(1)

    # ---- 3. cost on all 600 samples (median of 5 runs)
    Xs = StandardScaler().fit_transform(X); fit_t, inf_t = [], []
    for _ in range(5):
        t = time.time(); m = make_mlp().fit(Xs, y); fit_t.append(time.time() - t)
        t = time.time(); m.predict_proba(Xs); inf_t.append(time.time() - t)
    n_params = int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))

res = {
    'model': 'MLPClassifier(256,128) relu, alpha=1e-4, early_stopping (val 0.15), max_iter=500',
    'features': 'V7 149 (features_v7.csv)', 'protocol': 'repeated 10x5 stratified CV (r2_lib folds), scaler in-fold',
    'cv_macroF1_mean': float(ff.mean()), 'cv_macroF1_std': float(ff.std()), 'ci95': [float(ci[0]), float(ci[1])],
    'cv_acc_mean': float(fa.mean()), 'cv_acc_std': float(fa.std()),
    'oof_macroF1': float(f1_score(y, pred, average='macro')), 'oof_acc': float(accuracy_score(y, pred)),
    'per_class': L.per_class_stats(y, pred),
    'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(), 'cm_rows_cols': L.CLASSES,
    'epochs_median': float(np.median(nit)), 'epochs_range': [int(nit.min()), int(nit.max())],
    'fold_f1': [float(v) for v in ff],
    'cost_600': {'threads': 1, 'train_seconds_median': float(np.median(fit_t)),
                 'infer_ms_per_sample': float(1000 * np.median(inf_t) / len(y)),
                 'n_params': n_params, 'model_size_kb': float(len(pickle.dumps(m)) / 1024)},
    'sanity_226_single5fold': {'f1_mean': float(f226.mean()), 'f1_std': float(f226.std()),
                               'acc_mean': float(a226.mean()), 'acc_std': float(a226.std()),
                               'old_manuscript': '62.89% +/- 3.2 (acc 63.50% +/- 3.8)'},
}
json.dump(res, open(OUT / "B2_mlp_v7.json", 'w'), indent=2)

print("\n" + "=" * 60)
print(f"MLP (V7 149)  CV macro-F1 {res['cv_macroF1_mean']*100:.1f}% +/- {res['cv_macroF1_std']*100:.1f}"
      f"  [95% CI {ci[0]*100:.1f}, {ci[1]*100:.1f}]   CV acc {res['cv_acc_mean']*100:.1f}%")
print(f"  OOF macro-F1 {res['oof_macroF1']*100:.1f}%  OOF acc {res['oof_acc']*100:.1f}%")
print("  per-class F1: " + ", ".join(f"{c} {v['F1']*100:.1f}" for c, v in res['per_class'].items()))
print(f"  epochs median {res['epochs_median']:.0f} (range {res['epochs_range']})")
c = res['cost_600']
print(f"  cost (1 core, n=600): train {c['train_seconds_median']:.2f}s, infer {c['infer_ms_per_sample']:.3f} ms/sample,"
      f" {c['n_params']:,} params, {c['model_size_kb']:.0f} KB")
print(f"Saved -> {OUT/'B2_mlp_v7.json'}")
