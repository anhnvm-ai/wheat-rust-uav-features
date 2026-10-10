"""
r2_lib.py — shared evaluation harness for the SAT round-2 improvements.

Locked V7 protocol (identical to Av_consolidated_eval.py):
  - 5 base models: LightGBM, XGBoost, RandomForest, ExtraTrees, SVM
  - repeated 10x5 stratified CV, StandardScaler fit IN-FOLD
  - macro-F1-weighted soft voting ensemble
  - fixed class order ['Health','Other','Rust'] (NOT LabelEncoder), SEED=42

Every round-2 script imports from here so all new numbers stay protocol-consistent
with NUMBERS_LOCKED.md. The only thing that varies is the feature matrix X.
"""
import numpy as np, time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import lightgbm as lgb, xgboost as xgb

SEED = 42
CLASSES = ['Health', 'Other', 'Rust']
IDX = {c: i for i, c in enumerate(CLASSES)}
REPS = 10


def make_models():
    """The five locked base learners (n_jobs=1 for reproducibility)."""
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=20, max_depth=5,
                                       min_child_samples=15, subsample=0.7, colsample_bytree=0.7,
                                       reg_alpha=1.0, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, min_child_weight=8,
                                     subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
                                     random_state=SEED, eval_metric='mlogloss', verbosity=0, n_jobs=1),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=8,
                                               min_samples_leaf=4, max_features=0.5, random_state=SEED, n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_split=8,
                                           min_samples_leaf=4, max_features=0.5, random_state=SEED, n_jobs=1),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED),
    }


def one_cv(X, y, seed, sel_fn=None):
    """One 5-fold pass.
    sel_fn(Xtr, ytr) -> column index array, optional in-fold feature selector (no leakage).
    Returns (fold macro-F1 array, OOF prob matrix (n,3))."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    ff = []; oof = np.zeros((len(y), 3))
    for tr, va in skf.split(X, y):
        cols = sel_fn(X[tr], y[tr]) if sel_fn is not None else slice(None)
        Xtr, Xva = X[tr][:, cols], X[va][:, cols]
        sc = StandardScaler(); a = sc.fit_transform(Xtr); b = sc.transform(Xva)
        mf1, mp = {}, {}
        for nm, md in make_models().items():
            md.fit(a, y[tr]); p = md.predict_proba(b)
            mf1[nm] = f1_score(y[va], p.argmax(1), average='macro'); mp[nm] = p
        ens = sum(mf1[n] * mp[n] for n in mf1) / sum(mf1.values())
        ff.append(f1_score(y[va], ens.argmax(1), average='macro')); oof[va] = ens
    return np.array(ff), oof


def repeated_cv(X, y, reps=REPS, sel_fn=None, verbose=False, label=""):
    """Repeated reps x 5 CV. Returns dict with fold-F1 array, averaged OOF probs, summary stats."""
    t0 = time.time(); allf = []; oof_sum = np.zeros((len(y), 3))
    for r in range(reps):
        ff, oof = one_cv(X, y, SEED + r, sel_fn=sel_fn); allf += list(ff); oof_sum += oof
        if verbose:
            print(f"  [{label}] rep {r+1}/{reps}: {np.mean(ff)*100:.2f}%  ({time.time()-t0:.0f}s)")
    allf = np.array(allf); oof = oof_sum / reps
    rng = np.random.RandomState(SEED)
    boot = [rng.choice(allf, len(allf), replace=True).mean() for _ in range(10000)]
    ci = np.percentile(boot, [2.5, 97.5])
    return {'fold_f1': allf, 'oof': oof,
            'mean': float(allf.mean()), 'std': float(allf.std()),
            'ci95': [float(ci[0]), float(ci[1])],
            'oof_macroF1': float(f1_score(y, oof.argmax(1), average='macro')),
            'seconds': time.time() - t0}


def per_class_stats(y, pred):
    out = {}
    for c, i in IDX.items():
        out[c] = {'P': float(precision_score(y, pred, labels=[i], average='macro', zero_division=0)),
                  'R': float(recall_score(y, pred, labels=[i], average='macro')),
                  'F1': float(f1_score(y, pred, labels=[i], average='macro'))}
    return out


def load_features(csv_path):
    """Load a feature CSV (must have 'filename','label' cols). Returns X, y, filenames, feat_cols."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in ('filename', 'label')]
    X = np.nan_to_num(df[feat_cols].values.astype(np.float64), nan=0, posinf=0, neginf=0)
    y = df['label'].map(IDX).values
    fn = df['filename'].values if 'filename' in df.columns else np.arange(len(df))
    return X, y, fn, feat_cols
