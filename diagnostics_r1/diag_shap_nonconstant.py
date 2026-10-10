"""Agreement of the SHAP importances of the four tree-based members without the constant features
(revision, October 2026). V7_assets.py stored only the top-20 list and the mean pairwise Spearman
correlation over all 149 features (0.900), in which the 50 features that are constant over the 600 tiles
share one tied rank. This script repeats the SHAP step of V7_assets.py (same models, read from that file;
fitted on all 600 standardised tiles), checks that it reproduces the stored values, and then computes the
correlation over the non-constant features only. Usage: python3 diag_shap_nonconstant.py <out_dir>"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import ast, sys, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings("ignore")
import shap, lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1])
# model definitions taken verbatim from V7_assets.py (function mdls), so that the settings are identical
src = (ROOT / "code/V7_assets.py").read_text(encoding="utf-8")
fdef = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "mdls"][0]
ns = {'lgb': lgb, 'xgb': xgb, 'RandomForestClassifier': RandomForestClassifier,
      'ExtraTreesClassifier': ExtraTreesClassifier, 'SVC': SVC, 'SEED': 42}
exec(compile(ast.Module(body=[fdef], type_ignores=[]), "V7_assets.mdls", "exec"), ns)
mdls = ns['mdls']

df = pd.read_csv(ROOT / "outputs/reviewer_experiments/features_v7.csv")
names = [c for c in df.columns if c not in ('filename', 'label')]
X = df[names].to_numpy(); y = LabelEncoder().fit(['Health', 'Rust', 'Other']).transform(df['label'])
Xs = StandardScaler().fit_transform(X)

imp, raw, params = {}, {}, {}
for nm in ['LightGBM', 'XGBoost', 'RandomForest', 'ExtraTrees']:
    md = mdls()[nm]; md.fit(Xs, y); sv = shap.TreeExplainer(md).shap_values(Xs)
    if isinstance(sv, list): a = np.mean([np.abs(s).mean(0) for s in sv], 0)
    else: a = np.abs(sv).reshape(sv.shape[0], sv.shape[1], -1).mean(0).mean(-1) if sv.ndim == 3 else np.abs(sv).mean(0)
    a = np.asarray(a).ravel()[:150]; raw[nm] = a; imp[nm] = a / (a.max() + 1e-9)
    params[nm] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in md.get_params().items()}

def mean_rho(idx):
    rho = {f'{a}-{b}': float(spearmanr(imp[a][idx], imp[b][idx]).correlation) for a, b in combinations(imp, 2)}
    return float(np.mean(list(rho.values()))), rho

ens = np.mean(list(imp.values()), 0)
top20 = [(names[i], float(ens[i])) for i in np.argsort(ens)[::-1][:20]]
nonconst = np.where(X.std(0) > 0)[0]
all_mean, all_pairs = mean_rho(np.arange(X.shape[1]))
nc_mean, nc_pairs = mean_rho(nonconst)

stored = json.load(open(ROOT / "outputs/reviewer_experiments/V7_assets.json"))
check = {'mean_rho_all_rerun': all_mean, 'mean_rho_all_stored': stored['shap_mean_spearman'],
         'top20_names_equal': [t[0] for t in top20] == [t[0] for t in stored['shap_top20']],
         'top20_max_abs_diff': float(max(abs(a[1] - b[1]) for a, b in zip(top20, stored['shap_top20'])))}
check['reproduced'] = bool(abs(all_mean - stored['shap_mean_spearman']) < 1e-6 and check['top20_names_equal'])
out = {'n_features': int(X.shape[1]), 'n_constant': int(X.shape[1] - len(nonconst)), 'n_nonconstant': int(len(nonconst)),
       'constant_features': [names[i] for i in range(X.shape[1]) if i not in set(nonconst)],
       'mean_pairwise_spearman_all': all_mean, 'pairs_all': all_pairs,
       'mean_pairwise_spearman_nonconstant': nc_mean, 'pairs_nonconstant': nc_pairs,
       'reproduction_check': check}
# checks of the published SHAP statements against this run (Figure 3 and Section 4.5.1)
RGB, MS, HS = range(0, 30), range(30, 79), range(79, X.shape[1])
blk = {'RGB': float(ens[list(RGB)].sum()), 'MS': float(ens[list(MS)].sum()), 'HS': float(ens[list(HS)].sum())}
tot = sum(blk.values()); shares = {k: round(v / tot * 100, 1) for k, v in blk.items()}
order = list(np.argsort(ens)[::-1]); top15 = [names[i] for i in order[:15]]
fam = {'NIR/R': ['ms_NDVI', 'ms_SR', 'ms_SAVI', 'ms_NIR/Red'], 'NIR/G': ['ms_GNDVI', 'ms_CIgreen', 'ms_NDWI', 'ms_NIR/Green'],
       'NIR/RE': ['ms_NDRE', 'ms_CIre']}
in_fam = [f for f in top15 if any(f in v for v in fam.values())]
first_hs = next(r + 1 for r, i in enumerate(order) if names[i].startswith('hs_'))
def topk(m, k, idx): return set(np.array(idx)[np.argsort(imp[m][idx])[::-1][:k]].tolist())
overlap = {}
for k in (5, 10):
    for lab, idx in (('all', np.arange(X.shape[1])), ('nonconstant', nonconst)):
        sets = {m: topk(m, k, idx) for m in imp}
        overlap[f'top{k}_{lab}'] = {'common_to_all_four': sorted(names[i] for i in set.intersection(*sets.values())),
                                    'pairwise_jaccard': {f'{a}-{b}': len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in combinations(sets, 2)},
                                    'per_model': {m: [names[i] for i in sorted(sets[m], key=lambda i: -imp[m][i])] for m in sets}}
import sklearn, scipy, platform
out.update({'per_modality_shares_pct': shares, 'per_modality_shares_pct_stored': stored['per_modality_shap_pct'],
            'top15_rerun': top15, 'top15_stored': [t[0] for t in stored['shap_top20'][:15]],
            'top15_in_three_ratio_families': len(in_fam), 'first_hs_rank_rerun': first_hs,
            'first_hs_rank_stored': next(r + 1 for r, t in enumerate(stored['shap_top20']) if t[0].startswith('hs_')),
            'topk_overlap': overlap, 'feature_names': names, 'nonconstant_mask': [bool(i in set(nonconst)) for i in range(X.shape[1])],
            'mean_abs_shap_raw': {m: raw[m].tolist() for m in raw}, 'importance_normalised': {m: imp[m].tolist() for m in imp},
            'ensemble_importance': ens.tolist(), 'model_params': params,
            'training': 'four tree members refitted on all 600 tiles, StandardScaler on all tiles, features from features_v7.csv',
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                         'lightgbm': lgb.__version__, 'xgboost': xgb.__version__, 'shap': shap.__version__, 'scipy': scipy.__version__}})
json.dump(out, open(OUT / "diag_shap_nonconstant.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ['per_modality_shares_pct', 'per_modality_shares_pct_stored', 'top15_in_three_ratio_families',
                                       'first_hs_rank_rerun', 'first_hs_rank_stored', 'versions']}, indent=1))
print('pairs all        :', {k: round(v, 3) for k, v in all_pairs.items()})
print('pairs nonconstant:', {k: round(v, 3) for k, v in nc_pairs.items()})
for key in ['top5_nonconstant', 'top10_nonconstant']:
    print(key, 'common to all four:', overlap[key]['common_to_all_four'], '| Jaccard', {k: round(v, 2) for k, v in overlap[key]['pairwise_jaccard'].items()})
print(json.dumps({k: out[k] for k in ['n_constant', 'n_nonconstant', 'mean_pairwise_spearman_all',
                                       'mean_pairwise_spearman_nonconstant']}, indent=1))
print(json.dumps(check, indent=1))
