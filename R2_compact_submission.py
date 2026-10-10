"""
R2_compact_submission.py — generate the Compact-30 competition submission and measure how
closely it matches the competition-validated full-149 submission (private 0.790).

The ICPR-2026 leaderboard is closed and no ground-truth test labels ship locally, so a *new*
private score cannot be computed. Instead we report an honest proxy: the agreement between the
Compact-30 predictions and the locked 149-feature submission (md5 4bba408f, private 0.790), plus
the predicted Rust count (the private score tracks the rust fraction — see class-sensitivity table).
High agreement + matching rust count => Compact-30 would behave comparably on the leaderboard.

Replicates the EXACT V7 submission protocol (V7_honest_eval.py): StandardScaler fit on full train,
per-model weights = 5-fold CV accuracy, weighted soft-vote. Train features from the cached
features_v7.csv; val features extracted fresh with feat_v7_lib.
Output: outputs/.../submission_compact30.csv, V7_compact_submission.json
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
import feat_v7_lib as F
import r2_lib as L

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
CLASSES = ['Health', 'Rust', 'Other']                # exact V7 LabelEncoder input (sorts internally)
le = LabelEncoder().fit(CLASSES)
names = F.feature_names()

# train from cache (col order == feature_names), val extracted fresh
df = pd.read_csv(OUT / "features_v7.csv")
assert list(df.columns[:149]) == names, "feature column order mismatch"
Xtr = np.nan_to_num(df[names].values.astype(np.float64)); ytr = le.transform(df['label'].values)
print(f"Train {Xtr.shape}; extracting val features (300 tiles)...")
Xva, _, val_ids = F.build(ROOT / "val", is_val=True)
Xva = np.nan_to_num(Xva.astype(np.float64))
print(f"Val {Xva.shape}")

top30 = json.load(open(OUT / "V7_compact.json"))['global_top30']
idx30 = [names.index(n) for n in top30]


def gen_submission(cols):
    """Exact V7 submission protocol restricted to `cols`."""
    a = Xtr[:, cols]; b = Xva[:, cols]
    sc = StandardScaler(); As = sc.fit_transform(a); Bs = np.nan_to_num(sc.transform(b))
    cvk = StratifiedKFold(5, shuffle=True, random_state=L.SEED)
    w = {};
    for nm, md in L.make_models().items():
        w[nm] = accuracy_score(ytr, cross_val_predict(md, As, ytr, cv=cvk))
    vp = np.zeros((len(val_ids), 3))
    for nm, md in L.make_models().items():
        md.fit(As, ytr); vp += w[nm] * md.predict_proba(Bs)
    vp /= sum(w.values())
    return le.inverse_transform(vp.argmax(1)), vp, w


print("\n[full-149] regenerating (sanity vs locked 0.790 submission)...")
pred_full, _, w_full = gen_submission(list(range(149)))
print("[compact-30] generating...")
pred_c30, vp_c30, w_c30 = gen_submission(idx30)

# save compact submission
pd.DataFrame({'Id': [f"{s}.tif" for s in val_ids], 'Category': pred_c30}).to_csv(
    OUT / "submission_compact30.csv", index=False)

# agreement analysis
ref = pd.read_csv(ROOT / "submission_V7_repro.csv")          # locked, private 0.790
ref_map = dict(zip(ref['Id'], ref['Category']))
ids_tif = [f"{s}.tif" for s in val_ids]
ref_pred = np.array([ref_map[i] for i in ids_tif])
agree_full_vs_ref = float((pred_full == ref_pred).mean())
agree_c30_vs_ref = float((pred_c30 == ref_pred).mean())
agree_c30_vs_full = float((pred_c30 == pred_full).mean())


def dist(p): return {c: int((p == c).sum()) for c in ['Health', 'Other', 'Rust']}


res = {'item': 'compact30_submission',
       'note': 'leaderboard closed / no local test labels -> agreement proxy, not a new private score',
       'reference_submission': 'submission_V7_repro.csv (private 0.790, public 0.653, md5 4bba408f)',
       'full149_repro_agreement_vs_ref': agree_full_vs_ref,
       'compact30_agreement_vs_ref(0.790)': agree_c30_vs_ref,
       'compact30_agreement_vs_full149_repro': agree_c30_vs_full,
       'dist_reference_0790': dist(ref_pred),
       'dist_full149_repro': dist(pred_full),
       'dist_compact30': dist(pred_c30),
       'rust_count_reference': int((ref_pred == 'Rust').sum()),
       'rust_count_compact30': int((pred_c30 == 'Rust').sum())}
json.dump(res, open(OUT / "V7_compact_submission.json", 'w'), indent=2)

print("\n" + "=" * 70)
print(f"full-149 repro vs locked 0.790 submission : {agree_full_vs_ref*100:.1f}% agreement (sanity)")
print(f"compact-30   vs locked 0.790 submission   : {agree_c30_vs_ref*100:.1f}% agreement")
print(f"compact-30   vs full-149 repro            : {agree_c30_vs_full*100:.1f}% agreement")
print(f"Rust count  ref(0.790)={res['rust_count_reference']}  compact30={res['rust_count_compact30']}  (private score tracks rust fraction)")
print(f"dist ref={res['dist_reference_0790']}  compact30={res['dist_compact30']}")
print("Saved submission_compact30.csv + V7_compact_submission.json")
print("=" * 70)
