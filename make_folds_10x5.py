"""Write the fixed cross-validation folds of the manuscript to folds_10x5.csv (D7, R4.Q2.2).

Only splits labels with StratifiedKFold; no model is trained. Writes folds_10x5.csv next to this script.

Folds are those of r2_lib.repeated_cv / one_cv: repetition r = 0..9 uses
StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + r) on the 600 training tiles
sorted by file name, with labels Health=0, Other=1, Rust=2. The fair deep baseline
(B1_dl_baseline.py, diagnostics_r1/diag_dl_earlystop.py, diag_dl_protoC.py) uses repetition 0.
Output (long format, 6,000 rows): sample_id, label, repetition, fold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")    # data root, as in every script (README, Paths)
OUT = Path(__file__).resolve().parent / 'folds_10x5.csv'
LABELS = {'Health': 0, 'Other': 1, 'Rust': 2}

df = pd.read_csv(ROOT / 'outputs/reviewer_experiments/features_v7.csv', usecols=['filename', 'label'])
assert df['filename'].is_monotonic_increasing          # same order as r2_lib.load_features
y = df['label'].map(LABELS).to_numpy()
rows = []
for r in range(10):
    skf = StratifiedKFold(5, shuffle=True, random_state=42 + r)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        rows.append(pd.DataFrame({'sample_id': df['filename'].iloc[va].to_numpy(),
                                  'label': df['label'].iloc[va].to_numpy(),
                                  'repetition': r, 'fold': k}))
out = pd.concat(rows).sort_values(['repetition', 'sample_id']).reset_index(drop=True)
out.to_csv(OUT, index=False)

# checks
c = out.groupby(['repetition', 'fold', 'label']).size()
assert len(out) == 6000 and c.min() == 40 and c.max() == 40
assert (out.groupby(['repetition', 'sample_id']).size() == 1).all()   # each tile once per repetition
hs_dir = ROOT / 'train/HS'
if hs_dir.is_dir():                                     # fold construction of the deep-baseline scripts
    stems = sorted(p.stem for p in hs_dir.glob('*.tif'))
    y_dl = np.array([LABELS[s.split('_')[0]] for s in stems])
    rep0 = out[out.repetition == 0].set_index('sample_id')['fold']
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_dl)), y_dl)):
        assert (rep0.loc[[stems[i] for i in va]] == k).all()
    print('repetition 0 matches the deep-baseline folds')
print(f'{OUT}: {len(out)} rows; tiles per repetition x fold x class: {c.min()}-{c.max()}')
