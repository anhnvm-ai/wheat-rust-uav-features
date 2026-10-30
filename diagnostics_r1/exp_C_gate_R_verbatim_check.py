"""Gate R debug (30 Oct 2026): the original code of V7_assets.py (cv_f1, mdls), copied verbatim, rerun in the current
environment. Result: without HS 66.778, full 66.255, identical to exp_C_gate_R.json, so the gap to the published values
comes from the unrecorded June 2026 environment, not from the new script."""
# Gỡ lỗi cổng R: chạy NGUYÊN VĂN hàm cv_f1 và mdls() của code/V7_assets.py (dòng 27-47) cho "without HS" (cột 0..78)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import lightgbm as lgb, xgboost as xgb
SEED=42; CLASSES=['Health','Rust','Other']
df = pd.read_csv("/content/drive/MyDrive/Kaggle_Prepared/outputs/reviewer_experiments/features_v7.csv")
names = [c for c in df.columns if c not in ('filename','label')]
X = df[names].values; yl = df['label'].values; le=LabelEncoder().fit(CLASSES); y=le.transform(yl)
print('NaN in X:', int(np.isnan(X).sum()), X.dtype, X.shape)
def mdls():
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=20,max_depth=5,min_child_samples=15,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,verbose=-1,n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,min_child_weight=8,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,eval_metric='mlogloss',verbosity=0,n_jobs=1),
        'RandomForest': RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'SVM': SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=SEED),
    }
def cv_f1(cols, reps=3):
    f=[]
    for r in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=SEED+r)
        for tr,va in skf.split(X[:,cols],y):
            sc=StandardScaler(); a=sc.fit_transform(X[tr][:,cols]); b=sc.transform(X[va][:,cols])
            mf,mp={},{}
            for nm,md in mdls().items():
                md.fit(a,y[tr]); p=md.predict_proba(b); mf[nm]=f1_score(y[va],p.argmax(1),average='macro'); mp[nm]=p
            ens=sum(mf[n]*mp[n] for n in mf)/sum(mf.values()); f.append(f1_score(y[va],ens.argmax(1),average='macro'))
    return np.array(f)
for nm, cols in (('without HS', list(range(0, 79))), ('Full', list(range(0, 149)))):
    fa = cv_f1(cols); print(f"{nm}: verbatim original code now = {fa.mean()*100:.3f} ± {fa.std()*100:.2f}")
