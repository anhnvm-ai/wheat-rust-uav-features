"""Feature-engineering ablation on the ACTUAL V7 149-feature groups (cached features).
Incremental build-up matching the Methods section (no GLCM / Savitzky-Golay — those are not in V7).
Repeated 3x5 stratified CV, scaler in-fold, same models + macro-F1-weighted voting as the main eval.
"""
import numpy as np, pandas as pd, json, time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import lightgbm as lgb, xgboost as xgb

OUT=Path("/content/drive/MyDrive/Kaggle_Prepared/outputs/reviewer_experiments")
SEED=42; CLASSES=['Health','Other','Rust']; IDX={c:i for i,c in enumerate(CLASSES)}
df=pd.read_csv(OUT/"features_v7.csv")
allcols=[c for c in df.columns if c not in ('filename','label')]
y=df['label'].map(IDX).values

VEG=['ms_NDVI','ms_GNDVI','ms_NDRE','ms_SR','ms_MTCI','ms_CIgreen','ms_CIre','ms_PRI','ms_NDWI','ms_EVI','ms_SAVI','ms_NIR/Red','ms_NIR/Green','ms_RE/Red']
HS_SHAPE=['hs_d1_mean','hs_d1_std','hs_d1_max','hs_d1_maxpos','hs_REIP','hs_RE_slope','hs_NDVI','hs_band_corr','hs_AUC1','hs_AUC2','hs_AUC3']+[c for c in allcols if c.startswith('hs_CV')]
RGB_COLOR=['rgb_ExG','rgb_ExR','rgb_R/G','rgb_R/B','rgb_G/B','rgb_nd_RG','rgb_nd_GB','rgb_grad_mean','rgb_grad_std','rgb_grad_max','rgb_hue_mean','rgb_hue_std']
MS_CV=[c for c in allcols if c.startswith('ms_CV')]
ENGINEERED=set(VEG+HS_SHAPE+RGB_COLOR+MS_CV)
RAW=[c for c in allcols if c not in ENGINEERED]  # per-channel/band reflectance stats + sampled HS spectrum

steps=[("Raw reflectance statistics only",RAW),
       ("+ Vegetation indices (MS)",RAW+VEG),
       ("+ Red-edge / spectral-shape (HS)",RAW+VEG+HS_SHAPE),
       ("+ Colour indices, texture & CV (RGB+MS)  [Full V7]",RAW+VEG+HS_SHAPE+RGB_COLOR+MS_CV)]
print("group sizes: RAW",len(RAW),"VEG",len(VEG),"HS_SHAPE",len(HS_SHAPE),"RGB_COLOR",len(RGB_COLOR),"MS_CV",len(MS_CV),
      "| full",len(set(RAW+VEG+HS_SHAPE+RGB_COLOR+MS_CV)))

def models():
    return {'LightGBM':lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=20,max_depth=5,min_child_samples=15,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,verbose=-1,n_jobs=1),
            'XGBoost':xgb.XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,min_child_weight=8,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,eval_metric='mlogloss',verbosity=0,n_jobs=1),
            'RandomForest':RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
            'ExtraTrees':ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
            'SVM':SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=SEED)}

def cv_f1(cols,reps=3):
    X=np.nan_to_num(df[cols].values.astype(np.float64),nan=0,posinf=0,neginf=0); ff=[]
    for r in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=SEED+r)
        for tr,va in skf.split(X,y):
            sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[va]); mf1,mp={},{}
            for nm,md in models().items():
                md.fit(a,y[tr]); p=md.predict_proba(b); mf1[nm]=f1_score(y[va],p.argmax(1),average='macro'); mp[nm]=p
            ens=sum(mf1[n]*mp[n] for n in mf1)/sum(mf1.values()); ff.append(f1_score(y[va],ens.argmax(1),average='macro'))
    return float(np.mean(ff)),float(np.std(ff))

res=[]; t0=time.time(); prev=None
for name,cols in steps:
    m,s=cv_f1(cols); d=None if prev is None else (m-prev)*100
    res.append({'step':name,'n_features':len(cols),'macroF1':m,'std':s,'delta_pp':d})
    print(f"  {name:52s} n={len(cols):3d}  F1={m*100:.2f}% +/-{s*100:.2f}  d={'' if d is None else f'{d:+.2f}pp'}  ({time.time()-t0:.0f}s)")
    prev=m
json.dump(res,open(OUT/"V7_feature_ablation.json",'w'),indent=2)
print("Saved V7_feature_ablation.json")
