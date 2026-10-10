"""V7 paper assets on the 150-feature pipeline: modality ablation, ensemble SHAP (+Spearman),
per-modality SHAP, calibration (ECE/Brier). Saves JSON + a SHAP figure."""
import numpy as np, pandas as pd, json, time
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import lightgbm as lgb, xgboost as xgb
import shap
import feat_v7_lib as L

ROOT=Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT=ROOT/"outputs/reviewer_experiments"
SEED=42; CLASSES=['Health','Rust','Other']
names=L.feature_names()
print("Building V7 train matrix...")
X,yl,ids=L.build(ROOT/"train"); le=LabelEncoder().fit(CLASSES); y=le.transform(yl)
print(f"  X={X.shape}, names={len(names)}")
assert X.shape[1]==len(names), f"feat/name mismatch {X.shape[1]} vs {len(names)}"
pd.DataFrame(X,columns=names).assign(filename=ids,label=yl).to_csv(OUT/"features_v7.csv",index=False)

# modality blocks (RGB 30 + MS 49 + HS 70 = 149)
nf=X.shape[1]; RGB=list(range(0,30)); MS=list(range(30,79)); HS=list(range(79,nf))
def mdls():
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=20,max_depth=5,min_child_samples=15,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,verbose=-1,n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,min_child_weight=8,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,eval_metric='mlogloss',verbosity=0,n_jobs=1),
        'RandomForest': RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'SVM': SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=SEED),
    }
def cv_f1(cols, reps=3):
    f=[]; oof=np.zeros((len(y),3)); ot=np.zeros(len(y),int)
    for r in range(reps):
        skf=StratifiedKFold(5,shuffle=True,random_state=SEED+r)
        for tr,va in skf.split(X[:,cols],y):
            sc=StandardScaler(); a=sc.fit_transform(X[tr][:,cols]); b=sc.transform(X[va][:,cols])
            mf,mp={},{}
            for nm,md in mdls().items():
                md.fit(a,y[tr]); p=md.predict_proba(b); mf[nm]=f1_score(y[va],p.argmax(1),average='macro'); mp[nm]=p
            ens=sum(mf[n]*mp[n] for n in mf)/sum(mf.values()); f.append(f1_score(y[va],ens.argmax(1),average='macro'))
            if r==0: oof[va]=ens; ot[va]=y[va]
    return np.array(f),oof,ot

print("\n(A) Modality ablation..."); t0=time.time()
abl={}
for nm,cols in [('Full(HS+MS+RGB)',RGB+MS+HS),('without HS',RGB+MS),('without MS',RGB+HS),('without RGB',MS+HS),('HS only',HS),('MS only',MS),('RGB only',RGB)]:
    fa,_,_=cv_f1(cols); abl[nm]={'f1':float(fa.mean()),'std':float(fa.std())}
    print(f"  {nm:18s} {fa.mean()*100:.2f}% +/- {fa.std()*100:.2f}  ({time.time()-t0:.0f}s)")
full=abl['Full(HS+MS+RGB)']['f1']
for k in abl: abl[k]['delta']=round((abl[k]['f1']-full)*100,2)

print("\n(B) Calibration (OOF)...")
_,oof,ot=cv_f1(RGB+MS+HS,reps=1)
def ece_brier(probs,yt,nb=10):
    conf=probs.max(1); pred=probs.argmax(1); acc=(pred==yt).astype(float)
    ece=0
    for i in range(nb):
        lo,hi=i/nb,(i+1)/nb; m=(conf>lo)&(conf<=hi)
        if m.sum()>0: ece+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    oh=np.eye(3)[yt]; brier=np.mean(np.sum((probs-oh)**2,1))
    return float(ece),float(brier),float(conf[acc==1].mean()),float(conf[acc==0].mean())
ece,brier,cc,cw=ece_brier(oof,ot)
print(f"  ECE={ece:.3f} Brier={brier:.3f} conf correct/wrong={cc:.3f}/{cw:.3f}")

print("\n(C) Ensemble SHAP (tree models) + per-modality...")
sc=StandardScaler(); Xs=sc.fit_transform(X)
tree_models={'LightGBM':mdls()['LightGBM'],'XGBoost':mdls()['XGBoost'],'RandomForest':mdls()['RandomForest'],'ExtraTrees':mdls()['ExtraTrees']}
imp={}
for nm,md in tree_models.items():
    md.fit(Xs,y); ex=shap.TreeExplainer(md); sv=ex.shap_values(Xs)
    if isinstance(sv,list): a=np.mean([np.abs(s).mean(0) for s in sv],0)
    else: a=np.abs(sv).reshape(sv.shape[0],sv.shape[1],-1).mean(0).mean(-1) if sv.ndim==3 else np.abs(sv).mean(0)
    a=np.asarray(a).ravel()[:150]; imp[nm]=a/(a.max()+1e-9)
ens_imp=np.mean(list(imp.values()),0)
order=np.argsort(ens_imp)[::-1]
top20=[(names[i],float(ens_imp[i])) for i in order[:20]]
# spearman across models
mods=list(imp); pair=[]
for i in range(len(mods)):
    for j in range(i+1,len(mods)):
        pair.append(spearmanr(imp[mods[i]],imp[mods[j]]).correlation)
mean_rho=float(np.nanmean(pair))
# per-modality contribution (sum of ensemble importance per block)
permod={'RGB':float(ens_imp[RGB].sum()),'MS':float(ens_imp[MS].sum()),'HS':float(ens_imp[HS].sum())}
tot=sum(permod.values()); permod_pct={k:round(v/tot*100,1) for k,v in permod.items()}
print(f"  top5: {[t[0] for t in top20[:5]]}")
print(f"  mean pairwise Spearman={mean_rho:.3f}  per-modality SHAP%={permod_pct}")

# figure
fig,ax=plt.subplots(1,2,figsize=(13,5))
t15=top20[:15][::-1]
ax[0].barh([t[0] for t in t15],[t[1] for t in t15],color='#2c7fb8')
ax[0].set_title('Top-15 features by ensemble SHAP (V7, 150 feats)'); ax[0].set_xlabel('normalized |SHAP|')
ax[1].bar(permod_pct.keys(),permod_pct.values(),color=['#d95f02','#1b9e77','#7570b3'])
ax[1].set_title('Per-modality SHAP contribution (%)'); ax[1].set_ylabel('%')
plt.tight_layout(); plt.savefig(OUT/"V7_shap.png",dpi=150); plt.savefig(ROOT/"V7_shap.png",dpi=150)
print(f"  saved V7_shap.png")

json.dump({'modality_ablation':abl,'calibration':{'ECE':ece,'Brier':brier,'conf_correct':cc,'conf_wrong':cw},
           'shap_top20':top20,'shap_mean_spearman':mean_rho,'per_modality_shap_pct':permod_pct,
           'full_cv_f1':full},open(OUT/"V7_assets.json",'w'),indent=2)
print("\nSaved V7_assets.json")
