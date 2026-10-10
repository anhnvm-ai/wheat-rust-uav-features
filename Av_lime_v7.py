"""LIME fidelity analysis on the V7 149-feature ensemble (cached features).
Addresses Reviewer-2 critique (old mean local R^2 = 0.348, per-class weights unreliable):
reports V7's local-surrogate fidelity distribution honestly + per-class top LIME features.
Outputs: V7_lime.json + V7_lime_fidelity.png
"""
import numpy as np, pandas as pd, json, time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import lightgbm as lgb, xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT=Path("/content/drive/MyDrive/Kaggle_Prepared/outputs/reviewer_experiments")
SEED=42; np.random.seed(SEED); CLASSES=['Health','Other','Rust']; IDX={c:i for i,c in enumerate(CLASSES)}
df=pd.read_csv(OUT/"features_v7.csv"); feat=[c for c in df.columns if c not in ('filename','label')]
X=np.nan_to_num(df[feat].values.astype(np.float64),nan=0,posinf=0,neginf=0); y=df['label'].map(IDX).values

# macro-F1 weights from per-model CV (V7_permodel.json)
pm=json.load(open(OUT/"V7_permodel.json")); W={k:pm[k]['f1'] for k in pm}
mdls={'LightGBM':lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=20,max_depth=5,min_child_samples=15,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,verbose=-1,n_jobs=1),
      'XGBoost':xgb.XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,min_child_weight=8,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,eval_metric='mlogloss',verbosity=0,n_jobs=1),
      'RandomForest':RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
      'ExtraTrees':ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
      'SVM':SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=SEED)}
sc=StandardScaler().fit(X)
for m in mdls.values(): m.fit(sc.transform(X),y)
def predict_proba(Z):
    Zs=sc.transform(np.nan_to_num(Z,nan=0,posinf=0,neginf=0))
    p=sum(W[n]*mdls[n].predict_proba(Zs) for n in mdls)/sum(W.values()); return p

expl=LimeTabularExplainer(X,feature_names=feat,class_names=CLASSES,discretize_continuous=True,
                          mode='classification',random_state=SEED)
# sample 40 per class
idxs=[]
for c in range(3):
    ci=np.where(y==c)[0]; idxs+=list(np.random.RandomState(SEED+c).choice(ci,40,replace=False))
print(f"Explaining {len(idxs)} instances (LIME, num_samples=2000)..."); t0=time.time()
scores=[]; per_class_scores={c:[] for c in CLASSES}; per_class_feat={c:{} for c in CLASSES}
for j,i in enumerate(idxs):
    lab=int(predict_proba(X[i:i+1]).argmax())
    e=expl.explain_instance(X[i],predict_proba,num_features=10,num_samples=2000,labels=(lab,))
    sc_=e.score; scores.append(sc_); cn=CLASSES[lab]; per_class_scores[cn].append(sc_)
    for fn,w in e.as_list(label=lab):
        base=fn.split(' ')[0] if ('<' in fn or '>' in fn or '=' in fn) else fn
        # extract feature token (lime returns like "ms_NDVI > 0.40")
        tok=[t for t in fn.replace('=',' ').split() if t in feat]
        key=tok[0] if tok else fn
        per_class_feat[cn][key]=per_class_feat[cn].get(key,0)+abs(w)
    if (j+1)%30==0: print(f"  {j+1}/{len(idxs)}  ({time.time()-t0:.0f}s)")
scores=np.array(scores)
top_feat={c:sorted(per_class_feat[c].items(),key=lambda kv:-kv[1])[:6] for c in CLASSES}
res={'mean_local_R2':float(scores.mean()),'median_local_R2':float(np.median(scores)),'std':float(scores.std()),
     'per_class_mean_R2':{c:float(np.mean(per_class_scores[c])) for c in CLASSES},
     'n_instances':len(idxs),'top_features_by_class':{c:[[k,round(v,3)] for k,v in top_feat[c]] for c in CLASSES}}
json.dump(res,open(OUT/"V7_lime.json",'w'),indent=2)

plt.figure(figsize=(6,4)); plt.hist(scores,bins=20,color='#2c7fb8',edgecolor='w')
plt.axvline(scores.mean(),color='red',ls='--',label=f"mean R²={scores.mean():.2f}")
plt.xlabel("LIME local surrogate R² (fidelity)"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"V7_lime_fidelity.png",dpi=150); plt.close()
print("\n"+"="*60)
print(f"LIME mean local R² = {scores.mean():.3f} (median {np.median(scores):.3f}, sd {scores.std():.3f})")
print("per-class mean R²:",{c:round(np.mean(per_class_scores[c]),3) for c in CLASSES})
for c in CLASSES: print(f"  {c} top LIME feats:",[k for k,_ in top_feat[c]])
print("Saved V7_lime.json + V7_lime_fidelity.png"); print("="*60)
