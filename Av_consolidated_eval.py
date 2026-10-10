"""
Av_consolidated_eval.py  —  ONE careful, label-correct evaluation of the V7 (149-feat) pipeline.
Loads cached features (outputs/reviewer_experiments/features_v7.csv) so it is fast and reproduces
V7 exactly (same model configs, scaler in-fold, macro-F1-weighted soft voting).

Fixes the Rust/Other label swap in V7_honest_eval.json (LabelEncoder sorts classes).
Produces, on identical out-of-fold (OOF) probabilities averaged over repeated CV:
  1. Honest repeated 10x5 CV macro-F1 + bootstrap 95% CI
  2. Correct per-class P/R/F1 + confusion matrix  (A4)
  3. Cost-sensitive operating point: Health-boost sweep -> Health recall vs macro-F1 (A2)
  4. Calibration: raw multiclass ECE/Brier, and honest isotonic recalibration (A3) + reliability diagram
Outputs: outputs/reviewer_experiments/V7_consolidated_eval.json  and  V7_reliability_diagram.png
"""
import numpy as np, pandas as pd, json, time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT=Path("/content/drive/MyDrive/Kaggle_Prepared")
OUT=ROOT/"outputs/reviewer_experiments"
SEED=42; np.random.seed(SEED)
# Explicit, fixed label order (NOT LabelEncoder) to avoid the swap bug:
CLASSES=['Health','Other','Rust']; IDX={c:i for i,c in enumerate(CLASSES)}

df=pd.read_csv(OUT/"features_v7.csv")
feat_cols=[c for c in df.columns if c not in ('filename','label')]
X=np.nan_to_num(df[feat_cols].values.astype(np.float64),nan=0,posinf=0,neginf=0)
y=df['label'].map(IDX).values
print(f"X={X.shape}  y bincount={np.bincount(y)} (Health,Other,Rust)")

def models():
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=20,max_depth=5,min_child_samples=15,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,verbose=-1,n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,min_child_weight=8,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,eval_metric='mlogloss',verbosity=0,n_jobs=1),
        'RandomForest': RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'SVM': SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=SEED),
    }

def one_cv(seed):
    """One 5-fold pass: returns fold macro-F1 list and OOF probability matrix (n,3)."""
    skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    ff=[]; oof=np.zeros((len(y),3))
    for tr,va in skf.split(X,y):
        sc=StandardScaler(); a=sc.fit_transform(X[tr]); b=sc.transform(X[va])
        mf1,mp={},{}
        for nm,md in models().items():
            md.fit(a,y[tr]); p=md.predict_proba(b)
            mf1[nm]=f1_score(y[va],p.argmax(1),average='macro'); mp[nm]=p
        ens=sum(mf1[n]*mp[n] for n in mf1)/sum(mf1.values())
        ff.append(f1_score(y[va],ens.argmax(1),average='macro')); oof[va]=ens
    return np.array(ff), oof

REPS=10
print(f"\nRunning repeated {REPS}x5 CV (scaler in-fold, macro-F1-weighted voting)...")
t0=time.time(); allf=[]; oof_sum=np.zeros((len(y),3))
for r in range(REPS):
    ff,oof=one_cv(SEED+r); allf+=list(ff); oof_sum+=oof
    print(f"  rep {r+1}/{REPS}: {np.mean(ff)*100:.2f}%  ({time.time()-t0:.0f}s)")
allf=np.array(allf); oof=oof_sum/REPS            # averaged OOF probs (stable, still out-of-fold)
rng=np.random.RandomState(SEED)
boot=[rng.choice(allf,len(allf),replace=True).mean() for _ in range(10000)]
ci=np.percentile(boot,[2.5,97.5])
pred=oof.argmax(1)

# ---- (2/A4) correct per-class + confusion matrix ----
cm=confusion_matrix(y,pred).tolist()
perclass={}
for c,i in IDX.items():
    perclass[c]={'P':float(precision_score(y,pred,labels=[i],average='macro',zero_division=0)),
                 'R':float(recall_score(y,pred,labels=[i],average='macro')),
                 'F1':float(f1_score(y,pred,labels=[i],average='macro'))}
macro_oof=float(f1_score(y,pred,average='macro'))

# ---- (3/A2) cost-sensitive Health-boost operating points ----
def metrics_at(alpha):
    p=oof.copy(); p[:,IDX['Health']]*=alpha; p=p/p.sum(1,keepdims=True)
    pr=p.argmax(1)
    return float(f1_score(y,pr,average='macro')), float(recall_score(y,pr,labels=[IDX['Health']],average='macro'))
sweep={f"{a:.2f}":dict(zip(['macroF1','health_recall'],metrics_at(a))) for a in np.arange(1.0,2.01,0.1)}
# choose operating point: smallest alpha reaching Health recall >= 0.47, else best macroF1
chosen=None
for a in np.arange(1.0,2.01,0.05):
    mf,hr=metrics_at(a)
    if hr>=0.47: chosen={'alpha':round(float(a),3),'macroF1':mf,'health_recall':hr}; break
if chosen is None:
    a=max(np.arange(1.0,2.01,0.05),key=lambda a:metrics_at(a)[0]); mf,hr=metrics_at(a)
    chosen={'alpha':round(float(a),3),'macroF1':mf,'health_recall':hr}

# ---- (4/A3) calibration: raw + honest isotonic (split-and-swap, no leakage) ----
def ece(probs,labels,bins=10):
    conf=probs.max(1); predc=probs.argmax(1); acc=(predc==labels).astype(float)
    e=0.0
    for lo in np.linspace(0,1,bins+1)[:-1]:
        hi=lo+1.0/bins; m=(conf>lo)&(conf<=hi)
        if m.sum()>0: e+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)
def brier(probs,labels):
    oh=np.eye(3)[labels]; return float(np.mean(np.sum((probs-oh)**2,1)))
raw_ece, raw_brier = ece(oof,y), brier(oof,y)
# honest isotonic: fit on half A (one-vs-rest per class), apply to half B; both directions
sk=StratifiedKFold(2,shuffle=True,random_state=SEED)
cal=np.zeros_like(oof)
for fitidx,appidx in sk.split(oof,y):
    cc=np.zeros((len(appidx),3))
    for k in range(3):
        ir=IsotonicRegression(out_of_bounds='clip').fit(oof[fitidx,k],(y[fitidx]==k).astype(float))
        cc[:,k]=ir.predict(oof[appidx,k])
    cc=np.clip(cc,1e-6,None); cc=cc/cc.sum(1,keepdims=True); cal[appidx]=cc
cal_ece, cal_brier = ece(cal,y), brier(cal,y)
conf=oof.max(1); pc=(pred==y)
calib={'raw_ECE':raw_ece,'raw_Brier':raw_brier,'isotonic_ECE':cal_ece,'isotonic_Brier':cal_brier,
       'conf_correct':float(conf[pc].mean()),'conf_wrong':float(conf[~pc].mean())}

# reliability diagram (raw vs isotonic)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for axi,(P,nm,e) in zip(ax,[(oof,'Raw ensemble',raw_ece),(cal,'Isotonic-calibrated',cal_ece)]):
    conf=P.max(1); predc=P.argmax(1); accc=(predc==y).astype(float)
    bins=np.linspace(0,1,11); xs=[];ys=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.sum()>0: xs.append((lo+hi)/2); ys.append(accc[m].mean())
    axi.plot([0,1],[0,1],'k--',lw=1); axi.plot(xs,ys,'o-',color='#2c7fb8')
    axi.set_title(f"{nm}\nECE={e:.3f}"); axi.set_xlabel("Confidence"); axi.set_ylabel("Accuracy")
    axi.set_xlim(0,1); axi.set_ylim(0,1)
plt.tight_layout(); plt.savefig(OUT/"V7_reliability_diagram.png",dpi=150); plt.close()

res={'pipeline':'V7_149feat','protocol':f'repeated {REPS}x5 stratified CV, scaler in-fold, macro-F1-weighted soft voting',
     'class_order':CLASSES,
     'cv_macroF1_mean':float(allf.mean()),'cv_macroF1_std':float(allf.std()),'ci95':[float(ci[0]),float(ci[1])],
     'oof_macroF1':macro_oof,'per_class':perclass,'confusion_matrix':cm,'cm_rows_cols':CLASSES,
     'health_recall_raw':perclass['Health']['R'],
     'cost_sensitive_sweep':sweep,'chosen_operating_point':chosen,
     'calibration':calib,'n_features':X.shape[1]}
json.dump(res,open(OUT/"V7_consolidated_eval.json",'w'),indent=2)

print("\n"+"="*70)
print(f"V7 honest CV macro-F1 = {allf.mean()*100:.2f}% +/- {allf.std()*100:.2f}  CI[{ci[0]*100:.1f},{ci[1]*100:.1f}]")
print("CORRECT per-class F1:", {c:round(perclass[c]['F1'],3) for c in CLASSES})
print("Health recall (raw):", round(perclass['Health']['R'],3))
print("Confusion matrix (rows=true, cols=pred)", CLASSES);
for r_ in cm: print("   ",r_)
print(f"\nCost-sensitive operating point: alpha={chosen['alpha']} -> Health recall {chosen['health_recall']*100:.1f}%, macro-F1 {chosen['macroF1']*100:.1f}%")
print(f"Calibration RAW: ECE {raw_ece:.3f} Brier {raw_brier:.3f}  |  ISOTONIC: ECE {cal_ece:.3f} Brier {cal_brier:.3f}")
print("Saved V7_consolidated_eval.json + V7_reliability_diagram.png")
print("="*70)
