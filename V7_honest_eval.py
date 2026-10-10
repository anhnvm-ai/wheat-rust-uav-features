"""
V7 honest evaluation + reproduction.
- Reuses V7's exact 150-feature extractors (RGB30+MS50+HS70).
- (1) Honest CV: repeated 10x5 stratified, StandardScaler INSIDE folds, MACRO-F1 + bootstrap CI
      (fixes V7's full-data scaling leak + accuracy-instead-of-F1 metric).
- (2) V7-original CV for contrast (scaler on full train, accuracy).
- (3) Rebuild val features, train on all 600, regenerate submission, compare to submission_v7.csv.
"""
import os, numpy as np, pandas as pd, time, json
from pathlib import Path
import cv2, tifffile
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import lightgbm as lgb, xgboost as xgb

ROOT=Path("/content/drive/MyDrive/Kaggle_Prepared")
CLASSES=['Health','Rust','Other']; SEED=42; np.random.seed(SEED)

def load_sample(sid, base):
    rgb=cv2.imread(str(base/"RGB"/f"{sid}.png"))
    if rgb is not None: rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    try:
        ms=tifffile.imread(str(base/"MS"/f"{sid}.tif"))
        if ms.ndim==3 and ms.shape[0]<ms.shape[2]: ms=np.moveaxis(ms,0,-1)
    except: ms=None
    try:
        hs=tifffile.imread(str(base/"HS"/f"{sid}.tif"))
        if hs.ndim==3 and hs.shape[0]<hs.shape[2]: hs=np.moveaxis(hs,0,-1)
    except: hs=None
    return rgb,ms,hs

# ---- V7 extractors (verbatim) ----
def extract_rgb_raw_features(rgb):
    if rgb is None: return np.zeros(30)
    f=[]; rgb=rgb.astype(np.float32)
    for c in range(3):
        ch=rgb[:,:,c]; f+=[np.mean(ch),np.std(ch),np.min(ch),np.max(ch),np.percentile(ch,25),np.percentile(ch,75)]
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    f.append((2*np.mean(g)-np.mean(r)-np.mean(b))/255.0)
    f.append((1.4*np.mean(r)-np.mean(g))/255.0)
    mr,mg,mb=np.mean(r),np.mean(g),np.mean(b)
    f+= [mr/(mg+1e-8),mr/(mb+1e-8),mg/(mb+1e-8),(mr-mg)/(mr+mg+1e-8),(mg-mb)/(mg+mb+1e-8)]
    gray=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    gm=np.sqrt(gx**2+gy**2); f+=[np.mean(gm),np.std(gm),np.max(gm)]
    hsv=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
    f+=[np.mean(hsv[:,:,0]),np.std(hsv[:,:,0]),np.mean(hsv[:,:,1]),np.std(hsv[:,:,1])]
    return np.array(f[:30])

def extract_ms_raw_features(ms):
    if ms is None: return np.zeros(50)
    f=[]; ms=ms.astype(np.float32); nb=ms.shape[-1]
    for b in range(min(nb,5)):
        bd=ms[:,:,b]; f+=[np.mean(bd),np.std(bd),np.min(bd),np.max(bd),np.percentile(bd,10),np.percentile(bd,90)]
    if nb>=5:
        blue,green,red,re,nir=[np.mean(ms[:,:,i]) for i in range(5)]
        ndvi=(nir-red)/(nir+red+1e-8); gndvi=(nir-green)/(nir+green+1e-8); ndre=(nir-re)/(nir+re+1e-8); sr=nir/(red+1e-8)
        mtci=(nir-re)/(re-red+1e-8); ci_green=nir/(green+1e-8)-1; ci_re=nir/(re+1e-8)-1
        pri=(green-red)/(green+red+1e-8); ndwi=(green-nir)/(green+nir+1e-8)
        evi=2.5*(nir-red)/(nir+6*red-7.5*blue+1+1e-8); savi=((nir-red)/(nir+red+0.5))*1.5
        f+=[ndvi,gndvi,ndre,sr,mtci,ci_green,ci_re,pri,ndwi,evi,savi,nir/(red+1e-8),nir/(green+1e-8),re/(red+1e-8)]
        for b in range(5):
            bd=ms[:,:,b]; f.append(np.std(bd)/(np.mean(bd)+1e-8))
    else: f+=[0]*19
    return np.array(f[:50])

def extract_hs_raw_features(hs):
    if hs is None: return np.zeros(70)
    f=[]; hs=hs.astype(np.float32); nb=hs.shape[-1]
    mean_spectrum=np.mean(hs,axis=(0,1)); std_spectrum=np.std(hs,axis=(0,1))
    if nb>24: cs,ce=10,nb-14; mean_clean=mean_spectrum[cs:ce]; std_clean=std_spectrum[cs:ce]
    else: cs=0; mean_clean=mean_spectrum; std_clean=std_spectrum
    nc=len(mean_clean)
    if nc>=20: idx=np.linspace(0,nc-1,20).astype(int); f+=list(mean_clean[idx]/5000.0)
    else: f+=list(mean_clean/5000.0)+[0]*(20-nc)
    f+=[np.mean(mean_clean)/5000.0,np.std(mean_clean)/5000.0,np.min(mean_clean)/5000.0,np.max(mean_clean)/5000.0,(np.max(mean_clean)-np.min(mean_clean))/5000.0,np.median(mean_clean)/5000.0]
    if nc>1: f+=[np.argmax(mean_clean)/nc,np.argmin(mean_clean)/nc]
    else: f+=[0,0]
    if nc>2:
        d=np.gradient(mean_clean); f+=[np.mean(d)/5000.0,np.std(d)/5000.0,np.max(d)/5000.0,np.argmax(d)/nc]
    else: f+=[0,0,0,0]
    if nc>=50:
        rs,re_=int(nc*0.35),int(nc*0.50); rr=mean_clean[rs:re_]
        if len(rr)>1:
            rd=np.gradient(rr); f+=[(rs+np.argmax(rd))/nc,np.max(rd)/5000.0]
        else: f+=[0,0]
        ci,ni=int(nc*0.30),int(nc*0.65)
        if ci<nc and ni<nc:
            f.append((mean_clean[ni]-mean_clean[ci])/(mean_clean[ni]+mean_clean[ci]+1e-8))
        else: f.append(0)
    else: f+=[0,0,0]
    if nc>10:
        cors=[]
        for i in range(0,nc-10,10):
            c=np.corrcoef(mean_clean[i:i+5],mean_clean[i+5:i+10])[0,1]
            if not np.isnan(c): cors.append(c)
        f.append(np.mean(cors) if cors else 0)
    else: f.append(0)
    if nc>=6:
        tot=np.trapz(mean_clean); t1=np.trapz(mean_clean[:nc//3]); t2=np.trapz(mean_clean[nc//3:2*nc//3]); t3=np.trapz(mean_clean[2*nc//3:])
        f+=[t1/(tot+1e-8),t2/(tot+1e-8),t3/(tot+1e-8)]
    else: f+=[0,0,0]
    if nc>=10:
        si=np.linspace(0,nc-1,10).astype(int)
        for ix in si[:5]:
            bidx=min(ix+cs,hs.shape[-1]-1)
            bd=hs[:,:,bidx]; f.append(np.std(bd)/(np.mean(bd)+1e-8))
    else: f+=[0]*5
    while len(f)<70: f.append(0)
    return np.array(f[:70])

def feats(sid,base):
    rgb,ms,hs=load_sample(sid,base)
    return np.concatenate([extract_rgb_raw_features(rgb),extract_ms_raw_features(ms),extract_hs_raw_features(hs)])

def build(base,is_val):
    ids=sorted(p.stem for p in (base/"HS").glob("*.tif"))
    X=[]; y=[]
    for sid in ids:
        X.append(feats(sid,base))
        if not is_val: y.append(sid.split('_')[0])
    return np.nan_to_num(np.array(X),nan=0,posinf=0,neginf=0), (np.array(y) if not is_val else None), ids

print("Building train features (V7, 150-dim)..."); t0=time.time()
Xtr,ytr_lbl,tr_ids=build(ROOT/"train",False)
le=LabelEncoder().fit(CLASSES); ytr=le.transform(ytr_lbl)
print(f"  {Xtr.shape} in {time.time()-t0:.0f}s, classes={np.bincount(ytr)}")

def models():
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,num_leaves=20,max_depth=5,min_child_samples=15,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,verbose=-1,n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,min_child_weight=8,subsample=0.7,colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,random_state=SEED,eval_metric='mlogloss',verbosity=0,n_jobs=1),
        'RandomForest': RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200,max_depth=6,min_samples_split=8,min_samples_leaf=4,max_features=0.5,random_state=SEED,n_jobs=1),
        'SVM': SVC(C=1.0,kernel='rbf',gamma='scale',probability=True,random_state=SEED),
    }

# ---------- (1) HONEST CV: scaler in-fold, macro-F1, weighted-by-fold-F1 ----------
def honest_cv(seed):
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); ff=[]; oof=np.zeros((len(ytr),3)); ot=np.zeros(len(ytr),int)
    for tr,va in skf.split(Xtr,ytr):
        sc=StandardScaler(); a=sc.fit_transform(Xtr[tr]); b=sc.transform(Xtr[va])
        mf1,mp={},{}
        for nm,md in models().items():
            md.fit(a,ytr[tr]); p=md.predict_proba(b); mf1[nm]=f1_score(ytr[va],p.argmax(1),average='macro'); mp[nm]=p
        ens=sum(mf1[n]*mp[n] for n in mf1)/sum(mf1.values())
        ff.append(f1_score(ytr[va],ens.argmax(1),average='macro')); oof[va]=ens; ot[va]=ytr[va]
    return np.array(ff),oof,ot

print("\n(1) Honest repeated 10x5 CV (scaler in-fold, macro-F1)..."); t0=time.time()
allf=[]; last_oof=None; last_ot=None
for r in range(10):
    ff,oof,ot=honest_cv(SEED+r); allf+=list(ff); last_oof,last_ot=oof,ot
    print(f"  rep {r+1}/10: {np.mean(ff)*100:.2f}%  ({time.time()-t0:.0f}s)")
allf=np.array(allf); rng=np.random.RandomState(SEED)
boot=[rng.choice(allf,len(allf),replace=True).mean() for _ in range(10000)]; ci=np.percentile(boot,[2.5,97.5])
pred=last_oof.argmax(1)
pcf1={CLASSES[i]:f1_score(last_ot,pred,labels=[i],average='macro') for i in range(3)}
hr=recall_score(last_ot,pred,labels=[0],average='macro')

# ---------- (2) V7-original CV (full-data scaler, accuracy) ----------
print("\n(2) V7-original CV (scaler on full train, accuracy)...")
sc_full=StandardScaler().fit_transform(Xtr)
cv=StratifiedKFold(5,shuffle=True,random_state=SEED); acc7={}
for nm,md in models().items():
    yp=cross_val_predict(md,sc_full,ytr,cv=cv); acc7[nm]=accuracy_score(ytr,yp)
v7_acc=np.mean(list(acc7.values()))

# ---------- (3) Reproduce submission ----------
print("\n(3) Rebuilding val + regenerating submission...")
Xva,_,val_ids=build(ROOT/"val",True)
sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xva_s=sc.transform(Xva)
Xva_s=np.nan_to_num(Xva_s,nan=0,posinf=0,neginf=0)
w={}; cvk=StratifiedKFold(5,shuffle=True,random_state=SEED)
for nm,md in models().items():
    w[nm]=accuracy_score(ytr,cross_val_predict(md,Xtr_s,ytr,cv=cvk))
vp=np.zeros((len(val_ids),3))
for nm,md in models().items():
    md.fit(Xtr_s,ytr); vp+=w[nm]*md.predict_proba(Xva_s)
vp/=sum(w.values()); vpred=le.inverse_transform(vp.argmax(1))
sub=pd.DataFrame({'Id':[f"{s}.tif" for s in val_ids],'Category':vpred})
sub.to_csv(ROOT/"outputs/reviewer_experiments/submission_v7_repro.csv",index=False)
# compare to reference
ref=pd.read_csv(ROOT/"paper_extracted/submission_v7_ref.csv")
m=sub.merge(ref,on='Id',suffixes=('_repro','_ref'))
agree=(m['Category_repro']==m['Category_ref']).mean()

print("\n"+"="*64)
print(f"V7 HONEST CV macro-F1 = {allf.mean()*100:.2f}% +/- {allf.std()*100:.2f}  CI[{ci[0]*100:.1f},{ci[1]*100:.1f}]")
print(f"  per-class F1: "+", ".join(f"{k}={v*100:.1f}" for k,v in pcf1.items())+f"  | Health recall={hr*100:.1f}%")
print(f"V7-original CV (accuracy, leaky scaler) = {v7_acc*100:.2f}%  [inflated/optimistic]")
print(f"Reproduced submission agrees with submission_v7.csv on {agree*100:.1f}% of 300 val")
print(f"  repro dist: {dict(pd.Series(vpred).value_counts())}")
print("="*64)
json.dump({'honest_cv_macroF1':float(allf.mean()),'honest_cv_std':float(allf.std()),'ci95':[float(ci[0]),float(ci[1])],
           'per_class_f1':{k:float(v) for k,v in pcf1.items()},'health_recall':float(hr),
           'v7_original_cv_accuracy':float(v7_acc),'repro_agreement':float(agree),
           'n_features':int(Xtr.shape[1])}, open(ROOT/"outputs/reviewer_experiments/V7_honest_eval.json",'w'),indent=2)
print("Saved V7_honest_eval.json")
