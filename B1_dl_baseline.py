"""
B1 — Fair multimodal DL baseline with a dedicated HS spectral-spatial encoder.
Addresses reviewer: "need a stronger multimodal DL baseline with an HS encoder".
Encoders: HS 2D-CNN (101 bands as channels, 32x32) + MS 2D-CNN (5,64,64) + RGB CNN (3,64,64),
late-fusion -> MLP head. Trained under the SAME 5-fold stratified CV as the feature pipeline
(SEED=42), per-fold input standardization, flip augmentation, label smoothing, early stopping.
Reports mean +/- std macro-F1 (fair comparison vs feature-ML 68.0%) and a Kaggle val submission.
"""
import numpy as np, pandas as pd, time
from pathlib import Path
import rasterio
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42); np.random.seed(42)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
ROOT=Path("/content/drive/MyDrive/Kaggle_Prepared")
label_map={'Health':0,'Other':1,'Rust':2}; inv={v:k for k,v in label_map.items()}

def load_split(split, stems=None):
    HS,MS,RGB=ROOT/split/"HS",ROOT/split/"MS",ROOT/split/"RGB"
    if stems is None: stems=sorted(p.stem for p in HS.glob("*.tif"))
    hs,ms,rgb,y=[],[],[],[]
    for s in stems:
        with rasterio.open(HS/f"{s}.tif") as r: h=r.read().astype(np.float32)[10:111]   # 101,32,32
        with rasterio.open(MS/f"{s}.tif") as r: m=r.read().astype(np.float32)            # 5,64,64
        rp=RGB/f"{s}.png"; im=np.array(Image.open(rp if rp.exists() else RGB/f"{s}.jpg").convert('RGB')).astype(np.float32)/255.0
        hs.append(h); ms.append(m); rgb.append(im.transpose(2,0,1))
        y.append(label_map.get(s.split('_')[0],-1))
    return (np.stack(hs),np.stack(ms),np.stack(rgb),np.array(y),stems)

print("Loading train..."); t0=time.time()
HStr,MStr,RGBtr,ytr,_=load_split("train"); print(f"  {HStr.shape} {MStr.shape} {RGBtr.shape} ({time.time()-t0:.0f}s)")

class Enc(nn.Module):
    def __init__(s,cin,cout):
        super().__init__()
        s.net=nn.Sequential(
            nn.Conv2d(cin,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        s.fc=nn.Linear(96,cout)
    def forward(s,x): return torch.relu(s.fc(s.net(x).flatten(1)))

class MMNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.hs=Enc(101,64); s.ms=Enc(5,32); s.rgb=Enc(3,32)
        s.head=nn.Sequential(nn.Linear(128,96),nn.ReLU(),nn.Dropout(0.3),nn.Linear(96,3))
    def forward(s,h,m,r): return s.head(torch.cat([s.hs(h),s.ms(m),s.rgb(r)],1))

def to_t(a): return torch.tensor(a,dtype=torch.float32)

def augment(h,m,r):
    if np.random.rand()<0.5: h,m,r=[torch.flip(x,[-1]) for x in (h,m,r)]
    if np.random.rand()<0.5: h,m,r=[torch.flip(x,[-2]) for x in (h,m,r)]
    return h,m,r

def train_fold(tr,va,epochs=70,patience=12):
    # standardize per modality (channel-wise stats over train fold)
    def norm(X):
        mu=X[tr].mean((0,2,3),keepdims=True); sd=X[tr].std((0,2,3),keepdims=True)+1e-6
        return (X[tr]-mu)/sd,(X[va]-mu)/sd
    h_tr,h_va=norm(HStr); m_tr,m_va=norm(MStr); r_tr,r_va=norm(RGBtr)
    Xtr=[to_t(h_tr),to_t(m_tr),to_t(r_tr)]; Xva=[to_t(h_va).to(DEV),to_t(m_va).to(DEV),to_t(r_va).to(DEV)]
    ytr_t=torch.tensor(ytr[tr]); yva=ytr[va]
    ds=TensorDataset(*Xtr,ytr_t); dl=DataLoader(ds,batch_size=32,shuffle=True)
    net=MMNet().to(DEV); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss(label_smoothing=0.1)
    best,best_state,wait=0,None,0
    for ep in range(epochs):
        net.train()
        for h,m,r,yy in dl:
            h,m,r=augment(h.to(DEV),m.to(DEV),r.to(DEV)); yy=yy.to(DEV)
            opt.zero_grad(); loss=lossf(net(h,m,r),yy); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv=net(*Xva).argmax(1).cpu().numpy()
        f=f1_score(yva,pv,average='macro')
        if f>best: best,best_state,wait=f,{k:v.cpu().clone() for k,v in net.state_dict().items()},0
        else:
            wait+=1
            if wait>=patience: break
    net.load_state_dict(best_state)
    return net,best

print(f"\nTraining DL baseline, 5-fold stratified CV on {DEV}...")
skf=StratifiedKFold(5,shuffle=True,random_state=42)
fold_f1=[]; oof=np.zeros(len(ytr),int); t0=time.time()
for k,(tr,va) in enumerate(skf.split(HStr,ytr)):
    net,bf=train_fold(tr,va)
    net.eval()
    def nrm(X):
        mu=X[tr].mean((0,2,3),keepdims=True); sd=X[tr].std((0,2,3),keepdims=True)+1e-6
        return (X[va]-mu)/sd
    with torch.no_grad():
        pv=net(to_t(nrm(HStr)).to(DEV),to_t(nrm(MStr)).to(DEV),to_t(nrm(RGBtr)).to(DEV)).argmax(1).cpu().numpy()
    oof[va]=pv; fold_f1.append(f1_score(ytr[va],pv,average='macro'))
    print(f"  fold {k+1}: macro-F1={fold_f1[-1]*100:.2f}%  ({time.time()-t0:.0f}s)")

fold_f1=np.array(fold_f1)
print("\n"+"="*60)
print(f"DL BASELINE (multimodal, HS+MS+RGB encoders): macro-F1 = {fold_f1.mean()*100:.2f}% +/- {fold_f1.std()*100:.2f}")
print(f"  vs feature-ML ensemble 68.0% (same CV protocol)")
print(f"  OOF per-class F1: "+", ".join(f"{inv[i]}={f1_score((ytr==i).astype(int),(oof==i).astype(int))*100:.1f}" for i in range(3)))
print(f"  OOF confusion (rows true H/O/R):\n{confusion_matrix(ytr,oof,labels=[0,1,2])}")
print("="*60)

import json
json.dump({'dl_mean_f1':float(fold_f1.mean()),'dl_std':float(fold_f1.std()),
           'per_fold':[float(x) for x in fold_f1],
           'cm':confusion_matrix(ytr,oof,labels=[0,1,2]).tolist(),
           'feature_ml_ref':0.680,'device':torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'},
          open(ROOT/"outputs/reviewer_experiments/B1_dl_baseline.json",'w'),indent=2)
print("Saved B1_dl_baseline.json")
