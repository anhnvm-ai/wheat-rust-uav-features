"""Shared V7 feature library: 149-feature extractors (from wheat_disease_v7_raw_nopca.py, reformatted)
+ feature names + dataset builder. Used by V7_assets.py and submission scripts."""
import numpy as np
from pathlib import Path
import cv2, tifffile

def load_sample(sid, base):
    rgb=cv2.imread(str(base/"RGB"/f"{sid}.png"))
    if rgb is not None: rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    try:
        ms=tifffile.imread(str(base/"MS"/f"{sid}.tif"))
        if ms.ndim==3 and ms.shape[0]<ms.shape[2]: ms=np.moveaxis(ms,0,-1)
    except: ms=None
    try:
        hs=tifffile.imread(str(base/"HS"/f"{sid}.tif"))
        # NOTE (added during the revision, Oct 2026; code unchanged): HS cubes are stored as 32x32x125(126)
        # (rows x columns x bands), so shape[0] < shape[2] holds and this moveaxis turns the cube into 32x125x32.
        # extract_hs_raw_features then averages over columns and bands, i.e. its "mean spectrum" is a band-averaged
        # intensity profile over pixel rows (rows 11-18 after trimming); 47 of the 70 HS entries are constant.
        # Kept as is so that the features match the evaluated pipeline (Section 3.4.3 of the revised manuscript).
        if hs.ndim==3 and hs.shape[0]<hs.shape[2]: hs=np.moveaxis(hs,0,-1)
    except: hs=None
    return rgb,ms,hs

def extract_rgb_raw_features(rgb):
    if rgb is None: return np.zeros(30)
    f=[]; rgb=rgb.astype(np.float32)
    for c in range(3):
        ch=rgb[:,:,c]; f+=[np.mean(ch),np.std(ch),np.min(ch),np.max(ch),np.percentile(ch,25),np.percentile(ch,75)]
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    f.append((2*np.mean(g)-np.mean(r)-np.mean(b))/255.0)
    f.append((1.4*np.mean(r)-np.mean(g))/255.0)
    mr,mg,mb=np.mean(r),np.mean(g),np.mean(b)
    f+=[mr/(mg+1e-8),mr/(mb+1e-8),mg/(mb+1e-8),(mr-mg)/(mr+mg+1e-8),(mg-mb)/(mg+mb+1e-8)]
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
        f.append((mean_clean[ni]-mean_clean[ci])/(mean_clean[ni]+mean_clean[ci]+1e-8) if ci<nc and ni<nc else 0)
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
            bidx=min(ix+cs,hs.shape[-1]-1); bd=hs[:,:,bidx]; f.append(np.std(bd)/(np.mean(bd)+1e-8))
    else: f+=[0]*5
    while len(f)<70: f.append(0)
    return np.array(f[:70])

def feature_names():
    rgb=[]
    for c in ['R','G','B']:
        rgb+=[f'rgb_{c}_{s}' for s in ['mean','std','min','max','p25','p75']]
    rgb+=['rgb_ExG','rgb_ExR','rgb_R/G','rgb_R/B','rgb_G/B','rgb_nd_RG','rgb_nd_GB',
          'rgb_grad_mean','rgb_grad_std','rgb_grad_max','rgb_hue_mean','rgb_hue_std'][:12]
    rgb=rgb[:30]
    ms=[]
    for bn in ['blue','green','red','rededge','nir']:
        ms+=[f'ms_{bn}_{s}' for s in ['mean','std','min','max','p10','p90']]
    ms+=['ms_NDVI','ms_GNDVI','ms_NDRE','ms_SR','ms_MTCI','ms_CIgreen','ms_CIre','ms_PRI','ms_NDWI','ms_EVI','ms_SAVI','ms_NIR/Red','ms_NIR/Green','ms_RE/Red']
    ms+=[f'ms_CV_b{i}' for i in range(5)]
    ms=ms[:50]
    hs=[f'hs_spec_{i}' for i in range(20)]
    hs+=['hs_mean','hs_std','hs_min','hs_max','hs_range','hs_median','hs_peakpos','hs_valleypos',
         'hs_d1_mean','hs_d1_std','hs_d1_max','hs_d1_maxpos','hs_REIP','hs_RE_slope','hs_NDVI',
         'hs_band_corr','hs_AUC1','hs_AUC2','hs_AUC3','hs_CV_b0','hs_CV_b1','hs_CV_b2','hs_CV_b3','hs_CV_b4']
    hs+=[f'hs_pad_{i}' for i in range(70-len(hs))]
    hs=hs[:70]
    names=(rgb+ms+hs)
    return names

def build(base, is_val=False):
    base=Path(base)
    ids=sorted(p.stem for p in (base/"HS").glob("*.tif"))
    X=[]; y=[]
    for sid in ids:
        rgb,ms,hs=load_sample(sid,base)
        X.append(np.concatenate([extract_rgb_raw_features(rgb),extract_ms_raw_features(ms),extract_hs_raw_features(hs)]))
        if not is_val: y.append(sid.split('_')[0])
    X=np.nan_to_num(np.array(X),nan=0,posinf=0,neginf=0)
    return X, (np.array(y) if not is_val else None), ids
