"""Diagnostic (scratchpad only): optimism of the fair-DL baseline (B1_dl_baseline.py).
B1 picks the best epoch by macro-F1 on the *validation fold* and reports that same score.
Protocol A = B1 as is; Protocol B = early stopping on an inner 15% split of the training fold,
then a single evaluation on the untouched validation fold. Same architecture, data, folds (seed 42),
3 training seeds. Also records params, train time, per-class OOF metrics."""
import sys, json, time, numpy as np, rasterio
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1]); DEV = 'cuda'
label_map = {'Health': 0, 'Other': 1, 'Rust': 2}

def load_split(split):
    HS, MS, RGB = ROOT/split/"HS", ROOT/split/"MS", ROOT/split/"RGB"
    stems = sorted(p.stem for p in HS.glob("*.tif")); hs, ms, rgb, y = [], [], [], []
    for s in stems:
        with rasterio.open(HS/f"{s}.tif") as r: h = r.read().astype(np.float32)[10:111]
        with rasterio.open(MS/f"{s}.tif") as r: m = r.read().astype(np.float32)
        im = np.array(Image.open(RGB/f"{s}.png").convert('RGB')).astype(np.float32)/255.0
        hs.append(h); ms.append(m); rgb.append(im.transpose(2, 0, 1)); y.append(label_map[s.split('_')[0]])
    return np.stack(hs), np.stack(ms), np.stack(rgb), np.array(y)

HStr, MStr, RGBtr, ytr = load_split("train")

class Enc(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(96, cout)
    def forward(s, x): return torch.relu(s.fc(s.net(x).flatten(1)))

class MMNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.hs = Enc(101, 64); s.ms = Enc(5, 32); s.rgb = Enc(3, 32)
        s.head = nn.Sequential(nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96, 3))
    def forward(s, h, m, r): return s.head(torch.cat([s.hs(h), s.ms(m), s.rgb(r)], 1))

def T(a): return torch.tensor(a, dtype=torch.float32)

def augment(h, m, r):
    if np.random.rand() < 0.5: h, m, r = [torch.flip(x, [-1]) for x in (h, m, r)]
    if np.random.rand() < 0.5: h, m, r = [torch.flip(x, [-2]) for x in (h, m, r)]
    return h, m, r

torch.set_num_threads(2)
GPU = [torch.tensor(A, dtype=torch.float32, device=DEV) for A in (HStr, MStr, RGBtr)]   # all data on GPU once

def stats(fit_idx):
    fi = torch.as_tensor(fit_idx, device=DEV)
    return [(A[fi].mean((0, 2, 3), keepdim=True), A[fi].std((0, 2, 3), keepdim=True) + 1e-6) for A in GPU]

def view(idx, st):
    ii = torch.as_tensor(idx, device=DEV)
    return [(A[ii] - mu) / sd for A, (mu, sd) in zip(GPU, st)]

def predict(net, idx, fit_idx, st=None):
    net.eval()
    with torch.no_grad():
        return net(*view(idx, st or stats(fit_idx))).softmax(1).cpu().numpy()

def train(fit_idx, stop_idx, epochs=70, patience=12):
    """train on fit_idx; early-stop (keep best state) by macro-F1 on stop_idx"""
    st = stats(fit_idx); Xf = view(fit_idx, st); Xs = view(stop_idx, st)
    yf = torch.as_tensor(ytr[fit_idx], device=DEV); n = len(fit_idx)
    net = MMNet().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1); best, state, wait, best_ep = -1, None, 0, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=DEV)
        for b in range(0, n, 32):                                         # batch 32, shuffled (as B1)
            j = perm[b:b + 32]; h, m, r = augment(Xf[0][j], Xf[1][j], Xf[2][j]); opt.zero_grad()
            lossf(net(h, m, r), yf[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv = net(*Xs).argmax(1).cpu().numpy()
        f = f1_score(ytr[stop_idx], pv, average='macro')
        if f > best: best, state, wait, best_ep = f, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            wait += 1
            if wait >= patience: break
    net.load_state_dict(state); return net, best, best_ep

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(HStr, ytr))
res = {'A_bestepoch_on_valfold': [], 'B_inner_earlystop': []}; oofB = {}
n_params = sum(p.numel() for p in MMNet().parameters()); t_all = time.time(); times = []
for seed in (42, 43, 44):
    torch.manual_seed(seed); np.random.seed(seed); oof = np.zeros((len(ytr), 3))
    for k, (tr, va) in enumerate(folds):
        _, fA, _ = train(tr, va)                                            # A: B1 as published
        res['A_bestepoch_on_valfold'].append(fA)
        fit, stop = train_test_split(tr, test_size=0.15, stratify=ytr[tr], random_state=seed)
        t0 = time.time(); net, _, ep = train(fit, stop); times.append(time.time() - t0)
        p = predict(net, va, fit); oof[va] = p                              # B: proper
        res['B_inner_earlystop'].append(f1_score(ytr[va], p.argmax(1), average='macro'))
        print(f"seed {seed} fold {k+1}: A {fA*100:.1f}  B {res['B_inner_earlystop'][-1]*100:.1f}  (B best ep {ep})", flush=True)
    oofB[seed] = oof
P = sum(oofB.values()) / len(oofB); pred = P.argmax(1); cm = confusion_matrix(ytr, pred, labels=[0, 1, 2])
out = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'per_fold': [float(x) for x in v]} for k, v in res.items()}
out.update({'n_params': int(n_params), 'B_train_seconds_per_fold_mean': float(np.mean(times)),
            'B_seed_avg_oof_macroF1': float(f1_score(ytr, pred, average='macro')),
            'B_seed_avg_cm_rows_true_HOR': cm.tolist(), 'gpu': torch.cuda.get_device_name(0),
            'total_seconds': time.time() - t_all})
json.dump(out, open(OUT / "diag_dl_earlystop.json", "w"), indent=2)
print(json.dumps({k: (v['mean'] if isinstance(v, dict) and 'mean' in v else v) for k, v in out.items()}, indent=1))
