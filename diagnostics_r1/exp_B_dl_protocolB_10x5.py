"""Experiment B (final robustness check, pre-registered 19 Oct 2026; protocol in PREREG_final_robustness.md):
fair multimodal network, protocol B, on all 50 outer folds of the 10x5 protocol with three training seeds.

Architecture, data, normalisation, augmentation, optimiser and early-stopping rule are copied unchanged from
diag_dl_earlystop.py (protocol B): early stopping on Macro-F1 of a stratified 15% split of the training fold
(random_state = seed), at most 70 epochs, patience 12, best epoch restored; the outer fold is scored once.
Seeds are reset for every (repetition, fold, seed) fit; cuDNN runs in deterministic mode.

Primary fold score (author's choice, 19 Oct 2026): the mean of the three per-seed Macro-F1 values.
Secondary fold score: Macro-F1 of the class probabilities averaged over the three seeds.
Paired comparison with the ensemble (nested weights; per-fold scores from diag_nested_perfold.json) on the
same 50 folds, Nadeau-Bengio corrected paired t (J = 50, n_test/n_train = 1/4, 49 df).

Gates (see PREREG): F1 the folds equal folds_10x5.csv for all 50 folds (checked before any training);
F2 the repetition-0 mean of the primary score lies within 3 points of the published 65.2% (stop otherwise).

Usage:  python3 exp_B_dl_protocolB_10x5.py <out_dir> run 0          (repetitions to run, e.g. 0  or  1-9)
        python3 exp_B_dl_protocolB_10x5.py <out_dir> summarise
Every finished fit is appended to exp_B_seedlevel.jsonl, so an interrupted run resumes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, time, platform
import numpy as np, pandas as pd, rasterio
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy import stats
import torch, torch.nn as nn

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = Path(sys.argv[1]); MODE = sys.argv[2]
CKPT = OUT / "exp_B_seedlevel.jsonl"; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = (42, 43, 44); label_map = {'Health': 0, 'Other': 1, 'Rust': 2}
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


def stems_labels():
    stems = sorted(p.stem for p in (ROOT / "train" / "HS").glob("*.tif"))
    return stems, np.array([label_map[s.split('_')[0]] for s in stems])


def folds(y):
    return [(r, k, tr, va) for r in range(10)
            for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + r).split(np.zeros(len(y)), y))]


def gate_folds(stems, y):
    ref = pd.read_csv(ROOT / "code/folds_10x5.csv")
    bad = 0
    for r, k, tr, va in folds(y):
        want = set(ref[(ref.repetition == r) & (ref.fold == k)].sample_id)
        bad += want != {stems[i] for i in va}
    return bad == 0


# ---------------- copied unchanged from diag_dl_earlystop.py (protocol B) ----------------
def load_split(split):
    HS, MS, RGB = ROOT/split/"HS", ROOT/split/"MS", ROOT/split/"RGB"
    stems = sorted(p.stem for p in HS.glob("*.tif")); hs, ms, rgb, y = [], [], [], []
    for s in stems:
        with rasterio.open(HS/f"{s}.tif") as r: h = r.read().astype(np.float32)[10:111]
        with rasterio.open(MS/f"{s}.tif") as r: m = r.read().astype(np.float32)
        im = np.array(Image.open(RGB/f"{s}.png").convert('RGB')).astype(np.float32)/255.0
        hs.append(h); ms.append(m); rgb.append(im.transpose(2, 0, 1)); y.append(label_map[s.split('_')[0]])
    return np.stack(hs), np.stack(ms), np.stack(rgb), np.array(y)

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

def augment(h, m, r):
    if np.random.rand() < 0.5: h, m, r = [torch.flip(x, [-1]) for x in (h, m, r)]
    if np.random.rand() < 0.5: h, m, r = [torch.flip(x, [-2]) for x in (h, m, r)]
    return h, m, r
# ------------------------------------------------------------------------------------------


def run(reps):
    stems, ylab = stems_labels()
    assert gate_folds(stems, ylab), "gate F1 failed: DL folds differ from folds_10x5.csv"
    print("gate F1 passed: folds equal folds_10x5.csv for all 50 folds", flush=True)
    HStr, MStr, RGBtr, ytr = load_split("train"); assert (ytr == ylab).all()
    G = [torch.tensor(A, dtype=torch.float32, device=DEV) for A in (HStr, MStr, RGBtr)]

    def stats_(fit_idx):
        fi = torch.as_tensor(fit_idx, device=DEV)
        return [(A[fi].mean((0, 2, 3), keepdim=True), A[fi].std((0, 2, 3), keepdim=True) + 1e-6) for A in G]

    def view(idx, st):
        ii = torch.as_tensor(idx, device=DEV)
        return [(A[ii] - mu) / sd for A, (mu, sd) in zip(G, st)]

    def train(fit_idx, stop_idx, epochs=70, patience=12):
        st = stats_(fit_idx); Xf = view(fit_idx, st); Xs = view(stop_idx, st)
        yf = torch.as_tensor(ytr[fit_idx], device=DEV); n = len(fit_idx)
        net = MMNet().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.1); best, state, wait, best_ep = -1, None, 0, 0
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for b in range(0, n, 32):
                j = perm[b:b + 32]; h, m, r = augment(Xf[0][j], Xf[1][j], Xf[2][j]); opt.zero_grad()
                lossf(net(h, m, r), yf[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): pv = net(*Xs).argmax(1).cpu().numpy()
            f = f1_score(ytr[stop_idx], pv, average='macro')
            if f > best: best, state, wait, best_ep = f, {k: v.clone() for k, v in net.state_dict().items()}, 0, ep
            else:
                wait += 1
                if wait >= patience: break
        net.load_state_dict(state); return net, best, best_ep, st

    done = set()
    if CKPT.exists():
        for line in open(CKPT): d = json.loads(line); done.add((d['r'], d['k'], d['seed']))
    gpu = torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'
    with open(CKPT, "a") as fh:
        for r, k, tr, va in folds(ytr):
            if r not in reps: continue
            for seed in SEEDS:
                if (r, k, seed) in done: continue
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
                fit, stop = train_test_split(tr, test_size=0.15, stratify=ytr[tr], random_state=seed)
                t0 = time.time(); net, best_inner, ep, st = train(fit, stop); sec = time.time() - t0
                net.eval()
                with torch.no_grad(): p = net(*view(va, st)).softmax(1).cpu().numpy()
                pred = p.argmax(1)
                rec = {'r': r, 'k': k, 'seed': seed, 'selected_epoch': int(ep), 'best_inner_macroF1': float(best_inner),
                       'outer_macroF1': float(f1_score(ytr[va], pred, average='macro')),
                       'outer_OA': float(accuracy_score(ytr[va], pred)), 'train_seconds': sec, 'device': gpu,
                       'va': [int(i) for i in va], 'p': p.round(8).tolist()}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"r={r} k={k} seed={seed}: outer F1 {rec['outer_macroF1']*100:.1f} (epoch {ep}, {sec:.1f}s)", flush=True)


def corrected(v, J, ratio=0.25):
    v = np.asarray(v, float); m = v.mean(); s = v.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    h = stats.t.ppf(0.975, J - 1) * se; t = m / se
    return {'mean': float(m), 'sd_ddof0': float(v.std()), 'ci95': [float(m - h), float(m + h)],
            't': float(t), 'p_two_sided': float(2 * stats.t.sf(abs(t), J - 1))}


def classwise(yt, pred):
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2]).astype(float)
    ua = np.diag(cm) / cm.sum(0); pa = np.diag(cm) / cm.sum(1); f = 2 * ua * pa / (ua + pa)
    oa = np.trace(cm) / cm.sum(); pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    return {'UA': ua.tolist(), 'PA': pa.tolist(), 'F1': f.tolist(), 'OA': float(oa), 'kappa': float((oa - pe) / (1 - pe)),
            'macroF1': float(f.mean()), 'cm': cm.astype(int).tolist()}


def summarise():
    _, ytr = stems_labels()
    recs = [json.loads(l) for l in open(CKPT)]
    reps = sorted({d['r'] for d in recs})
    by = {}
    for d in recs: by.setdefault((d['r'], d['k']), {})[d['seed']] = d
    keys = sorted(k for k in by if len(by[k]) == 3)
    prim = np.array([np.mean([by[k][s]['outer_macroF1'] for s in SEEDS]) for k in keys])
    sec_ = []
    for k in keys:
        va = np.array(by[k][SEEDS[0]]['va']); P = np.mean([np.array(by[k][s]['p']) for s in SEEDS], 0)
        sec_.append(f1_score(ytr[va], P.argmax(1), average='macro'))
    sec_ = np.array(sec_); J = len(keys)
    out = {'n_folds': J, 'reps': reps,
           'primary_mean_of_seed_f1': corrected(prim, J), 'secondary_f1_of_seed_avg_probs': corrected(sec_, J),
           'sd_over_all_runs_ddof0': float(np.std([d['outer_macroF1'] for d in recs])),
           'selected_epoch_range': [min(d['selected_epoch'] for d in recs), max(d['selected_epoch'] for d in recs)],
           'train_seconds_mean': float(np.mean([d['train_seconds'] for d in recs])), 'device': recs[0]['device']}
    # gate F2: repetition 0 against the published 15 runs (65.2%)
    pub = json.load(open(ROOT / "code/diagnostics_r1/diag_dl_earlystop.json"))['B_inner_earlystop']['per_fold']
    r0 = [i for i, k in enumerate(keys) if k[0] == 0]
    if len(r0) == 5:
        m0 = float(prim[r0].mean()); out['gate_F2_rep0'] = {'new_rep0_mean': m0, 'published_rep0_mean': float(np.mean(pub)),
                                                            'abs_diff': abs(m0 - float(np.mean(pub))),
                                                            'pass': bool(abs(m0 - float(np.mean(pub))) <= 0.03)}
    # paired comparison with the ensemble (nested weights) on the same folds
    ens = {(d['r'], d['k']): d['f1_nested'] for d in json.load(open(ROOT / "code/diagnostics_r1/diag_nested_perfold.json"))['per_fold']}
    e = np.array([ens[k] for k in keys])
    out['ensemble_same_folds'] = corrected(e, J)
    out['paired_primary_DL_minus_ensemble'] = corrected(prim - e, J) | {'folds_DL_higher': int((prim > e).sum())}
    out['paired_secondary_DL_minus_ensemble'] = corrected(sec_ - e, J) | {'folds_DL_higher': int((sec_ > e).sum())}
    # aggregated out-of-fold class-wise values (seed-averaged probabilities, then averaged over repetitions)
    if len(reps) == 10:
        agg = np.zeros((len(ytr), 3))
        for k in keys:
            va = np.array(by[k][SEEDS[0]]['va']); agg[va] += np.mean([np.array(by[k][s]['p']) for s in SEEDS], 0)
        agg /= 10; out['aggregated_oof_classwise'] = classwise(ytr, agg.argmax(1))
    import sklearn
    out['versions'] = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                       'scikit-learn': sklearn.__version__, 'cuda': torch.version.cuda}
    out['per_fold'] = [{'r': k[0], 'k': k[1], 'primary': float(p), 'secondary': float(s), 'ensemble': float(x)}
                       for k, p, s, x in zip(keys, prim, sec_, e)]
    json.dump(out, open(OUT / "exp_B_summary.json", "w"), indent=1)
    q = out['primary_mean_of_seed_f1']; d = out['paired_primary_DL_minus_ensemble']
    print(f"folds {J}; DL primary {q['mean']*100:.1f} ± {q['sd_ddof0']*100:.1f}; ensemble {out['ensemble_same_folds']['mean']*100:.1f}; "
          f"DL-ens {d['mean']*100:+.2f} [{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}] p={d['p_two_sided']:.3f}")
    if 'gate_F2_rep0' in out: print('gate F2:', out['gate_F2_rep0'])


if MODE == 'run':
    a = sys.argv[3]; reps = list(range(int(a.split('-')[0]), int(a.split('-')[-1]) + 1))
    run(reps)
elif MODE == 'summarise':
    summarise()
else:
    sys.exit("mode: run <reps> | summarise")
