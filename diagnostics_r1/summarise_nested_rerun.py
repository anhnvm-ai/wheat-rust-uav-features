"""Summaries computed from the outputs of diag_nested_perfold.py and the stored deep-baseline runs
(revision, October 2026); nothing is refitted here.
  1. Paired comparison on the five folds of repetition 0 (the folds of the fair deep baseline): per-fold
     difference deep network (mean over its three training seeds) minus ensemble (nested weights), with the
     Nadeau-Bengio corrected variance (1/J + n_test/n_train) s_d^2, J = 5, n_test/n_train = 1/4, t on J-1 df.
  2. Selective prediction with nested cut-offs: fold means and confusion matrices pooled over the 50 folds.
  3. Class-wise results of the ensemble with nested weights on the aggregated out-of-fold probabilities
     (mean over the ten repetitions), to compare with the validation-fold-weight values of the class-wise table.
Usage: python3 summarise_nested_rerun.py <dir with diag_*.json and diag_nested_perfold_probs.npz>"""
import sys, json
import numpy as np
from pathlib import Path
from scipy import stats
sys.path.insert(0, "/content/drive/MyDrive/Kaggle_Prepared/code")
import r2_lib as L
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

D = Path(sys.argv[1]); ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared")
run = json.load(open(D / "diag_nested_perfold.json"))
assert run['summary']['reproduced'], "the re-run did not reproduce the stored values; stop"
pf = run['per_fold']

# 1. paired comparison, repetition 0
ens0 = np.array([d['f1_nested'] for d in sorted(pf, key=lambda d: d['k']) if d['r'] == 0])
def seed_avg(path, key):
    v = np.array(json.load(open(path))[key]['per_fold']); return v.reshape(3, 5).mean(0)   # seed-major order
dl = {'B': seed_avg(D / "diag_dl_earlystop.json", 'B_inner_earlystop'),
      'C': seed_avg(D / "diag_dl_protoC.json", 'C_inner_epochs_retrain_full')}
J, ratio = 5, 1 / 4
paired = {'ensemble_rep0_fold_f1': ens0.tolist(), 'ensemble_rep0_mean': float(ens0.mean())}
for name, f in dl.items():
    d = f - ens0; m = d.mean(); s = d.std(ddof=1); se = np.sqrt((1 / J + ratio) * s ** 2)
    t = m / se; p = 2 * stats.t.sf(abs(t), J - 1); h = stats.t.ppf(0.975, J - 1) * se
    paired[name] = {'dl_fold_f1_seed_mean': f.tolist(), 'dl_mean': float(f.mean()), 'diff_per_fold': d.tolist(),
                    'mean_diff': float(m), 'sd_diff_ddof1': float(s), 'corrected_ci95': [float(m - h), float(m + h)],
                    't': float(t), 'p_two_sided': float(p), 'folds_dl_higher': int((d > 0).sum())}

# 2. selective prediction with nested cut-offs
sw = run['summary']['sweep_fold_means']; cms = run['summary']['sweep_cm_pooled_50_folds']
tri = {}
for c in sw:
    cm = np.array(cms[c]); n = cm.sum()
    tri[c] = {**{k: float(v) for k, v in sw[c].items()}, 'n_pooled': int(n), 'cm_pooled': cm.tolist(),
              'health_recall_pooled': float(cm[0, 0] / cm[0].sum()), 'health_precision_pooled': float(cm[0, 0] / cm[:, 0].sum()),
              'rust_miss_pooled': float(1 - cm[2, 2] / cm[2].sum()), 'acc_pooled': float(np.trace(cm) / n)}

# 3. class-wise results with nested weights (aggregated out-of-fold probabilities)
X, y, fn, cols = L.load_features(ROOT / "outputs/reviewer_experiments/features_v7.csv")
z = np.load(D / "diag_nested_perfold_probs.npz", allow_pickle=True)
agg = np.zeros((len(y), 3))
for va, p in zip(z['va'], z['p']): agg[np.asarray(va, dtype=int)] += np.asarray(p, dtype=float)
agg /= L.REPS; pred = agg.argmax(1); cm = confusion_matrix(y, pred, labels=[0, 1, 2])
ua = np.diag(cm) / cm.sum(0); pa = np.diag(cm) / cm.sum(1); f1 = 2 * ua * pa / (ua + pa)
oa = np.trace(cm) / cm.sum(); pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
classwise = {'cm': cm.tolist(), 'UA': ua.tolist(), 'PA': pa.tolist(), 'F1': f1.tolist(), 'OA': float(oa),
             'kappa': float((oa - pe) / (1 - pe)), 'macroF1': float(f1_score(y, pred, average='macro')),
             'same_cm_as_validation_fold_weights': cm.tolist() == [[81, 42, 77], [19, 169, 12], [37, 0, 163]]}

# fold means of Health recall/precision, pooled kappa, deferred composition, predictive entropy
def kappa(cm):
    cm = np.asarray(cm, float); n = cm.sum(); po = np.trace(cm) / n; pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe))
for c in tri:
    hr = [np.asarray(d['sweep'][c]['cm'])[0, 0] / np.asarray(d['sweep'][c]['cm'])[0].sum() for d in pf]
    hp = [np.asarray(d['sweep'][c]['cm'])[0, 0] / max(np.asarray(d['sweep'][c]['cm'])[:, 0].sum(), 1) for d in pf]
    cm = np.asarray(tri[c]['cm_pooled'])
    tri[c].update({'health_recall_fold_mean': float(np.mean(hr)), 'health_precision_fold_mean': float(np.mean(hp)),
                   'kappa_pooled': kappa(cm), 'deferred_by_class_pooled': (np.asarray(tri['1.00']['cm_pooled']).sum(1) - cm.sum(1)).tolist()})
ent_c, ent_w = [], []
for va, p_, d in zip(z['va'], z['p'], sorted(pf, key=lambda d: (d['r'], d['k']))):
    P = np.asarray(p_, dtype=float); yt = y[np.asarray(va, dtype=int)]
    e = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1); ok = P.argmax(1) == yt
    ent_c += e[ok].tolist(); ent_w += e[~ok].tolist()
entropy = {'mean_entropy_correct': float(np.mean(ent_c)), 'mean_entropy_wrong': float(np.mean(ent_w))}
out = {'paired_rep0': paired, 'selective_prediction_nested': tri, 'classwise_nested_aggregated_oof': classwise, 'entropy_nested': entropy}
json.dump(out, open(D / "nested_rerun_summary.json", "w"), indent=1)
for name in ['B', 'C']:
    q = paired[name]
    print(f"{name}: DL {q['dl_mean']*100:.1f} vs ensemble {paired['ensemble_rep0_mean']*100:.1f} on rep 0; diff {q['mean_diff']*100:+.1f} "
          f"[{q['corrected_ci95'][0]*100:+.1f}, {q['corrected_ci95'][1]*100:+.1f}], p={q['p_two_sided']:.2f}, DL higher in {q['folds_dl_higher']}/5 folds")
for c in tri:
    q = tri[c]
    print(f"cov {c}: realised {q['cov']*100:.1f} acc {q['acc']*100:.1f} F1 {q['f1']*100:.1f} rust miss {q['rust_miss']*100:.1f} "
          f"| pooled n={q['n_pooled']} Hrec {q['health_recall_pooled']*100:.1f} Hprec {q['health_precision_pooled']*100:.1f}")
print('class-wise (nested, aggregated OOF):', classwise['cm'], 'same as Table 7 row:', classwise['same_cm_as_validation_fold_weights'])
