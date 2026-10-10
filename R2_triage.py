"""
R2_triage.py  —  Item 3: Selective Triage + Uncertainty-aware Decision Support (merged).

Turns the weak point (low Health recall, Health<->Rust spectral overlap) into a deployment
feature: the model abstains on its least-confident cases ("refer to agronomist") and stays
accurate on what it auto-classifies. Pure decision-support framing that fits SAT.

Built on the SAME averaged repeated-CV OOF probabilities used for the locked calibration
numbers (no new model, no leakage). Reports a risk-coverage operating table:
  coverage -> macro-F1 / accuracy / Rust miss-rate / Health precision&recall / ECE
and a selective-prediction figure.

Output: outputs/reviewer_experiments/V7_triage.json  +  V7_risk_coverage.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import r2_lib as L

ROOT = Path("/content/drive/MyDrive/Kaggle_Prepared"); OUT = ROOT / "outputs/reviewer_experiments"
X, y, fn, cols = L.load_features(OUT / "features_v7.csv")
H, O, R = L.IDX['Health'], L.IDX['Other'], L.IDX['Rust']

print("Computing averaged repeated 10x5 OOF probabilities (locked protocol)...")
cv = L.repeated_cv(X, y, reps=L.REPS, verbose=True, label='triage')
oof = cv['oof']; pred = oof.argmax(1)
conf = oof.max(1)
entropy = -np.sum(np.clip(oof, 1e-9, 1) * np.log(np.clip(oof, 1e-9, 1)), axis=1)
print(f"Full-coverage macro-F1={f1_score(y,pred,average='macro')*100:.2f}%  acc={accuracy_score(y,pred)*100:.2f}%")


def ece(probs, labels, mask, bins=10):
    p = probs[mask]; lab = labels[mask]; c = p.max(1); pc = p.argmax(1); acc = (pc == lab).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1.0 / bins; m = (c > lo) & (c <= hi)
        if m.sum() > 0: e += (m.sum() / len(c)) * abs(acc[m].mean() - c[m].mean())
    return float(e)


# ---- risk-coverage table: abstain on the lowest-confidence (1-coverage) fraction ----
order = np.argsort(-conf)                      # most confident first
cov_levels = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
table = {}
for cov in cov_levels:
    n = int(round(cov * len(y))); keep = np.zeros(len(y), bool); keep[order[:n]] = True
    yk, pk = y[keep], pred[keep]
    rust_recall = recall_score(yk, pk, labels=[R], average='macro') if (yk == R).any() else float('nan')
    h_prec = precision_score(yk, pk, labels=[H], average='macro', zero_division=0)
    h_rec = recall_score(yk, pk, labels=[H], average='macro') if (yk == H).any() else float('nan')
    # composition of the abstained (referred) set
    ref = ~keep; ref_comp = {c: int(((y == i) & ref).sum()) for c, i in L.IDX.items()}
    table[f"{cov:.2f}"] = {
        'coverage': cov, 'n_auto': int(n), 'n_referred': int((~keep).sum()),
        'macroF1': float(f1_score(yk, pk, average='macro')),
        'accuracy': float(accuracy_score(yk, pk)),
        'rust_miss_rate': float(1 - rust_recall),
        'health_precision': float(h_prec), 'health_recall': float(h_rec),
        'ECE': ece(oof, y, keep),
        'referred_composition': ref_comp}

# ---- AURC-style summary + selective gain ----
full = table['1.00']; cov80 = table['0.80']
summary = {
    'full_macroF1': full['macroF1'], 'full_accuracy': full['accuracy'],
    'at80_macroF1': cov80['macroF1'], 'at80_accuracy': cov80['accuracy'],
    'macroF1_gain_at80': cov80['macroF1'] - full['macroF1'],
    'accuracy_gain_at80': cov80['accuracy'] - full['accuracy'],
    'rust_miss_full': full['rust_miss_rate'], 'rust_miss_at80': cov80['rust_miss_rate']}

# fraction of referred cases that are the hard Health/Rust pair, at 80% coverage
ref80 = cov80['referred_composition']
summary['referred_at80_health_plus_rust_frac'] = float(
    (ref80['Health'] + ref80['Rust']) / max(1, sum(ref80.values())))

res = {'item': 'selective_triage_uncertainty', 'n': int(len(y)),
       'uncertainty_measures': ['max_prob_confidence', 'predictive_entropy'],
       'note': 'abstain on lowest-confidence (1-coverage) fraction; abstained = referred to agronomist',
       'risk_coverage_table': table, 'summary': summary,
       'mean_entropy_correct': float(entropy[pred == y].mean()),
       'mean_entropy_wrong': float(entropy[pred != y].mean())}
json.dump(res, open(OUT / "V7_triage.json", 'w'), indent=2)

# ---- figure: macro-F1 & accuracy & rust-miss vs coverage ----
cv_x = [table[f"{c:.2f}"]['coverage'] * 100 for c in cov_levels]
f1y = [table[f"{c:.2f}"]['macroF1'] * 100 for c in cov_levels]
accy = [table[f"{c:.2f}"]['accuracy'] * 100 for c in cov_levels]
rmy = [table[f"{c:.2f}"]['rust_miss_rate'] * 100 for c in cov_levels]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(cv_x, f1y, 'o-', color='#2c7fb8', lw=2, label='Macro-F1')
ax.plot(cv_x, accy, 's-', color='#31a354', lw=2, label='Accuracy')
ax.plot(cv_x, rmy, '^--', color='#d95f0e', lw=2, label='Rust miss-rate')
ax.axvline(80, ls=':', color='gray')
ax.set_xlabel("Coverage (% auto-classified; rest referred to agronomist)")
ax.set_ylabel("%"); ax.invert_xaxis()
ax.set_title("Selective prediction: abstaining on low-confidence cases\nraises accuracy on the auto-classified set")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "V7_risk_coverage.png", dpi=150); plt.close()

print("\n" + "=" * 70)
print("coverage  macroF1  acc    rustMiss  Hprec  Hrec   ECE   referred(H/O/R)")
for c in cov_levels:
    t = table[f"{c:.2f}"]; rc = t['referred_composition']
    print(f"  {c*100:4.0f}%   {t['macroF1']*100:5.1f}  {t['accuracy']*100:5.1f}   {t['rust_miss_rate']*100:5.1f}    "
          f"{t['health_precision']*100:4.0f}  {t['health_recall']*100:4.0f}  {t['ECE']:.3f}  "
          f"{rc['Health']}/{rc['Other']}/{rc['Rust']}")
print(f"\nAt 80% coverage: macro-F1 {full['macroF1']*100:.1f}->{cov80['macroF1']*100:.1f}%  "
      f"acc {full['accuracy']*100:.1f}->{cov80['accuracy']*100:.1f}%  "
      f"Rust miss {full['rust_miss_rate']*100:.1f}->{cov80['rust_miss_rate']*100:.1f}%")
print(f"Referred set at 80% cov is {summary['referred_at80_health_plus_rust_frac']*100:.0f}% Health/Rust (the hard pair)")
print(f"Entropy correct vs wrong: {res['mean_entropy_correct']:.3f} vs {res['mean_entropy_wrong']:.3f}")
print("Saved V7_triage.json + V7_risk_coverage.png")
print("=" * 70)
