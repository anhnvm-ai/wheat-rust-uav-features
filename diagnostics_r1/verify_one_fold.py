"""Round 4: before resuming on another machine, recompute a stored fold and require exact equality. Prints no scores.
    python3 verify_one_fold.py C <config> <r> <k>       e.g.  C K_mask_sub 0 0
    python3 verify_one_fold.py D <row> <hs> <r> <k>     e.g.  D tuned sub 0 0   (a tuned fold takes several minutes)
Runs the experiment script without its mode block (data and functions only, nothing written), recomputes the fold and
compares every stored field except the timing. Exit code 0 only if all fields are identical."""
import os, sys, json, tempfile
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"   # as in the scripts
from pathlib import Path

D = Path(__file__).resolve().parent
exp, key = sys.argv[1], sys.argv[2:]
script = D / ('exp_C_auxiliary_nested.py' if exp == 'C' else 'exp_D_nested_hyperparams.py')
src = script.read_text(); g = {'__name__': 'verify', '__file__': str(script)}
sys.argv = [str(script), tempfile.mkdtemp(), 'none']
exec(compile(src[:src.index("\nif MODE == 'gate':")], str(script), 'exec'), g)
assert g['folds_ok'](), 'folds differ from folds_10x5.csv'
r, k = int(key[-2]), int(key[-1])
Xc = g['config'](key[0]) if exp == 'C' else g['matrix'](key[1])
tr, va = list(g['StratifiedKFold'](5, shuffle=True, random_state=g['L'].SEED + r).split(Xc, g['y']))[k]


def stored(name, match):
    for line in open(D / name):
        d = json.loads(line)
        if all(d[f] == v for f, v in match.items()): return d
    sys.exit(f'fold not found in {name}')


same = lambda new, old: sorted(f for f in old if f != 'seconds' and json.loads(json.dumps(new.get(f), default=float)) != old[f])
if exp == 'C':
    rec, pr = g['run_fold'](key[0], Xc, r, k, tr, va); m = {'config': key[0], 'r': r, 'k': k}
    bad = same(rec, stored('exp_C_folds.jsonl', m)) + ['probs.' + f for f in same(pr, stored('exp_C_probs.jsonl', m))]
else:
    rec = g['run_fold'](key[0], key[1], Xc, r, k, tr, va)
    bad = same(rec, stored('exp_D_folds.jsonl', {'row': key[0], 'hs': key[1], 'r': r, 'k': k}))
print(' '.join([exp] + key), 'identical' if not bad else f'DIFFERENT in {bad}')
sys.exit(1 if bad else 0)
