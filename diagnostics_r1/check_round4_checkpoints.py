"""Round 4: check the per-fold checkpoint files of experiments C and D before resuming after an interruption.
Prints no scores. Checks that every line parses (a hard stop can leave a truncated last line, which would stop the resume),
that no fold appears twice, and that exp_C_folds.jsonl and exp_C_probs.jsonl hold the same folds (summarise adds up every
line of exp_C_probs.jsonl, so a missing or repeated line there would change the pooled per-class scores of Table 9).
    python3 check_round4_checkpoints.py [--fix]
--fix first copies each file to <name>.bak_<UTC time>, then drops unreadable lines, repeated folds and C folds that are
missing from one of the two files; the dropped folds are recomputed by the next run. Run it only while no experiment runs."""
import json, sys, time, shutil
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent
FILES = {'C_folds': ('exp_C_folds.jsonl', ('config', 'r', 'k')), 'C_probs': ('exp_C_probs.jsonl', ('config', 'r', 'k')),
         'D_folds': ('exp_D_folds.jsonl', ('row', 'hs', 'r', 'k'))}


def read(name, keys):
    p = D / name; good, bad = [], []
    if p.exists():
        for i, line in enumerate(open(p), 1):
            try: d = json.loads(line); good.append((tuple(d[k] for k in keys), line if line.endswith('\n') else line + '\n'))
            except (json.JSONDecodeError, KeyError): bad.append(i)
            if not line.endswith('\n') and i not in bad: bad.append(f'{i} (no newline)')   # the next append would join it
    return good, bad


data = {t: read(n, k) for t, (n, k) in FILES.items()}
problems = 0
for t, (good, bad) in data.items():
    c = Counter(k for k, _ in good); dup = sorted(k for k, v in c.items() if v > 1)
    print(f"{FILES[t][0]}: {len(good)} lines, {len(c)} folds, unreadable lines {bad or 'none'}, repeated folds {dup[:5] or 'none'}")
    problems += len(bad) + len(dup)
kf, kp = {k for k, _ in data['C_folds'][0]}, {k for k, _ in data['C_probs'][0]}
only_f, only_p = sorted(kf - kp), sorted(kp - kf)
print(f"C folds only in exp_C_folds.jsonl: {only_f or 'none'}; only in exp_C_probs.jsonl: {only_p or 'none'}")
problems += len(only_f) + len(only_p)
for t in ('C_folds', 'D_folds'):
    print(t, dict(sorted(Counter(k[:-2] for k, _ in data[t][0]).items())))

if problems and '--fix' in sys.argv:
    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime()); both = kf & kp
    for t, (good, bad) in data.items():
        p = D / FILES[t][0]; seen, keep = set(), []
        if not p.exists(): continue
        for k, line in good:
            if k in seen or (t != 'D_folds' and k not in both): continue
            seen.add(k); keep.append(line)
        shutil.copy2(p, p.with_name(p.name + '.bak_' + stamp)); p.write_text(''.join(keep))
        print(f"{p.name}: kept {len(keep)} lines (backup {p.name}.bak_{stamp})")
print('OK: consistent' if not problems else f"{problems} problem(s)" + ('' if '--fix' in sys.argv else '; rerun with --fix while nothing runs'))
sys.exit(1 if problems and '--fix' not in sys.argv else 0)
