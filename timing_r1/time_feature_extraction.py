"""Timing of the V7 feature extraction (Reviewer 3, comment 6; added during the revision, 9 Oct 2026).

Times only (1) reading the RGB/MS/HS files of a tile and (2) computing the 149 features, with the
extractor used for every reported result (feat_v7_lib.load_sample + extract_*_raw_features, i.e. the
HS block as submitted). No model is trained and no reported result changes.

Single CPU thread; the OS page cache is dropped before each repetition when permitted (root), so file
reading is timed from disk rather than from memory. A sanity check confirms that the features computed
here equal the saved feature table used in the paper.

Usage:
    python3 time_feature_extraction.py <data_root_with_train_and_val> <out_json> [repeats=3]
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

cv2.setNumThreads(1)
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import feat_v7_lib as F  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
splits = {'train': root / 'train', 'test': root / 'val'}


def drop_page_cache():
    try:
        subprocess.run(['sync'], check=True)
        with open('/proc/sys/vm/drop_caches', 'w') as fh:
            fh.write('3\n')
        return True
    except Exception:
        return False


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


runs, cache_dropped, X_train, ids_train = [], [], None, None
for r in range(repeats):
    cache_dropped.append(drop_page_cache())
    run = {}
    for name, base in splits.items():
        ids = sorted(p.stem for p in (base / 'HS').glob('*.tif'))
        t_read = t_rgb = t_ms = t_hs = 0.0
        X = []
        for sid in ids:
            t0 = time.perf_counter()
            rgb, ms, hs = F.load_sample(sid, base)
            t1 = time.perf_counter()
            a = F.extract_rgb_raw_features(rgb)
            t2 = time.perf_counter()
            b = F.extract_ms_raw_features(ms)
            t3 = time.perf_counter()
            c = F.extract_hs_raw_features(hs)
            t4 = time.perf_counter()
            t_read += t1 - t0
            t_rgb += t2 - t1
            t_ms += t3 - t2
            t_hs += t4 - t3
            if r == 0 and name == 'train':
                X.append(np.concatenate([a, b, c]))
        n = len(ids)
        feat = t_rgb + t_ms + t_hs
        run[name] = {'n_tiles': n, 'read_s': t_read, 'features_s': feat, 'rgb_s': t_rgb, 'ms_s': t_ms,
                     'hs_s': t_hs, 'read_ms_per_tile': 1e3 * t_read / n,
                     'features_ms_per_tile': 1e3 * feat / n,
                     'total_ms_per_tile': 1e3 * (t_read + feat) / n}
        if r == 0 and name == 'train':
            X_train = np.nan_to_num(np.array(X), nan=0, posinf=0, neginf=0)
            ids_train = ids
    runs.append(run)
    print(f"repeat {r + 1}: " + "; ".join(
        f"{k}: read {v['read_ms_per_tile']:.2f} ms/tile, features {v['features_ms_per_tile']:.2f} ms/tile"
        for k, v in run.items()), flush=True)

# sanity check against the saved feature table used in the paper
saved = {}
with open(CODE.parent / 'outputs' / 'reviewer_experiments' / 'features_v7.csv') as fh:
    rd = csv.reader(fh)
    header = next(rd)
    for row in rd:
        saved[Path(row[header.index('filename')]).stem] = np.array(row[:149], dtype=float)
diffs = [float(np.max(np.abs(X_train[i] - saved[s]))) for i, s in enumerate(ids_train) if s in saved]


def median(key, split):
    return float(np.median([run[split][key] for run in runs]))


summary = {split: {k: median(k, split) for k in ('read_ms_per_tile', 'features_ms_per_tile', 'total_ms_per_tile')}
           for split in splits}
all_tiles = sum(runs[0][s]['n_tiles'] for s in splits)
summary['all_tiles'] = {
    'n_tiles': all_tiles,
    'read_s_median': float(np.median([sum(run[s]['read_s'] for s in splits) for run in runs])),
    'features_s_median': float(np.median([sum(run[s]['features_s'] for s in splits) for run in runs])),
}
result = {
    'what': 'V7 feature extraction timing (file reading + 149 features), no model training',
    'extractor': 'code/feat_v7_lib.py (load_sample + extract_rgb/ms/hs_raw_features; HS block as submitted)',
    'repeats': repeats, 'threads': 1, 'page_cache_dropped_before_each_repeat': cache_dropped,
    'storage': str(root), 'cpu_model': cpu_model(), 'logical_cpus': os.cpu_count(),
    'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
    'tifffile': tifffile.__version__,
    'sanity_check_vs_features_v7_csv': {'n_compared': len(diffs), 'max_abs_diff': max(diffs) if diffs else None},
    'summary_median_over_repeats': summary, 'runs': runs,
}
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(out, 'w'), indent=2)
print(json.dumps({'summary': summary, 'sanity': result['sanity_check_vs_features_v7_csv'],
                  'cpu': result['cpu_model'], 'cache_dropped': cache_dropped}, indent=1))
