#!/bin/bash
# Round 5 (31 Oct 2026): run experiment E only after the pre-registration (tag pre_compact_nested) is on GitHub.
# The run resumes from its per-fold checkpoint (exp_E_folds.jsonl, exp_E_probs.jsonl), so after a Colab disconnect simply start
# this script again:  nohup setsid bash run_round5_chain.sh > run_round5_chain.log 2>&1 &
# Progress is written to round5_chain_status.txt (on Drive). NJOBS defaults to the number of CPU threads; results do not depend
# on it (each fold is computed on its own with fixed seeds and single-threaded BLAS).
D=/content/drive/MyDrive/Kaggle_Prepared/code/diagnostics_r1
REPO=/content/drive/MyDrive/Kaggle_Prepared/revision_r1/github_repo
ST=$D/round5_chain_status.txt
cd "$D" || exit 1
stamp() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" | tee -a "$ST"; }

COMMIT=$(git -C "$REPO" rev-parse 'pre_compact_nested^{commit}' 2>/dev/null)
[ -n "$COMMIT" ] || { stamp "STOP: no local tag pre_compact_nested"; exit 1; }
stamp "chain started; waiting until tag pre_compact_nested -> $COMMIT is on GitHub"
for i in $(seq 1 1440); do                                   # up to 24 h, checked every minute
  LS=$(timeout 30 git -C "$REPO" ls-remote origin refs/tags/pre_compact_nested 'refs/tags/pre_compact_nested^{}' 2>/dev/null)
  TAG=$(echo "$LS" | awk '$2=="refs/tags/pre_compact_nested^{}"{print $1}')       # annotated tag: peeled commit
  [ -n "$TAG" ] || TAG=$(echo "$LS" | awk '$2=="refs/tags/pre_compact_nested"{print $1}')   # lightweight tag
  [ "$TAG" = "$COMMIT" ] && break
  sleep 60
done
if [ "$TAG" != "$COMMIT" ]; then stamp "STOP: tag not on GitHub after 24 h"; exit 1; fi
REMOTE=$(timeout 30 git -C "$REPO" ls-remote origin refs/heads/main 2>/dev/null | cut -f1)
stamp "pushed: tag pre_compact_nested on GitHub -> $TAG; GitHub main = ${REMOTE:-?}"
for f in exp_E_compact_nested.py PREREG_compact_nested.md; do
  cmp -s "$f" "$REPO/diagnostics_r1/$f" || { stamp "STOP: $f differs from the committed version"; exit 1; }
  [ "$(git -C "$REPO" show "pre_compact_nested:diagnostics_r1/$f" | sha256sum)" = "$(sha256sum < "$f")" ] || { stamp "STOP: $f differs from the tag"; exit 1; }
done

export NJOBS=${NJOBS:-$(nproc)}
stamp "NJOBS=$NJOBS on $(nproc) CPU threads; $(lscpu | sed -n 's/^Model name: *//p')"
stamp "E run: start";  python3 exp_E_compact_nested.py "$D" run >> exp_E_run.log 2>&1;  rc=$?;  stamp "E run: exit $rc"
[ $rc -eq 0 ] || { stamp "STOP: run failed (gate E1 or error; see exp_E_run.log)"; exit 1; }
stamp "E summarise: start"; python3 exp_E_compact_nested.py "$D" summarise > exp_E_summarise.log 2>&1; stamp "E summarise: exit $?"
stamp "CHAIN_DONE"
