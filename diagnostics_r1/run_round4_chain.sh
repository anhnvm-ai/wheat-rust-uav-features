#!/bin/bash
# Round 4 (30 Oct 2026): run experiments D and C only after the pre-registration commit 5e24d99 is on GitHub.
# Every stage resumes from its per-fold checkpoint (exp_C_folds.jsonl, exp_D_folds.jsonl), so after a Colab disconnect
# simply start this script again:  nohup setsid bash run_round4_chain.sh > run_round4_chain.log 2>&1 &
# Progress is written to round4_chain_status.txt (on Drive). Before a restart after a hard stop, the script checks the
# checkpoint files (check_round4_checkpoints.py) and stops if they disagree. NJOBS defaults to the number of CPU threads;
# results do not depend on it (each fold is computed on its own with fixed seeds and single-threaded BLAS).
D=/content/drive/MyDrive/Kaggle_Prepared/code/diagnostics_r1
REPO=/content/drive/MyDrive/Kaggle_Prepared/revision_r1/github_repo
COMMIT=5e24d99615d94d83271299e299d5cdf7d715c42f
ST=$D/round4_chain_status.txt
cd "$D" || exit 1
stamp() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" | tee -a "$ST"; }

stamp "chain started; waiting until the pre-registration commit $COMMIT is on GitHub (main or tag pre_aux_hyperparam)"
for i in $(seq 1 1440); do                                   # up to 24 h, checked every minute
  REMOTE=$(timeout 30 git -C "$REPO" ls-remote origin refs/heads/main 2>/dev/null | cut -f1)
  TAG=$(timeout 30 git -C "$REPO" ls-remote origin 'refs/tags/pre_aux_hyperparam^{}' 2>/dev/null | cut -f1)
  [ "$REMOTE" = "$COMMIT" ] || [ "$TAG" = "$COMMIT" ] && break
  sleep 60
done
if [ "$REMOTE" != "$COMMIT" ] && [ "$TAG" != "$COMMIT" ]; then stamp "STOP: commit not on GitHub after 24 h"; exit 1; fi
stamp "pushed: GitHub main = ${REMOTE:-?}; tag pre_aux_hyperparam on GitHub -> ${TAG:-NOT PUSHED}"
if ! python3 check_round4_checkpoints.py > check_round4_checkpoints.log 2>&1; then
  stamp "STOP: checkpoint files disagree (see check_round4_checkpoints.log); run check_round4_checkpoints.py --fix"; exit 1
fi

export NJOBS=${NJOBS:-$(nproc)}
stamp "checkpoints consistent; NJOBS=$NJOBS on $(nproc) CPU threads; $(lscpu | sed -n 's/^Model name: *//p')"
stamp "D sub: start";  python3 exp_D_nested_hyperparams.py "$D" run sub >> exp_D_sub.log 2>&1;  stamp "D sub: exit $?"
stamp "C sub: start";  python3 exp_C_auxiliary_nested.py "$D" run K_mask_sub K_ctx_sub A_full_sub A_noHS A_noRGB_sub \
  A_noMS_sub A_MS A_RGB A_HS_sub B1_raw_sub B2_veg_sub B3_shape_sub >> exp_C_sub.log 2>&1;  stamp "C sub: exit $?"
stamp "C calib: start"; python3 exp_C_auxiliary_nested.py "$D" calib > exp_C_calib.log 2>&1; stamp "C calib: exit $?"
stamp "C cor: start";  python3 exp_C_auxiliary_nested.py "$D" run A_full_cor A_noRGB_cor A_noMS_cor A_HS_cor B1_raw_cor \
  B2_veg_cor B3_shape_cor K_mask_cor K_ctx_cor >> exp_C_cor.log 2>&1;  stamp "C cor: exit $?"
stamp "D cor: start";  python3 exp_D_nested_hyperparams.py "$D" run cor >> exp_D_cor.log 2>&1;  stamp "D cor: exit $?"
stamp "summarise"
python3 exp_C_auxiliary_nested.py "$D" summarise > exp_C_summarise.log 2>&1; stamp "C summarise: exit $?"
python3 exp_D_nested_hyperparams.py "$D" summarise > exp_D_summarise.log 2>&1; stamp "D summarise: exit $?"
stamp "CHAIN_DONE"
