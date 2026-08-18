#!/bin/bash
# Queue the SMC ablation for a THIRD architecture (head_type=fcos, the original
# P1/P2/P3 + FPN + separate-head baseline) and launch it only once GPUs free up.
#
# [why a third architecture] The running 2x2 answers "does SMC in the training pool
# hurt?" for fcos_lite and for the subset head. If fcos agrees, the effect belongs to
# the DATA and the pool decision is settled; if it disagrees, the cost is head-specific
# and the two heads need different pools - a very different conclusion. Taegum's fcos
# run reached Joint 0.868, so it is a well-characterised reference point.
#
# [why queued rather than launched now] GPUs 0 and 3 were at 100%/97% utilisation with
# under 8 GB free. Two more batch-16 runs there risked an OOM that would kill arms
# already several epochs in. This waits for real headroom instead.
#
# The two arms differ ONLY by whether --smc_* is passed.
set -u
cd /home/sogang/jaehoon/BeatFCOS
PY=/home/sogang/mnt/db_2/anaconda3/envs/beatfcos/bin/python
F=$PWD/dataset_folds
B=/disk1/taegum/mnt/labeled_data
SMCA=/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio
NEED_MIB=24000
POLL=180

CORE="--ballroom_audio_dir $B/ballroom/data --ballroom_annot_dir $F/ballroom/label \
--hainsworth_audio_dir $B/hains/data --hainsworth_annot_dir $F/hainsworth/label \
--rwc_popular_audio_dir $B/rwc_popular/data --rwc_popular_annot_dir $F/rwc_popular/label \
--carnatic_audio_dir /disk4/taegum/carnatic/data --carnatic_annot_dir $F/carnatic/label \
--harmonix_audio_dir /disk4/taegum/harmonix_griffinlim/audio --harmonix_annot_dir $F/harmonix/label \
--validation_fold 0 --batch_size 16 --num_workers 6 --preload --patience 10 --epochs 100"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v need="$NEED_MIB" '($3-$2) >= need {print $1}'
}

echo "[queue] $(date +%H:%M) waiting for 2 GPUs with >= ${NEED_MIB} MiB free"
while true; do
  mapfile -t G < <(free_gpus)
  if [ "${#G[@]}" -ge 2 ]; then
    A=${G[0]}; C=${G[1]}
    echo "[queue] $(date +%H:%M) launching on GPU $A (with smc) and GPU $C (without)"
    rm -rf checkpoints_fcos_smc checkpoints_fcos_nosmc
    setsid $PY train.py --head_type fcos $CORE \
      --smc_audio_dir "$SMCA" --smc_annot_dir "$F/smc/label" \
      --gpu "$A" --checkpoint_dir ./checkpoints_fcos_smc \
      --log_file ./fcos_smc.log </dev/null >/dev/null 2>&1 &
    sleep 10
    setsid $PY train.py --head_type fcos $CORE \
      --gpu "$C" --checkpoint_dir ./checkpoints_fcos_nosmc \
      --log_file ./fcos_nosmc.log </dev/null >/dev/null 2>&1 &
    sleep 20
    echo "[queue] $(date +%H:%M) launched; logs fcos_smc.log / fcos_nosmc.log"
    exit 0
  fi
  sleep "$POLL"
done
