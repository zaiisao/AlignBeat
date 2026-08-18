#!/bin/bash
# Queue the fixed-b arms behind the learn_b arm: the GPUs are shared and currently
# hold other users' work, so wait for real headroom rather than OOM-ing on launch.
cd /home/sogang/jaehoon/BeatFCOS
PY=/home/sogang/mnt/db_2/anaconda3/envs/beatfcos/bin/python
D="--ballroom_audio_dir /disk1/taegum/mnt/labeled_data/ballroom/data --ballroom_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/ballroom/label --hainsworth_audio_dir /disk1/taegum/mnt/labeled_data/hains/data --hainsworth_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/hainsworth/label --rwc_popular_audio_dir /disk1/taegum/mnt/labeled_data/rwc_popular/data --rwc_popular_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/rwc_popular/label --carnatic_audio_dir /disk4/taegum/carnatic/data --carnatic_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/carnatic/label --harmonix_audio_dir /disk4/taegum/harmonix_griffinlim/audio --harmonix_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/harmonix/label --validation_fold 0 --batch_size 16 --num_workers 6 --preload --epochs 100 --no_adj --omega_db 3.2 --optimizer radam_lookahead --lr 1e-3 --lr_factor 0.5 --lr_patience 5 --min_lr 1e-6 --patience 12"
# b = 0.005 is the current default (~149 ms assumed timing noise on a 29.7 s window);
# the measured mean absolute matched residual on a trained model is ~0.22, i.e. ~44x
# larger, which is why lambda_L1 = 1/b = 200 makes the per-pair cost ~96% timing.
for b in 0.02 0.05; do
  tag=$(echo "b$b" | tr -d '.')
  while true; do
    G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', ' '($3-$2) >= 15000 {print $1; exit}')
    if [ -n "$G" ]; then
      echo "$(date '+%H:%M') launching b=$b on GPU $G"
      setsid nohup $PY train.py --head_type subset $D --b_scale $b --gpu "$G" \
        --checkpoint_dir "./checkpoints_subset_$tag" --log_file "./subset_$tag.log" \
        > /dev/null 2>&1 < /dev/null &
      sleep 300   # let it allocate before considering the next
      break
    fi
    sleep 120
  done
done
echo "$(date '+%H:%M') b sweep queued"
