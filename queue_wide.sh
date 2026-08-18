#!/bin/bash
# Wait for a GPU with >=30GB free (subset_phase/subset_count are finishing on 0/1),
# then launch the encoder-width arm: dmodel 256 / d_hid 1024 against our 128/512.
# BT training recipe throughout, so width is the only variable vs subset_bt1e3.
cd /home/sogang/jaehoon/BeatFCOS
PY=/home/sogang/mnt/db_2/anaconda3/envs/beatfcos/bin/python
D="--ballroom_audio_dir /disk1/taegum/mnt/labeled_data/ballroom/data --ballroom_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/ballroom/label --hainsworth_audio_dir /disk1/taegum/mnt/labeled_data/hains/data --hainsworth_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/hainsworth/label --rwc_popular_audio_dir /disk1/taegum/mnt/labeled_data/rwc_popular/data --rwc_popular_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/rwc_popular/label --carnatic_audio_dir /disk4/taegum/carnatic/data --carnatic_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/carnatic/label --harmonix_audio_dir /disk4/taegum/harmonix_griffinlim/audio --harmonix_annot_dir /home/sogang/jaehoon/BeatFCOS/dataset_folds/harmonix/label --validation_fold 0 --batch_size 16 --num_workers 6 --preload --epochs 100 --no_adj --omega_db 3.2 --optimizer radam_lookahead --lr 1e-3 --lr_factor 0.2 --lr_patience 2 --min_lr 1e-7 --patience 10"

while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', ' '($3-$2) >= 30000 {print $1; exit}')
  if [ -n "$GPU" ]; then
    echo "$(date '+%H:%M') launching subset_wide on GPU $GPU"
    nohup $PY train.py --head_type subset $D --dmodel 256 --d_hid 1024 --nhead 8 \
      --gpu "$GPU" --checkpoint_dir ./checkpoints_subset_wide \
      --log_file ./subset_wide.log > /dev/null 2>&1 &
    exit 0
  fi
  echo "$(date '+%H:%M') no GPU with 30GB free yet"
  sleep 120
done
