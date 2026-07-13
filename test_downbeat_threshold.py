#!/usr/bin/env python3
"""
fold0 체크포인트로, score_threshold를 낮췄을 때 downbeat F-measure=0.000인
곡 비율이 줄어드는지 확인 (재학습 없이 순수 재평가만). BEAT는 0.000이 거의
안 나오는데 DOWNBEAT만 37.9%가 0.000이 나온 게, downbeat 채널의 confidence가
threshold(0.20)를 못 넘어서 그런 건지(threshold 문제) 확인하기 위함.
"""
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import torch
from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater
from beatfcos.beat_eval import evaluate_beat_f_measure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data_clusters = torch.tensor([0.42574675, 0.66719675, 1.24245649, 1.93286828, 2.78558922])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=training_data_clusters, args=None,
    head_type="fcos",
    dmodel=128, nhead=2, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=512,
    centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=22050, backbone_type="wavebeat",
)

state_dict = torch.load("/disk1/taegum/mnt/BeatFCOS/checkpoints_fold0_test/retinanet_27.pt", map_location="cpu")
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)

DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label"),
    "beatles": ("/disk1/taegum/mnt/labeled_data/beatles/data", "/disk1/taegum/mnt/labeled_data/beatles/label"),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label"),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "/disk1/taegum/mnt/labeled_data/rwc_popular/label"),
}

val_datasets = []
for name, (audio_dir, annot_dir) in DATASETS.items():
    val_datasets.append(BeatDataset(
        audio_dir, annot_dir, dataset=name,
        audio_sample_rate=22050, audio_downsampling_factor=512,
        subset="val", augment=False, half=True, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=0,
    ))
val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, batch_size=1, shuffle=False, collate_fn=collater)

for thresh in [0.20, 0.10, 0.05, 0.02, 0.01]:
    beat_f, downbeat_f, results = evaluate_beat_f_measure(
        val_dataloader, model, 512, 22050, score_threshold=0.20, downbeat_score_threshold=thresh
    )
    downbeat_scores = [r['downbeat_scores']['F-measure'] for r in results]
    beat_scores = [r['beat_scores']['F-measure'] for r in results]
    n = len(downbeat_scores)
    db_zero = sum(1 for s in downbeat_scores if s == 0.0)
    beat_zero = sum(1 for s in beat_scores if s == 0.0)
    print(f"threshold={thresh:.2f} | Beat F(avg)={beat_f:.3f} (0.000인 곡 {beat_zero}/{n}) | "
          f"Downbeat F(avg)={downbeat_f:.3f} (0.000인 곡 {db_zero}/{n} = {db_zero/n*100:.1f}%)")
