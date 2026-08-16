#!/usr/bin/env python3
"""
현재 3개 실험(fcos/fcos_lite/fcos_no_fpn) 체크포인트에 대해, soft-NMS의
downbeat_sigma와 downbeat_phase_reweight 옵션을 스윕해서 downbeat F-measure에
영향이 있는지 확인. score_threshold/downbeat_score_threshold는 이미 검증된
0.20으로 고정하고(evaluate_all_datasets.py와 동일 조건), sigma/phase_reweight만
바꿔가며 6개 데이터셋(ballroom/beatles/hainsworth/rwc_popular/carnatic/harmonix)
macro-average로 비교.
"""
import argparse
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import numpy as np
import torch
from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater
from beatfcos.beat_eval import evaluate_beat_f_measure

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--head_type', type=str, default="fcos", choices=['fcos', 'fcos_lite', 'fcos_no_fpn'])
parser.add_argument('--clusters', type=str, required=True)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--validation_fold', type=int, default=0)
args = parser.parse_args()

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512
SCORE_THRESHOLD = 0.20
DOWNBEAT_SCORE_THRESHOLD = 0.20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data_clusters = torch.tensor([float(x) for x in args.clusters.split(",")])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=training_data_clusters, args=None,
    head_type=args.head_type,
    dmodel=128, nhead=args.nhead, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
    centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=AUDIO_SAMPLE_RATE, backbone_type="wavebeat",
)
state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if unexpected:
    raise RuntimeError(f"체크포인트에 모델에 없는 키가 있음: {unexpected}")
model = model.to(device)
model.eval()

DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label", "val", args.validation_fold),
    "beatles": ("/disk1/taegum/mnt/labeled_data/beatles/data", "/disk1/taegum/mnt/labeled_data/beatles/label", "val", args.validation_fold),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label", "val", args.validation_fold),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "/disk1/taegum/mnt/labeled_data/rwc_popular/label", "val", args.validation_fold),
    "carnatic": ("/disk4/taegum/carnatic/data", "/disk4/taegum/carnatic/label", "val", args.validation_fold),
    "harmonix": ("/disk4/taegum/harmonix_griffinlim/audio", "/disk4/taegum/harmonix_griffinlim/annotations_urinieto", "val", args.validation_fold),
}

val_datasets = []
for name, (audio_dir, annot_dir, subset, validation_fold) in DATASETS.items():
    val_datasets.append(BeatDataset(
        audio_dir, annot_dir, dataset=name,
        audio_sample_rate=AUDIO_SAMPLE_RATE, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
        subset=subset, augment=False, half=True, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=validation_fold,
    ))
val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, batch_size=1, shuffle=False, collate_fn=collater)

print(f"체크포인트: {args.checkpoint} (head_type={args.head_type})\n", flush=True)

sigmas = [None, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
for phase_reweight in [False, True]:
    for sigma in sigmas:
        beat_f, downbeat_f, _ = evaluate_beat_f_measure(
            val_dataloader, model, AUDIO_DOWNSAMPLING_FACTOR, AUDIO_SAMPLE_RATE,
            score_threshold=SCORE_THRESHOLD, downbeat_score_threshold=DOWNBEAT_SCORE_THRESHOLD,
            downbeat_sigma=sigma, downbeat_phase_reweight=phase_reweight,
        )
        sigma_label = "default(0.5)" if sigma is None else f"{sigma:.1f}"
        print(f"phase_reweight={phase_reweight!s:5s} sigma={sigma_label:>13s} | Beat F={beat_f:.3f} | Downbeat F={downbeat_f:.3f} | Joint F={(beat_f+downbeat_f)/2:.3f}", flush=True)
