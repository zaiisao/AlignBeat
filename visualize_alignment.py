#!/usr/bin/env python3
"""
fold0 체크포인트로 몇 곡을 뽑아서 예측 beat/downbeat 위치와 GT 위치를
초 단위로 뽑아 JSON으로 저장. Downbeat score가 안 좋은 이유를 스코어
숫자만으로는 알기 어려워서, 실제 interval이 GT랑 얼마나 정렬되는지
시각적으로 보기 위함.
"""
import sys
import json
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import torch
from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512

training_data_clusters = torch.tensor([0.42574675, 0.66719675, 1.24245649, 1.93286828, 2.78558922])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=training_data_clusters, args=None,
    head_type="fcos",
    dmodel=128, nhead=2, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
    centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=AUDIO_SAMPLE_RATE, backbone_type="wavebeat",
)
state_dict = torch.load("/disk1/taegum/mnt/BeatFCOS/checkpoints_fold0_test/retinanet_27.pt", map_location="cpu")
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label"),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label"),
}

N_SONGS_PER_DATASET = 3
SCORE_THRESHOLD = 0.20       # beat
DOWNBEAT_SCORE_THRESHOLD = 0.05

output = []

for dataset_name, (audio_dir, annot_dir) in DATASETS.items():
    val_dataset = BeatDataset(
        audio_dir, annot_dir, dataset=dataset_name,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
        subset="val", augment=False, half=True, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=0,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collater)

    count = 0
    for data in val_dataloader:
        if count >= N_SONGS_PER_DATASET:
            break
        audio, target, metadata = data
        metadata = metadata[0]

        audio_gpu = audio.to(device)
        target_gpu = target.to(device)

        with torch.no_grad():
            predicted_scores, predicted_labels, predicted_boxes, losses = model(
                (audio_gpu, target_gpu),
                iou_threshold=0.5,
                score_threshold=SCORE_THRESHOLD,
                downbeat_score_threshold=DOWNBEAT_SCORE_THRESHOLD,
                max_thresh=1,
            )

        predicted_scores = predicted_scores.cpu()
        predicted_labels = predicted_labels.cpu()
        predicted_boxes = predicted_boxes.cpu()

        pred_beats = []
        pred_downbeats = []
        for i in range(predicted_boxes.shape[0]):
            label = int(predicted_labels[i])
            score = float(predicted_scores[i])
            left_time = float(predicted_boxes[i, 0]) * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE
            if label == 1:
                pred_beats.append((left_time, score))
            elif label == 0:
                pred_downbeats.append((left_time, score))

        gt_beats = []
        gt_downbeats = []
        for interval in target[0]:
            label = int(interval[2])
            left_time = float(interval[0]) * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE
            if label == 1:
                gt_beats.append(left_time)
            elif label == 0:
                gt_downbeats.append(left_time)

        duration = max(
            [t for t, _ in pred_beats] + gt_beats + [0]
        )

        song_entry = {
            "dataset": dataset_name,
            "filename": metadata["Filename"].split("/")[-1],
            "duration": duration,
            "gt_beats": sorted(gt_beats),
            "gt_downbeats": sorted(gt_downbeats),
            "pred_beats": sorted(pred_beats),
            "pred_downbeats": sorted(pred_downbeats),
        }
        output.append(song_entry)

        print(f"[{dataset_name}] {song_entry['filename']}: "
              f"GT beats={len(gt_beats)} GT downbeats={len(gt_downbeats)} | "
              f"pred beats={len(pred_beats)} pred downbeats={len(pred_downbeats)}")

        count += 1

with open("/tmp/claude-1002/-home-taegum/08e05cf0-c28f-4af7-bcdd-4a1cbe00e2f2/scratchpad/alignment_data.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to alignment_data.json")
