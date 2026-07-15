#!/usr/bin/env python3
"""
박사님 요청: soft-NMS를 통과해서 최종적으로 살아남은 예측 박스들의 실제
위치/폭을 보고, 연속된 두 박스가 서로 많이 겹치는지 확인.

지금까지는 GT랑 비교하는 점(tick) 시각화만 했지, 예측 박스 자체의 폭(interval)과
연속 박스간 IoU는 안 봤음 - 이 스크립트는 그 부분을 직접 확인한다.
"""
import json
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import torch
from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater
from beatfcos.beat_eval import get_results_from_model
from beatfcos.utils import calc_iou

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clusters = torch.tensor([0.42574675, 0.66719675, 1.93286828])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=clusters, args=None, head_type="fcos",
    dmodel=128, nhead=2, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
    centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=AUDIO_SAMPLE_RATE, backbone_type="wavebeat",
)
state_dict = torch.load("/disk1/taegum/mnt/BeatFCOS/checkpoints_fold0_levelfix_test/retinanet_82.pt", map_location="cpu", weights_only=False)
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label"),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label"),
}
N_SONGS_PER_DATASET = 3
SCORE_THRESHOLD = 0.20
DOWNBEAT_SCORE_THRESHOLD = 0.20  # 최종 확정된 최적값

output = []
overlap_stats = {"beat": [], "downbeat": []}

for dataset_name, (audio_dir, annot_dir) in DATASETS.items():
    val_dataset = BeatDataset(
        audio_dir, annot_dir, dataset=dataset_name,
        audio_sample_rate=AUDIO_SAMPLE_RATE, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
        subset="val", augment=False, half=True, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=0,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collater)

    count = 0
    for data in val_dataloader:
        if count >= N_SONGS_PER_DATASET:
            break
        audio, target, metadata = data

        predicted_scores, predicted_labels, predicted_boxes, losses = get_results_from_model(
            audio, target, model,
            score_threshold=SCORE_THRESHOLD, iou_threshold=0.5, max_thresh=1,
        )
        predicted_scores = predicted_scores.cpu()
        predicted_labels = predicted_labels.cpu()
        predicted_boxes = predicted_boxes.cpu()

        def boxes_for_label(label):
            mask = predicted_labels == label
            boxes = predicted_boxes[mask]
            scores = predicted_scores[mask]
            if boxes.shape[0] == 0:
                return boxes, scores
            order = boxes[:, 0].argsort()
            return boxes[order], scores[order]

        beat_boxes, beat_scores = boxes_for_label(1)
        downbeat_boxes, downbeat_scores = boxes_for_label(0)

        def consecutive_ious(boxes):
            if boxes.shape[0] < 2:
                return []
            ious = []
            for i in range(boxes.shape[0] - 1):
                iou = calc_iou(boxes[i:i+1], boxes[i+1:i+2])
                ious.append(float(iou[0, 0]))
            return ious

        beat_ious = consecutive_ious(beat_boxes)
        downbeat_ious = consecutive_ious(downbeat_boxes)
        overlap_stats["beat"].extend(beat_ious)
        overlap_stats["downbeat"].extend(downbeat_ious)

        gt_beats, gt_downbeats = [], []
        for iv in target[0]:
            label = int(iv[2])
            l = float(iv[0]) * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE
            r = float(iv[1]) * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE
            (gt_beats if label == 1 else gt_downbeats).append([l, r])

        def to_seconds(boxes):
            return (boxes * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE).tolist()

        song_entry = {
            "dataset": dataset_name,
            "filename": metadata[0]["Filename"].split("/")[-1],
            "duration": float(predicted_boxes[:, 1].max() * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE) if predicted_boxes.shape[0] > 0 else 0,
            "gt_beats": gt_beats,
            "gt_downbeats": gt_downbeats,
            "pred_beat_boxes": to_seconds(beat_boxes) if beat_boxes.shape[0] > 0 else [],
            "pred_beat_scores": beat_scores.tolist(),
            "pred_downbeat_boxes": to_seconds(downbeat_boxes) if downbeat_boxes.shape[0] > 0 else [],
            "pred_downbeat_scores": downbeat_scores.tolist(),
            "beat_consecutive_ious": beat_ious,
            "downbeat_consecutive_ious": downbeat_ious,
        }
        output.append(song_entry)
        print(f"[{dataset_name}] {song_entry['filename']}: beat boxes={len(beat_boxes)} (avg consec IoU={sum(beat_ious)/max(1,len(beat_ious)):.3f}) | "
              f"downbeat boxes={len(downbeat_boxes)} (avg consec IoU={sum(downbeat_ious)/max(1,len(downbeat_ious)):.3f})")
        count += 1

print("\n=== 전체 평균 연속 박스 IoU ===")
for label, ious in overlap_stats.items():
    if ious:
        print(f"{label}: 평균={sum(ious)/len(ious):.3f}, 최대={max(ious):.3f}, 곡당 겹침(IoU>0.3) 비율={sum(1 for x in ious if x>0.3)/len(ious)*100:.1f}%")

with open("/tmp/claude-1002/-home-taegum/08e05cf0-c28f-4af7-bcdd-4a1cbe00e2f2/scratchpad/final_boxes_data.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to final_boxes_data.json")
