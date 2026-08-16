#!/usr/bin/env python3
"""
evaluate_with_dbn.py의 후속. 이전 시도에서 진단된 문제(anchor 기반 sparse
score를 scatter-max로 rasterize하면, downbeat 신호가 실제로 몰려있는 level2
(stride=4)의 성긴 간격 때문에 4프레임마다 한 번만 값이 있고 나머지는 0인
스파이크 모양이 되어 DBN이 기대하는 매끈한 연속 activation과 거리가 멈)를
가우시안 스무딩으로 보완해서 재시도.

동기: DBN은 HMM 기반이라 프레임 사이의 완만한 상승/하강을 보고 transition을
추정하는데, scatter-max 직후의 활성화는 "0, 0, 0, 0.8, 0, 0, 0, 0.9, ..." 같은
디랙 델타에 가까운 형태라 이 가정이 깨짐. scatter-max 결과에 1D 가우시안
컨볼루션을 한 번 더 적용해서 스파이크 주변으로 값을 퍼뜨려 연속적인 곡선으로
만듦 - sigma는 downbeat가 몰려있는 level2의 stride(4)의 절반 정도(2프레임)로
설정.
"""
import argparse
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import numpy as np
np.int = int
np.float = float

import torch
import torch.nn.functional as F
import mir_eval
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--clusters', type=str, default="0.42574675,0.66719675,1.93286828")
parser.add_argument('--validation_fold', type=int, default=0)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--sigma', type=float, default=2.0, help="가우시안 스무딩 표준편차(프레임 단위)")
parser.add_argument('--datasets', type=str, default="ballroom,hainsworth,beatles,rwc_popular")
args = parser.parse_args()

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512
FPS = AUDIO_SAMPLE_RATE / AUDIO_DOWNSAMPLING_FACTOR

ALL_DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label"),
    "beatles": ("/disk1/taegum/mnt/labeled_data/beatles/data", "/disk1/taegum/mnt/labeled_data/beatles/label"),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label"),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "/disk1/taegum/mnt/labeled_data/rwc_popular/label"),
}
DATASETS = {k: ALL_DATASETS[k] for k in args.datasets.split(",")}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data_clusters = torch.tensor([float(x) for x in args.clusters.split(",")])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=training_data_clusters, args=None,
    head_type="fcos",
    dmodel=128, nhead=args.nhead, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
    centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=AUDIO_SAMPLE_RATE, backbone_type="wavebeat",
)
state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


def make_gaussian_kernel(sigma, device):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)


def rasterize_smooth(scores, anchors, length, kernel):
    """anchor별 score를 level0 프레임 배열로 scatter-max한 뒤, 가우시안
    컨볼루션으로 스파이크를 주변 프레임까지 퍼뜨려 연속적인 곡선으로 만듦."""
    frame_idx = anchors.long().clamp(0, length - 1)
    activation = torch.zeros(length, device=scores.device)
    activation = activation.scatter_reduce(0, frame_idx, scores, reduce="amax", include_self=True)
    padded = activation.view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    smoothed = F.conv1d(padded, kernel, padding=pad).view(-1)
    # 컨볼루션은 스파이크를 퍼뜨리면서 최댓값을 낮추는 경향이 있어서, DBN의
    # threshold 파라미터와 스케일을 맞추기 위해 원래 최댓값 근처로 재정규화.
    if smoothed.max() > 1e-8:
        smoothed = smoothed / smoothed.max() * activation.max().clamp(min=1e-8)
    return smoothed.cpu().numpy()


def get_gt_times(target_row):
    beat_times, downbeat_times = [], []
    for interval in target_row:
        label = int(interval[2])
        t = float(interval[0]) * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE
        if label == 1:
            beat_times.append(t)
        elif label == 0:
            downbeat_times.append(t)
    return np.array(sorted(set(beat_times))), np.array(sorted(set(downbeat_times)))


DBN_KWARGS = dict(observation_lambda=6, transition_lambda=100, threshold=0.2)
dbn_beat = DBNBeatTrackingProcessor(fps=FPS, **DBN_KWARGS)
dbn_downbeat = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS, **DBN_KWARGS)

kernel = make_gaussian_kernel(args.sigma, device)

results = {}
for name, (audio_dir, annot_dir) in DATASETS.items():
    val_dataset = BeatDataset(
        audio_dir, annot_dir, dataset=name,
        audio_sample_rate=AUDIO_SAMPLE_RATE, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
        subset="val", augment=False, half=True, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=args.validation_fold,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collater)

    beat_scores_list, downbeat_scores_list = [], []
    beat_cmlt_list, beat_amlt_list, downbeat_cmlt_list, downbeat_amlt_list = [], [], [], []

    for index, (audio, target, metadata) in enumerate(val_dataloader):
        audio = audio.to(device)
        target_gpu = target.to(device)

        with torch.no_grad():
            beat_raw, downbeat_raw, all_anchors = model((audio, target_gpu), return_raw_scores=True)

        length = int(all_anchors.max().item()) + 1
        beat_activation = rasterize_smooth(beat_raw[0], all_anchors, length, kernel)
        downbeat_activation = rasterize_smooth(downbeat_raw[0], all_anchors, length, kernel)

        gt_beats, gt_downbeats = get_gt_times(target[0])
        gt_beats = mir_eval.beat.trim_beats(gt_beats)
        gt_downbeats = mir_eval.beat.trim_beats(gt_downbeats)

        try:
            est_beats = dbn_beat.process_offline(beat_activation)
        except Exception:
            est_beats = np.array([])
        est_beats = mir_eval.beat.trim_beats(est_beats)
        beat_score = mir_eval.beat.evaluate(gt_beats, est_beats)

        combined_activation = np.stack([beat_activation, downbeat_activation], axis=1)
        try:
            downbeat_result = dbn_downbeat.process(combined_activation)
            est_downbeats = downbeat_result[downbeat_result[:, 1] == 1][:, 0]
        except Exception:
            est_downbeats = np.array([])
        est_downbeats = mir_eval.beat.trim_beats(est_downbeats)
        downbeat_score = mir_eval.beat.evaluate(gt_downbeats, est_downbeats)

        beat_scores_list.append(beat_score['F-measure'])
        beat_cmlt_list.append(beat_score['Correct Metric Level Total'])
        beat_amlt_list.append(beat_score['Any Metric Level Total'])
        downbeat_scores_list.append(downbeat_score['F-measure'])
        downbeat_cmlt_list.append(downbeat_score['Correct Metric Level Total'])
        downbeat_amlt_list.append(downbeat_score['Any Metric Level Total'])

        print(f"{index+1}/{len(val_dataloader)} {metadata[0]['Filename']} "
              f"BEAT F:{beat_score['F-measure']:.3f} DOWNBEAT F:{downbeat_score['F-measure']:.3f}", flush=True)

    results[name] = {
        'beat_f': np.mean(beat_scores_list), 'beat_cmlt': np.mean(beat_cmlt_list), 'beat_amlt': np.mean(beat_amlt_list),
        'downbeat_f': np.mean(downbeat_scores_list), 'downbeat_cmlt': np.mean(downbeat_cmlt_list), 'downbeat_amlt': np.mean(downbeat_amlt_list),
    }
    r = results[name]
    print(f"\n[{name}] (DBN-v2, sigma={args.sigma}) Beat F:{r['beat_f']:.3f} CMLt:{r['beat_cmlt']:.3f} AMLt:{r['beat_amlt']:.3f} | "
          f"Downbeat F:{r['downbeat_f']:.3f} CMLt:{r['downbeat_cmlt']:.3f} AMLt:{r['downbeat_amlt']:.3f}\n", flush=True)

print("\n=== DBN-v2 디코딩 요약 ===")
for name, r in results.items():
    print(f"{name:<12} Beat  F:{r['beat_f']:.3f} CMLt:{r['beat_cmlt']:.3f} AMLt:{r['beat_amlt']:.3f}  |  "
          f"Downbeat  F:{r['downbeat_f']:.3f} CMLt:{r['downbeat_cmlt']:.3f} AMLt:{r['downbeat_amlt']:.3f}")
