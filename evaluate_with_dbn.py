#!/usr/bin/env python3
"""
soft-NMS 대신 madmom DBN(Dynamic Bayesian Network)으로 디코딩했을 때 성능이
얼마나 달라지는지 진단하는 스크립트. 재학습 없이 이미 학습된 체크포인트의
anchor별 원본 score(NMS 걸기 전, model.forward(..., return_raw_scores=True))를
뽑아서, level0 기준 프레임 배열로 rasterize한 뒤 madmom DBN에 통과시킨다.

동기: Beat Transformer 논문 등 비교 대상 논문 수치는 전부 madmom DBN으로
디코딩된 결과인데, 우리는 soft-NMS만 쓰고 있어서 이게 성능 격차의 상당 부분을
설명할 수 있다는 가설을 검증하기 위함 (특히 F-measure보다 CMLt/AMLt가 더
크게 떨어지는 패턴이 DBN 부재의 전형적 signature).

결과(2026-07-14, fold0 levelfix 체크포인트 기준): 가설은 부분적으로만 맞았음.
- DBN은 continuity 지표(CMLt/AMLt)는 실제로 개선시킴 (특히 downbeat AMLt).
- 근데 downbeat F-measure는 4개 데이터셋 전부에서 soft-NMS보다 오히려 나쁨
  (observation_lambda/threshold 그리드서치로 최적값 찾아봐도 마찬가지).
- WaveBeat 원본 저장소(/home/taegum/wavebeat)의 실제 DBN 코드도 확인했으나,
  그건 TCN이 자연스럽게 만드는 dense한 프레임별 activation을 전제로 함 - 우리
  FCOS anchor 기반 sparse score를 rasterize해서 넣는 이 스크립트의 변환 방식
  자체가 아직 제대로 캘리브레이션되지 않은 것으로 보임(anchor 위치가 downbeat는
  레벨2 stride=4 간격이라 activation이 성글게 나옴 등). 이 스크립트는 검증된
  기존 파이프라인이 아니라 이번에 새로 설계한 것이므로, 더 개선하려면 rasterize
  방식 자체를 재설계해야 함 - 현재는 결론이 안 난 상태로 보류.
"""
import argparse
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import numpy as np
# madmom(0.16.1)이 numpy 신버전에서 제거된 별칭(np.int/np.float)을 컴파일된
# Cython 코드(madmom/ml/hmm.pyx) 안에서 그대로 참조해서, import 전에 임시로
# 복원해줘야 함 (np.bool/np.object 등은 numpy 자체 내부 코드가 실제 numpy 타입을
# 기대하므로 건드리면 numpy.ma가 깨짐 - int/float만 안전하게 확인됨).
np.int = int
np.float = float

import torch
import mir_eval
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--clusters', type=str, default="0.42574675,0.66719675,1.93286828")
parser.add_argument('--validation_fold', type=int, default=0)
parser.add_argument('--nhead', type=int, default=2)
args = parser.parse_args()

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512
FPS = AUDIO_SAMPLE_RATE / AUDIO_DOWNSAMPLING_FACTOR  # level0 anchor 간격 = 1 frame

DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label"),
    "beatles": ("/disk1/taegum/mnt/labeled_data/beatles/data", "/disk1/taegum/mnt/labeled_data/beatles/label"),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label"),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "/disk1/taegum/mnt/labeled_data/rwc_popular/label"),
}

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


def rasterize(scores, anchors, length):
    """anchor별 score를 level0 프레임(0..length-1) 배열로 scatter-max."""
    frame_idx = anchors.long().clamp(0, length - 1)
    activation = torch.zeros(length, device=scores.device)
    activation = activation.scatter_reduce(0, frame_idx, scores, reduce="amax", include_self=True)
    return activation.cpu().numpy()


def get_gt_times(target_row):
    """target[0] (곡 하나의 annotation 텐서, (N,3)=(left,right,class))에서
    beat/downbeat GT 시간(초) 배열을 뽑음 (beat_eval.py와 동일 컨벤션)."""
    beat_times, downbeat_times = [], []
    for interval in target_row:
        label = int(interval[2])
        t = float(interval[0]) * AUDIO_DOWNSAMPLING_FACTOR / AUDIO_SAMPLE_RATE
        if label == 1:
            beat_times.append(t)
        elif label == 0:
            downbeat_times.append(t)
    return np.array(sorted(set(beat_times))), np.array(sorted(set(downbeat_times)))


# Beat Transformer 논문이 쓴 것과 동일한 madmom DBN 하이퍼파라미터
# (obs_lambda=6, trans_lambda=100, threshold=0.2). 기본값(observation_lambda=16,
# threshold=0)으로 테스트했을 때는 beat가 GT 대비 ~4배 과다검출됐는데, 이
# 파라미터로 바꾸니 정확히 맞아떨어짐 - 논문과 같은 조건으로 비교하기 위해서도
# 이 값을 반드시 써야 함.
DBN_KWARGS = dict(observation_lambda=6, transition_lambda=100, threshold=0.2)
dbn_beat = DBNBeatTrackingProcessor(fps=FPS, **DBN_KWARGS)
dbn_downbeat = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS, **DBN_KWARGS)

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
        beat_activation = rasterize(beat_raw[0], all_anchors, length)
        downbeat_activation = rasterize(downbeat_raw[0], all_anchors, length)

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
            # DBNDownBeatTrackingProcessor는 DBNBeatTrackingProcessor와 달리
            # process_offline이 없고 process()만 있음.
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
              f"BEAT F:{beat_score['F-measure']:.3f} DOWNBEAT F:{downbeat_score['F-measure']:.3f}")

    results[name] = {
        'beat_f': np.mean(beat_scores_list), 'beat_cmlt': np.mean(beat_cmlt_list), 'beat_amlt': np.mean(beat_amlt_list),
        'downbeat_f': np.mean(downbeat_scores_list), 'downbeat_cmlt': np.mean(downbeat_cmlt_list), 'downbeat_amlt': np.mean(downbeat_amlt_list),
    }
    r = results[name]
    print(f"\n[{name}] (DBN) Beat F:{r['beat_f']:.3f} CMLt:{r['beat_cmlt']:.3f} AMLt:{r['beat_amlt']:.3f} | "
          f"Downbeat F:{r['downbeat_f']:.3f} CMLt:{r['downbeat_cmlt']:.3f} AMLt:{r['downbeat_amlt']:.3f}\n")

print("\n=== DBN 디코딩 요약 ===")
for name, r in results.items():
    print(f"{name:<12} Beat  F:{r['beat_f']:.3f} CMLt:{r['beat_cmlt']:.3f} AMLt:{r['beat_amlt']:.3f}  |  "
          f"Downbeat  F:{r['downbeat_f']:.3f} CMLt:{r['downbeat_cmlt']:.3f} AMLt:{r['downbeat_amlt']:.3f}")
