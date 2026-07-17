#!/usr/bin/env python3
"""
학습 끝난 체크포인트를 각 데이터셋(val subset)별로 따로 평가.
train.py는 4개 데이터셋(ballroom/beatles/hainsworth/rwc_popular)을 합쳐서
하나의 val_dataloader로만 평가하기 때문에, 데이터셋별 성능이 안 보였음 - 이
스크립트는 데이터셋마다 따로 BeatDataset(val)/DataLoader를 만들어서 개별 평가.

gtzan, smc 둘 다 train.py에는 CLI 인자 자체가 없어서 학습에는 전혀 안 쓰였음 -
즉 이 체크포인트한테는 완전히 처음 보는 데이터셋들임.
smc는 다운비트 구분이 없는 데이터셋인데, dataloader.py의 파서가 모든 이벤트를
beat=1로 하드코딩하고(line 461) 그 값이 다운비트 판정 로직에도 그대로
재사용돼서(line 483-485), SMC의 ground truth downbeat == ground truth beat가
되어버림. 즉 "다운비트가 없어서 0이 나오는" 게 아니라 "모든 beat가 다운비트로도
라벨링된" 상태로 정상 평가됨 - Downbeat F-measure가 0이 아니어도 이상한 게
아님.

fold 정합성: ballroom/beatles/hainsworth/rwc_popular 4개는 8-fold CV로
학습했으므로 반드시 --validation_fold로 실제 held-out fold를 지정해야 함
(subset="val", validation_fold=None으로 두면 8-fold 분할과 무관한 옛
80/10/10 split의 val 부분을 봐서, 학습에 쓰인 곡이 섞여 들어가는 data
leakage가 생김). gtzan/smc는 fold 파일 자체가 없고 학습에도 전혀 안 쓰였으니
subset="full-val"로 전체 데이터를 봄.
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
parser.add_argument('--score_threshold', type=float, default=0.20)
parser.add_argument('--downbeat_score_threshold', type=float, default=0.05)
parser.add_argument('--downbeat_sigma', type=float, default=None,
                     help="soft-NMS의 downbeat 전용 sigma (안 주면 기존처럼 beat/downbeat 둘 다 0.5 사용)")
parser.add_argument('--validation_fold', type=int, default=0,
                     help="ballroom/beatles/hainsworth/rwc_popular의 8-fold CV held-out fold 번호 (체크포인트를 학습시킨 validation_fold와 일치해야 함)")
# 체크포인트를 학습시킨 training_data_clusters와 반드시 일치해야 함 (anchor
# base_size가 여기서 나오기 때문 - 학습 때와 다른 clusters로 평가하면 anchor
# 스케일이 어긋남). 예전 5-클러스터 체크포인트는 기본값 그대로, FPN
# 레벨/클러스터 개수 불일치를 고친 이후 체크포인트는 3개짜리로 넘겨야 함
# (train.py의 training_data_clusters 줄 주석 참고).
parser.add_argument('--clusters', type=str, default="0.42574675,0.66719675,1.24245649,1.93286828,2.78558922",
                     help="콤마로 구분된 클러스터 값. 체크포인트 학습에 쓴 값과 일치시킬 것")
args = parser.parse_args()

# (audio_dir, annot_dir, subset, validation_fold)
DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label", "val", args.validation_fold),
    "beatles": ("/disk1/taegum/mnt/labeled_data/beatles/data", "/disk1/taegum/mnt/labeled_data/beatles/label", "val", args.validation_fold),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label", "val", args.validation_fold),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "/disk1/taegum/mnt/labeled_data/rwc_popular/label", "val", args.validation_fold),
    "gtzan": ("/disk1/taegum/mnt/labeled_data/gtzan/data", "/disk1/taegum/mnt/labeled_data/gtzan/label", "full-val", None),
    "smc": ("/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio", "/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Annotations", "full-val", None),
}

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_data_clusters = torch.tensor([float(x) for x in args.clusters.split(",")])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=training_data_clusters, args=None,
    head_type="fcos",
    dmodel=128, nhead=2, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
    centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=AUDIO_SAMPLE_RATE, backbone_type="wavebeat",
)

state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
# strict=False: phase auxiliary head(regressionModel.phase.*)가 새로 추가돼서
# phase 도입 이전 체크포인트에는 이 키가 없음 - eval(soft-NMS 후처리)에는 phase
# 출력이 안 쓰이므로 누락돼도 무방함 (0으로 초기화된 채로 그냥 안 쓰이고 남음).
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if unexpected:
    raise RuntimeError(f"체크포인트에 모델에 없는 키가 있음: {unexpected}")
if missing and missing != ["regressionModel.phase.weight", "regressionModel.phase.bias"]:
    raise RuntimeError(f"예상 못한 키 누락: {missing}")
model = model.to(device)

print(f"체크포인트: {args.checkpoint}")
print(f"score_threshold(beat)={args.score_threshold} | downbeat_score_threshold={args.downbeat_score_threshold} | downbeat_sigma={args.downbeat_sigma}\n")

results = {}
for name, (audio_dir, annot_dir, subset, validation_fold) in DATASETS.items():
    val_dataset = BeatDataset(
        audio_dir, annot_dir, dataset=name,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
        subset=subset, augment=False, half=True, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=validation_fold,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collater
    )

    beat_f, downbeat_f, song_results = evaluate_beat_f_measure(
        val_dataloader, model, AUDIO_DOWNSAMPLING_FACTOR, AUDIO_SAMPLE_RATE,
        score_threshold=args.score_threshold,
        downbeat_score_threshold=args.downbeat_score_threshold,
        downbeat_sigma=args.downbeat_sigma,
    )
    # mir_eval.beat.evaluate가 곡마다 돌려주는 dict에서 CMLt(Correct Metric
    # Level Total)/AMLt(Any Metric Level Total)를 뽑아서 곡 평균을 냄 -
    # evaluate_beat_f_measure 자체는 F-measure만 집계해서 반환하기 때문.
    beat_cmlt = np.mean([r['beat_scores']['Correct Metric Level Total'] for r in song_results])
    beat_amlt = np.mean([r['beat_scores']['Any Metric Level Total'] for r in song_results])
    downbeat_cmlt = np.mean([r['downbeat_scores']['Correct Metric Level Total'] for r in song_results])
    downbeat_amlt = np.mean([r['downbeat_scores']['Any Metric Level Total'] for r in song_results])

    results[name] = (beat_f, downbeat_f, beat_cmlt, beat_amlt, downbeat_cmlt, downbeat_amlt)
    print(f"[{name}] Beat F: {beat_f:.3f} CMLt: {beat_cmlt:.3f} AMLt: {beat_amlt:.3f} | "
          f"Downbeat F: {downbeat_f:.3f} CMLt: {downbeat_cmlt:.3f} AMLt: {downbeat_amlt:.3f} | "
          f"Joint F: {(beat_f+downbeat_f)/2:.3f}")

print("\n=== 요약 ===")
for name, (beat_f, downbeat_f, beat_cmlt, beat_amlt, downbeat_cmlt, downbeat_amlt) in results.items():
    print(f"{name:<12} Beat  F:{beat_f:.3f} CMLt:{beat_cmlt:.3f} AMLt:{beat_amlt:.3f}  |  "
          f"Downbeat  F:{downbeat_f:.3f} CMLt:{downbeat_cmlt:.3f} AMLt:{downbeat_amlt:.3f}  |  "
          f"Joint F:{(beat_f+downbeat_f)/2:.3f}")
