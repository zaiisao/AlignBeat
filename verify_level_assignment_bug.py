#!/usr/bin/env python3
"""
실제 학습 데이터(4개 데이터셋, fold0 train split)의 beat/downbeat GT interval
길이(초)를 뽑아서, 현재 3-레벨 FPN 구조에서 실제로 쓰이는
interval_length_ranges[0..2] (5개 클러스터 중 앞 3개 구간만)에 몇 %가
들어가는지, 그리고 제안한 3-클러스터 수정안으로는 몇 %가 들어가는지 직접
계산해서 비교한다. 재학습/추론 없이 순수 annotation 길이 분포만 검사.
"""
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import torch
from beatfcos.dataloader import BeatDataset
from beatfcos.losses import clusters_to_interval_length_ranges

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 512

DATASETS = {
    "ballroom": ("/disk1/taegum/mnt/labeled_data/ballroom/data", "/disk1/taegum/mnt/labeled_data/ballroom/label"),
    "beatles": ("/disk1/taegum/mnt/labeled_data/beatles/data", "/disk1/taegum/mnt/labeled_data/beatles/label"),
    "hainsworth": ("/disk1/taegum/mnt/labeled_data/hains/data", "/disk1/taegum/mnt/labeled_data/hains/label"),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "/disk1/taegum/mnt/labeled_data/rwc_popular/label"),
}

all_beat_lengths = []
all_downbeat_lengths = []

for name, (audio_dir, annot_dir) in DATASETS.items():
    ds = BeatDataset(
        audio_dir, annot_dir, dataset=name,
        audio_sample_rate=AUDIO_SAMPLE_RATE, audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
        subset="train", augment=False, half=False, preload=False,
        length=2097152, dry_run=False, spectral=True, validation_fold=0,
    )
    beat_count = 0
    downbeat_count = 0
    for i in range(len(ds.audio_files)):
        annot_filename = ds.annot_files[i]
        beat_samples, downbeat_samples, beat_indices, time_signature = ds.load_annot(annot_filename)

        beat_samples = sorted(beat_samples)
        downbeat_samples = sorted(downbeat_samples)

        for j in range(len(beat_samples) - 1):
            length_sec = (beat_samples[j+1] - beat_samples[j]) / AUDIO_SAMPLE_RATE
            if length_sec > 0:
                all_beat_lengths.append(length_sec)
                beat_count += 1

        for j in range(len(downbeat_samples) - 1):
            length_sec = (downbeat_samples[j+1] - downbeat_samples[j]) / AUDIO_SAMPLE_RATE
            if length_sec > 0:
                all_downbeat_lengths.append(length_sec)
                downbeat_count += 1

    print(f"{name}: {len(ds.audio_files)} train files -> beat intervals={beat_count}, downbeat intervals={downbeat_count}")

all_beat_lengths = torch.tensor(all_beat_lengths)
all_downbeat_lengths = torch.tensor(all_downbeat_lengths)

print(f"\n총 beat interval 개수: {len(all_beat_lengths)}, 평균 길이: {all_beat_lengths.mean():.3f}s, 범위: [{all_beat_lengths.min():.3f}, {all_beat_lengths.max():.3f}]")
print(f"총 downbeat interval 개수: {len(all_downbeat_lengths)}, 평균 길이: {all_downbeat_lengths.mean():.3f}s, 범위: [{all_downbeat_lengths.min():.3f}, {all_downbeat_lengths.max():.3f}]")

# ---- 현재(버그) 설정: 5개 클러스터, 실제로는 앞 3개 구간만 쓰임 ----
CURRENT_CLUSTERS = torch.tensor([0.42574675, 0.66719675, 1.24245649, 1.93286828, 2.78558922])
current_ranges_full = clusters_to_interval_length_ranges(CURRENT_CLUSTERS)
current_ranges_used = current_ranges_full[:3]  # get_fcos_positives가 실제로 순회하는 3개 레벨만

print(f"\n[현재 설정] 5클러스터 → 전체 구간: {current_ranges_full}")
print(f"[현재 설정] 실제 사용되는 구간(레벨 0~2만): {current_ranges_used}")

def in_any_range(lengths, ranges):
    covered = torch.zeros(len(lengths), dtype=torch.bool)
    for lo, hi in ranges:
        covered |= (lengths >= lo) & (lengths <= hi)
    return covered

beat_covered_current = in_any_range(all_beat_lengths, current_ranges_used)
downbeat_covered_current = in_any_range(all_downbeat_lengths, current_ranges_used)

print(f"\n[현재 설정] beat: {beat_covered_current.float().mean()*100:.1f}% 가 유효 레벨에 들어감 (positive anchor 받을 자격 있음)")
print(f"[현재 설정] downbeat: {downbeat_covered_current.float().mean()*100:.1f}% 가 유효 레벨에 들어감 <- 이게 핵심 확인 대상")

# ---- 제안한 수정: 3개 클러스터 (beat 2개 그대로 + downbeat 대표값 1개) ----
PROPOSED_CLUSTERS = torch.tensor([0.42574675, 0.66719675, 1.93286828])
proposed_ranges = clusters_to_interval_length_ranges(PROPOSED_CLUSTERS)
print(f"\n[제안 설정] 3클러스터 → 구간: {proposed_ranges}")

beat_covered_proposed = in_any_range(all_beat_lengths, proposed_ranges)
downbeat_covered_proposed = in_any_range(all_downbeat_lengths, proposed_ranges)

print(f"\n[제안 설정] beat: {beat_covered_proposed.float().mean()*100:.1f}% 가 유효 레벨에 들어감")
print(f"[제안 설정] downbeat: {downbeat_covered_proposed.float().mean()*100:.1f}% 가 유효 레벨에 들어감")
