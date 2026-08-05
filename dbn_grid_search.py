#!/usr/bin/env python3
"""
evaluate_with_dbn.py에서 soft-NMS보다 DBN이 downbeat F-measure가 낮게 나온 게
observation_lambda/threshold 하이퍼파라미터를 잘못 골라서인지 확인하기 위한
그리드서치. ballroom val 20곡의 activation을 미리 뽑아 캐싱해두고
(observation_lambda, threshold) 조합별로 재평가.

결과(2026-07-14, fold0 levelfix 체크포인트): obs_lambda=10, threshold=0.05~0.2
근처가 최적이었으나(Downbeat F≈0.677, 20곡 기준), 그래도 soft-NMS의 Downbeat F
(0.752, 85곡 기준)보다 낮았음. 즉 파라미터 튜닝으로는 격차가 안 메워짐 -
evaluate_with_dbn.py 상단 docstring의 결론(rasterize 변환 자체의 재설계가
필요) 참고.
"""
import numpy as np
np.int = int
np.float = float

import sys
sys.path.insert(0, '.')
import torch
import mir_eval
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater

device = torch.device('cuda')
clusters = torch.tensor([0.42574675, 0.66719675, 1.93286828])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=clusters, args=None, head_type='fcos',
    # dmodel=128, nhead=2, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,  # nhead=2는 dilated head가 죽는 버그
    dmodel=128, nhead=8, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, audio_downsampling_factor=512, centerness=False,
    postprocessing_type='soft_nms', audio_sample_rate=22050, backbone_type='wavebeat',
).to(device)
sd = torch.load('checkpoints_fold0_levelfix_test/retinanet_82.pt', map_location='cpu', weights_only=False)
sd = {k.replace('module.','',1): v for k,v in sd.items()}
model.load_state_dict(sd)
model.eval()

ds = BeatDataset('/disk1/taegum/mnt/labeled_data/ballroom/data','/disk1/taegum/mnt/labeled_data/ballroom/label',
    dataset='ballroom', audio_sample_rate=22050, audio_downsampling_factor=512,
    subset='val', augment=False, half=True, preload=False, length=2097152, dry_run=False, spectral=True, validation_fold=0)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collater)

FPS = 22050/512

def rasterize(scores, anchors, length):
    frame_idx = anchors.long().clamp(0, length-1)
    act = torch.zeros(length, device=scores.device).scatter_reduce(0, frame_idx, scores, reduce='amax', include_self=True)
    return act.cpu().numpy()

def gt_times(target_row):
    beats, downbeats = [], []
    for iv in target_row:
        label = int(iv[2]); t = float(iv[0]) * 512/22050
        (beats if label==1 else downbeats).append(t)
    return np.array(sorted(set(beats))), np.array(sorted(set(downbeats)))

# 곡 20개 미리 뽑아서 activation 캐싱 (GPU forward는 한 번만)
N_SONGS = 20
cache = []
it = iter(dl)
for i in range(N_SONGS):
    audio, target, metadata = next(it)
    audio = audio.to(device); target_gpu = target.to(device)
    with torch.no_grad():
        beat_raw, downbeat_raw, all_anchors = model((audio, target_gpu), return_raw_scores=True)
    length = int(all_anchors.max().item())+1
    beat_act = rasterize(beat_raw[0], all_anchors, length)
    down_act = rasterize(downbeat_raw[0], all_anchors, length)
    gt_beats, gt_downbeats = gt_times(target[0])
    cache.append((beat_act, down_act, mir_eval.beat.trim_beats(gt_beats), mir_eval.beat.trim_beats(gt_downbeats)))

print(f"{N_SONGS}곡 activation 캐싱 완료\n")

results = []
for obs_lambda in [2, 6, 10, 16]:
    for threshold in [0.0, 0.05, 0.1, 0.2]:
        dbn_beat = DBNBeatTrackingProcessor(fps=FPS, observation_lambda=obs_lambda, transition_lambda=100, threshold=threshold)
        dbn_down = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=FPS, observation_lambda=obs_lambda, transition_lambda=100, threshold=threshold)

        beat_fs, down_fs = [], []
        for beat_act, down_act, gt_beats, gt_downbeats in cache:
            try:
                est_beats = mir_eval.beat.trim_beats(dbn_beat.process_offline(beat_act))
                bs = mir_eval.beat.evaluate(gt_beats, est_beats)
                beat_fs.append(bs['F-measure'])
            except Exception:
                beat_fs.append(0.0)
            try:
                combined = np.stack([beat_act, down_act], axis=1)
                down_result = dbn_down.process(combined)
                est_downbeats = mir_eval.beat.trim_beats(down_result[down_result[:,1]==1][:,0])
                ds_ = mir_eval.beat.evaluate(gt_downbeats, est_downbeats)
                down_fs.append(ds_['F-measure'])
            except Exception:
                down_fs.append(0.0)

        beat_f_avg = np.mean(beat_fs)
        down_f_avg = np.mean(down_fs)
        results.append((obs_lambda, threshold, beat_f_avg, down_f_avg))
        print(f"obs_lambda={obs_lambda:>2} threshold={threshold:.2f} | Beat F={beat_f_avg:.3f} | Downbeat F={down_f_avg:.3f}")

print("\n=== 정렬 (Downbeat F 기준) ===")
for obs_lambda, threshold, bf, df in sorted(results, key=lambda x: -x[3]):
    print(f"obs_lambda={obs_lambda:>2} threshold={threshold:.2f} | Beat F={bf:.3f} | Downbeat F={df:.3f}")
