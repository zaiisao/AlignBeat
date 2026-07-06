#!/usr/bin/env python3
"""
retinanet_1.pt (epoch 1, 마지막으로 정상이었던 체크포인트) 진단.
GPU 1을 써서 지금 GPU 0에서 도는 학습(v4 clamp run)과 절대 안 겹치게 함.

확인할 것:
1) query 300개가 예측한 (l, r) 박스가 실제로 서로 다른 위치를 보고 있는지
   (collapse면 대부분 거의 같은 위치로 뭉쳐있을 것)
2) class 예측이 "no object"로 쏠려있는지 (즉 애초에 아무것도 안 찍는 상태인지)
3) log_var 값 확인 (clamp 범위 안에 있는지, 어느 쪽으로 치우쳤는지)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

import torch
from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater

device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES=1 이라 내부적으로는 0번

training_data_clusters = torch.tensor([0.42574675, 0.66719675, 1.24245649, 1.93286828, 2.78558922])

model = model_module.create_beatfcos_model(
    num_classes=2,
    clusters=training_data_clusters,
    args=None,
    head_type="hungarian",
    num_queries=300,
    decoder_layers=3,
    dmodel=128,
    nhead=2,
    d_hid=512,
    nlayers=9,
    attn_len=5,
    dropout=0.1,
    downbeat_weight=0.6,
    audio_downsampling_factor=512,
    centerness=False,
    postprocessing_type="soft_nms",
    audio_sample_rate=22050,
    backbone_type="wavebeat",
)

ckpt_path = "/disk1/taegum/mnt/BeatFCOS/checkpoints/retinanet_1.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device).eval()

print(f"[체크포인트] {ckpt_path} 로드 완료")
print(f"[log_var]  raw={model.set_criterion.log_vars.data.tolist()}")
clamped = torch.clamp(model.set_criterion.log_vars.data, -2.0, 2.0)
print(f"[log_var]  clamp 후=[class,bbox,giou]={clamped.tolist()}")
print(f"[implicit weight = exp(-log_var)] = {torch.exp(-clamped).tolist()}")

val_dataset = BeatDataset(
    "/disk1/taegum/mnt/labeled_data/ballroom/data",
    "/disk1/taegum/mnt/labeled_data/ballroom/label",
    dataset="ballroom",
    audio_sample_rate=22050,
    audio_downsampling_factor=512,
    subset="val",
    augment=False,
    half=True,
    preload=False,
    length=2097152,
    dry_run=False,
    spectral=True,
    validation_fold=None,
)

loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collater)

with torch.no_grad():
    for i, (audio, target, metadata) in enumerate(loader):
        audio = audio.to(device)
        C1, C2, C3 = model.encoder(audio)
        C1 = C1.transpose(1, 2); C2 = C2.transpose(1, 2); C3 = C3.transpose(1, 2)
        C1 = model.rho1(C1); C2 = model.rho2(C2); C3 = model.rho3(C3)
        feature_maps = model.fpn([C1, C2, C3])

        class_logits, boxes = model.set_prediction_head(feature_maps)
        # class_logits: (1, 300, 3), boxes: (1, 300, 2) normalized [0,1]

        probs = class_logits.softmax(-1)[0]  # (300, 3)
        scores, labels = probs[:, :-1].max(-1)  # no-object 제외하고 최고 클래스
        no_obj_prob = probs[:, -1]

        print(f"\n{'='*60}")
        print(f"샘플: {metadata[0]['Filename']}")
        print(f"{'='*60}")
        print(f"[클래스] no-object 평균 확률: {no_obj_prob.mean().item():.4f}  "
              f"(1.0에 가까우면 거의 다 '아무것도 없음'으로 예측 중)")
        print(f"[클래스] beat(1)/downbeat(0)로 분류된 query 수: "
              f"beat={int((labels==1).sum())}, downbeat={int((labels==0).sum())}, "
              f"총 query={labels.shape[0]}")

        centers = boxes[0].mean(dim=-1)  # (300,)
        unique_centers = torch.unique((centers * 1000).round() / 1000)  # 소수 3자리 반올림 후 유니크 개수
        print(f"[박스 위치] center 값 고유 개수(소수 3자리 기준): {unique_centers.numel()} / 300")
        print(f"[박스 위치] center 분포: min={centers.min().item():.4f} max={centers.max().item():.4f} "
              f"std={centers.std().item():.4f}  (0에 가까운 std면 다 뭉쳐있는 것)")

        lengths = (boxes[0][:, 1] - boxes[0][:, 0])
        print(f"[박스 길이] min={lengths.min().item():.4f} max={lengths.max().item():.4f} "
              f"mean={lengths.mean().item():.4f} std={lengths.std().item():.4f}")

        # score>0.5로 confident한 예측만 따로 확인
        confident = scores > 0.5
        print(f"[신뢰도] score>0.5인 query 수: {int(confident.sum())} / 300")
        if confident.sum() > 0:
            conf_centers = centers[confident]
            print(f"  -> 그 중 center 고유값 개수: {torch.unique((conf_centers*1000).round()/1000).numel()}")

        if i >= 2:
            break

print("\n완료.")
