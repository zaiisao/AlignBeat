#!/usr/bin/env python3
"""
regression head에 phase(위상/주기성) auxiliary target을 추가하면서 새로
생긴 파라미터(regressionModel.phase.*)는 기존 체크포인트에 없으므로, 새
모델을 만들어 놓고(이 파라미터는 BeatFCOS.__init__에서 이미 0으로
초기화됨) 나머지는 기존 체크포인트에서 그대로 불러와 warm-start용
체크포인트를 만든다. nhead=2 유지 이유: nhead=8 수정이 성능에 유의미한
차이를 안 만들어서(이번 세션 앞서 확인), 굳이 두 실험을 겹치지 않기
위해 검증된 nhead=2 levelfix 체크포인트를 그대로 기반으로 삼음.
"""
import sys
sys.path.insert(0, '.')
import torch
from beatfcos import model_module

SRC_CHECKPOINT = "checkpoints_fold0_levelfix_test/retinanet_82.pt"
DST_DIR = "checkpoints_fold0_phase_target_test"
DST_EPOCH = 83  # 이어서 학습하는 것처럼 보이게 다음 epoch 번호로 저장

import os
os.makedirs(DST_DIR, exist_ok=True)

clusters = torch.tensor([0.42574675, 0.66719675, 1.93286828])
model = model_module.create_beatfcos_model(
    num_classes=2, clusters=clusters, args=None, head_type="fcos",
    dmodel=128, nhead=2, d_hid=512, nlayers=9, attn_len=5, dropout=0.1,
    downbeat_weight=0.6, beat_radius=2.5, downbeat_radius=4.5, phase_weight=1.0,
    audio_downsampling_factor=512, centerness=False, postprocessing_type="soft_nms",
    audio_sample_rate=22050, backbone_type="wavebeat",
)
new_state_dict = model.state_dict()

old_state_dict = torch.load(SRC_CHECKPOINT, map_location="cpu", weights_only=False)
old_state_dict = {k.replace("module.", "", 1): v for k, v in old_state_dict.items()}

reused, kept_new = 0, 0
for k in new_state_dict:
    if k in old_state_dict and old_state_dict[k].shape == new_state_dict[k].shape:
        new_state_dict[k] = old_state_dict[k]
        reused += 1
    else:
        kept_new += 1
        print(f"새로 초기화된 채로 유지: {k} (shape={tuple(new_state_dict[k].shape)})")

model.load_state_dict(new_state_dict)

# train.py는 DataParallel로 감싼 state_dict(모든 키에 "module." 접두어)를
# strict 모드로 로드하므로 그 형식에 맞춰 저장.
wrapped = torch.nn.DataParallel(model)
save_path = os.path.join(DST_DIR, f"retinanet_{DST_EPOCH}.pt")
torch.save(wrapped.state_dict(), save_path)

print(f"\n총 {len(new_state_dict)}개 파라미터 중 {reused}개 재사용, {kept_new}개 새로 초기화")
print(f"저장 완료: {save_path}")
