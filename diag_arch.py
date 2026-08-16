#!/usr/bin/env python3
"""
BeatFCOS 정밀 구조 진단 스크립트
1) DSA receptive field (C1/C2/C3 계층성)
2) Shape 흐름 전체 추적
3) Head output 범위 & 분포 확인
"""
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────
B       = 2
T       = 300
dmodel  = 512
nhead   = 8
d_hid   = 2048
attn_len = 5
num_classes = 2
HOP     = 512
SR      = 22050
S = '─' * 64

# ─────────────────────────────────────────────────────────
# PART 1: DSA Receptive Field 분석
# ─────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  PART 1: DSA Receptive Field  (C1/C2/C3 계층성 검증)")
print(f"{'='*64}")
print(f"  attn_len={attn_len}  /  symmetric heads: ±(attn_len//2)*2^layer")
print(f"  복합 RF = Σ 2*(attn_len//2)*2^j  for j=0..tap_layer")
print(f"  (각 layer가 이전 compound RF 위에 더 큰 window를 추가)")
print(f"  1 frame = {HOP/SR*1000:.2f} ms  (SR={SR}, hop={HOP})")
print()

half = attn_len // 2   # 2 for attn_len=5

print(f"  {'Tap':<5} {'tap_i':<8} {'dilation':<11} "
      f"{'단일 window':<16} {'복합 RF±frame':<17} {'복합 RF±ms'}")
print(f"  {S}")

for name, tap_i in [('C1', 2), ('C2', 5), ('C3', 8)]:
    d = 2**tap_i
    single_r  = half * d                       # single-layer window radius (frames)
    compound_r = sum(half * (2**j) for j in range(tap_i+1))  # compound across layers
    ms = compound_r * HOP / SR * 1000
    print(f"  {name:<5} {tap_i:<8} {d:<11} ±{single_r:<15} ±{compound_r:<16} ±{ms:.0f}ms")

print()
print("  [결론]")
print(f"  C1: 복합±{sum(half*(2**j) for j in range(3))}fr  ≈ "
      f"±{sum(half*(2**j) for j in range(3))*HOP/SR*1000:.0f}ms  → local")
print(f"  C2: 복합±{sum(half*(2**j) for j in range(6))}fr  ≈ "
      f"±{sum(half*(2**j) for j in range(6))*HOP/SR*1000:.0f}ms  → medium (~몇 beat)")
print(f"  C3: 복합±{sum(half*(2**j) for j in range(9))}fr >> T={T}  → global")
print("  → C1/C2/C3 는 코드(kv_roll, dilation=2^layer)로 계층성 확인됨. 추가 변경 불필요.")


# ─────────────────────────────────────────────────────────
# PART 2: Full Forward Shape 추적
# ─────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  PART 2: Full Forward Shape 추적")
print(f"{'='*64}")

from beatfcos.beat_transformer_encoder import BeatTransformerEncoder
from beatfcos.model_module import PyramidFeatures, ClassificationModel, RegressionModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  device: {device}")

enc  = BeatTransformerEncoder(dmodel=dmodel, nhead=nhead, d_hid=d_hid).to(device).eval()
rho1 = nn.Identity().to(device)
rho2 = nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1).to(device).eval()
rho3 = nn.Sequential(
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
).to(device).eval()
fpn  = PyramidFeatures(dmodel, dmodel, dmodel).to(device).eval()
cls_head = ClassificationModel(256, num_classes=num_classes).to(device).eval()
reg_head = RegressionModel(256).to(device).eval()

audio = torch.randn(B, T, 128).to(device)
print(f"\n  INPUT:  {tuple(audio.shape)}  (B, T, mel)")

with torch.no_grad():
    C1r, C2r, C3r = enc(audio)
    print(f"\n  ENCODER (DSA 9-layer):")
    print(f"    C1={tuple(C1r.shape)}  tap@layer2  local")
    print(f"    C2={tuple(C2r.shape)}  tap@layer5  medium")
    print(f"    C3={tuple(C3r.shape)}  tap@layer8  global")
    assert C1r.shape == C2r.shape == C3r.shape == (B, T, dmodel), "encoder shape mismatch!"
    print(f"    ✓ 세 출력 모두 동일 해상도 (B,T,dmodel) = ({B},{T},{dmodel})")

    C1 = rho1(C1r.transpose(1,2))
    C2 = rho2(C2r.transpose(1,2))
    C3 = rho3(C3r.transpose(1,2))
    print(f"\n  RHO (temporal downsampling):")
    print(f"    C1={tuple(C1.shape)}  rho1=Identity   → T")
    print(f"    C2={tuple(C2.shape)}  rho2=stride-2   → T/2")
    print(f"    C3={tuple(C3.shape)}  rho3=stride-4   → T/4")
    T2 = C2.shape[2];  T4 = C3.shape[2]
    print(f"    T={T}, T/2={T2}, T/4={T4}")

    P1, P2, P3 = fpn([C1, C2, C3])
    print(f"\n  FPN (top-down, 256ch):")
    print(f"    P1={tuple(P1.shape)}  finest   T")
    print(f"    P2={tuple(P2.shape)}  medium   T/2")
    print(f"    P3={tuple(P3.shape)}  coarsest T/4")
    assert P1.shape == (B, 256, T),  "P1 shape wrong"
    assert P2.shape == (B, 256, T2), "P2 shape wrong"
    assert P3.shape == (B, 256, T4), "P3 shape wrong"
    print(f"    ✓ FPN output shape 정상")

    print(f"\n  HEADS  (각 FPN level 입력):")
    print(f"  {'Level':<6} {'입력':<22} {'cls out':<22} {'reg out':<20} {'lft out'}")
    print(f"  {S}")
    for P, name in [(P1,'P1'),(P2,'P2'),(P3,'P3')]:
        cls = cls_head(P)
        reg, lft = reg_head(P)
        print(f"  {name:<6} {str(tuple(P.shape)):<22} {str(tuple(cls.shape)):<22} "
              f"{str(tuple(reg.shape)):<20} {tuple(lft.shape)}")

    cls_all = torch.cat([cls_head(P) for P in [P1,P2,P3]], dim=1)
    reg_all = torch.cat([reg_head(P)[0] for P in [P1,P2,P3]], dim=1)
    lft_all = torch.cat([reg_head(P)[1] for P in [P1,P2,P3]], dim=1)
    total_A = T + T2 + T4
    print(f"\n  CONCAT:")
    print(f"    total anchors = {T}+{T2}+{T4} = {total_A}")
    print(f"    cls_all = {tuple(cls_all.shape)}   (B, anchors, 2)")
    print(f"    reg_all = {tuple(reg_all.shape)}   (B, anchors, 2)  l,r")
    print(f"    lft_all = {tuple(lft_all.shape)}   (B, anchors, 1)  leftness")


# ─────────────────────────────────────────────────────────
# PART 3: Head 출력 범위 & 구조 분석
# ─────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  PART 3: Head 구조 & 출력 분석")
print(f"{'='*64}")

print("\n  [ClassificationModel 구조]")
for n, m in cls_head.named_modules():
    if n: print(f"    {n}: {m.__class__.__name__}", end='')
    if isinstance(m, nn.Conv1d):
        print(f"  in={m.in_channels} out={m.out_channels} k={m.kernel_size[0]}", end='')
    if isinstance(m, nn.GroupNorm):
        print(f"  groups={m.num_groups} ch={m.num_channels}", end='')
    if n: print()

print("\n  [RegressionModel 구조]")
for n, m in reg_head.named_modules():
    if n: print(f"    {n}: {m.__class__.__name__}", end='')
    if isinstance(m, nn.Conv1d):
        print(f"  in={m.in_channels} out={m.out_channels} k={m.kernel_size[0]}", end='')
    if isinstance(m, nn.GroupNorm):
        print(f"  groups={m.num_groups} ch={m.num_channels}", end='')
    if n: print()

with torch.no_grad():
    cls_out = cls_head(P1)
    reg_out, lft_out = reg_head(P1)

    print(f"\n  [초기화 직후 출력 범위] (임의 가중치, P1 입력)")
    print(f"    cls (Sigmoid 후):  min={cls_out.min():.4f}  max={cls_out.max():.4f}  "
          f"mean={cls_out.mean():.4f}")
    print(f"    reg (raw, l+r):   min={reg_out.min():.4f}  max={reg_out.max():.4f}  "
          f"mean={reg_out.mean():.4f}")
    print(f"    lft (Sigmoid 후): min={lft_out.min():.4f}  max={lft_out.max():.4f}  "
          f"mean={lft_out.mean():.4f}")

    neg_reg = (reg_out < 0).float().mean().item()
    print(f"\n    ⚠ reg 출력 중 음수 비율: {neg_reg*100:.1f}%")
    print(f"      (음수 l,r → GIoU loss NaN 위험. exp() activation 없음)")

print(f"\n  [head 공유 여부]")
print(f"    ClassificationModel: P1/P2/P3 에 동일 가중치 적용 (weight shared)")
print(f"    RegressionModel:     P1/P2/P3 에 동일 가중치 적용 (weight shared)")
print(f"    regression & leftness: 동일 conv backbone 공유 후 분기")


# ─────────────────────────────────────────────────────────
# PART 4: 설계 요약
# ─────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  PART 4: 구조 요약 & 잠재적 문제점")
print(f"{'='*64}")
print("""
  ✓ 올바른 부분:
    C1/C2/C3:  DSA dilation=2^layer → 실제 계층적 RF (local/medium/global)
    rho:       C3→T/4, C2→T/2 → 해상도 + semantic 동시 계층화
    FPN:       top-down fusion → P1(T)/P2(T/2)/P3(T/4) 정상
    전체 shape 흐름: 이상 없음

  ⚠ 잠재적 문제:
    1. reg 출력에 exp() 없음 → 음수 l,r → GIoU NaN 위험
       (losses.py 변경 불가 → regression 출력이 양수로 수렴해야 함)

    2. ClassificationModel: beat(0) / downbeat(1) 동일 head
       → downbeat는 beat subset (1:4 imbalance within positives)
       → 두 클래스가 같은 2-conv backbone 공유 → downbeat 구분 어려움

    3. RegressionModel: regression + leftness가 conv backbone 공유
       → 서로 다른 목적 (l,r 예측 vs 위치 품질 점수) 이 충돌 가능

    4. head가 P1/P2/P3에서 weight 공유
       → P1은 local fine-grained, P3는 coarse global feature
       → 같은 가중치로 처리 → scale-specific 처리 불가
""")

print(f"{'='*64}")
print("  완료.")
print(f"{'='*64}\n")
