"""
BeatFCOS 전체 forward shape 디버그 스크립트
실제 학습과 동일한 config로 모델을 만들고 각 단계 shape을 출력함
"""
import torch
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

# ── 학습과 동일한 config ────────────────────────────────────────
B      = 2        # batch (메모리 절약)
T      = 300      # train: 153600 / 512
dmodel = 512
nhead  = 8
d_hid  = 2048
num_classes = 2

clusters = torch.tensor([0.42574675, 0.66719675, 1.24245649, 1.93286828, 2.78558922])
audio_sr = 22050
audio_df = 512    # hop size (audio_downsampling_factor)
# ────────────────────────────────────────────────────────────────

from beatfcos.beat_transformer_encoder import BeatTransformerEncoder
from beatfcos.model_module import PyramidFeatures, ClassificationModel, RegressionModel
import torch.nn as nn

def hdr(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# ── 모델 컴포넌트 ────────────────────────────────────────────────
encoder = BeatTransformerEncoder(dmodel=dmodel, nhead=nhead, d_hid=d_hid).to(device).eval()

rho1 = nn.Identity().to(device)
rho2 = nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1).to(device).eval()
rho3 = nn.Sequential(
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
).to(device).eval()

fpn = PyramidFeatures(dmodel, dmodel, dmodel).to(device).eval()
cls_head = ClassificationModel(256, num_classes=num_classes).to(device).eval()
reg_head = RegressionModel(256).to(device).eval()

# ── 더미 입력 ────────────────────────────────────────────────────
audio = torch.randn(B, T, 128).to(device)

hdr(f"INPUT  →  (B={B}, T={T}, mel=128)")

with torch.no_grad():

    # 1. Encoder
    C1_raw, C2_raw, C3_raw = encoder(audio)
    hdr("ENCODER  (DSA 9-layer, tap C1@2 C2@5 C3@8)")
    print(f"  C1 = {tuple(C1_raw.shape)}   # (B, T, dmodel)")
    print(f"  C2 = {tuple(C2_raw.shape)}   # (B, T, dmodel)")
    print(f"  C3 = {tuple(C3_raw.shape)}   # (B, T, dmodel)")

    # 2. Transpose → conv 형태로
    C1 = C1_raw.transpose(1, 2)
    C2 = C2_raw.transpose(1, 2)
    C3 = C3_raw.transpose(1, 2)
    hdr("TRANSPOSE  (B,T,d) → (B,d,T)")
    print(f"  C1 = {tuple(C1.shape)}")
    print(f"  C2 = {tuple(C2.shape)}")
    print(f"  C3 = {tuple(C3.shape)}")

    # 3. Rho downsampling
    C1 = rho1(C1)
    C2 = rho2(C2)
    C3 = rho3(C3)
    hdr("RHO  (temporal downsampling)")
    print(f"  C1 = {tuple(C1.shape)}   rho1: identity  → T")
    print(f"  C2 = {tuple(C2.shape)}  rho2: stride-2  → T/2")
    print(f"  C3 = {tuple(C3.shape)}   rho3: stride-4  → T/4")

    # 4. FPN
    P1, P2, P3 = fpn([C1, C2, C3])
    hdr("FPN  (PyramidFeatures, top-down)")
    print(f"  P1 = {tuple(P1.shape)}   finest,   256ch")
    print(f"  P2 = {tuple(P2.shape)}  256ch")
    print(f"  P3 = {tuple(P3.shape)}   coarsest, 256ch")

    # 5. Heads
    hdr("HEADS  (입력: 각 FPN level)")
    for i, (P, name) in enumerate([(P1,'P1'), (P2,'P2'), (P3,'P3')]):
        cls_out = cls_head(P)
        reg_out, lft_out = reg_head(P)
        if i == 0:
            print(f"  {'Level':<6} {'FPN in':<20} {'cls out':<22} {'reg out':<18} {'lft out'}")
            print(f"  {'─'*90}")
        print(f"  {name:<6} {str(tuple(P.shape)):<20} {str(tuple(cls_out.shape)):<22} {str(tuple(reg_out.shape)):<18} {tuple(lft_out.shape)}")

    # 6. Concat across levels
    cls_all = torch.cat([cls_head(P) for P in [P1, P2, P3]], dim=1)
    reg_all = torch.cat([reg_head(P)[0] for P in [P1, P2, P3]], dim=1)
    lft_all = torch.cat([reg_head(P)[1] for P in [P1, P2, P3]], dim=1)
    total_anchors = P1.shape[2] + P2.shape[2] + P3.shape[2]
    hdr("CONCATENATED  (모든 pyramid level 합산)")
    print(f"  total anchor points = T + T/2 + T/4 = {P1.shape[2]} + {P2.shape[2]} + {P3.shape[2]} = {total_anchors}")
    print(f"  cls_all = {tuple(cls_all.shape)}    # (B, anchors, num_classes=2)")
    print(f"  reg_all = {tuple(reg_all.shape)}    # (B, anchors, 2)  l,r")
    print(f"  lft_all = {tuple(lft_all.shape)}    # (B, anchors, 1)  leftness")

print(f"\n{'='*55}")
print("  완료. 헤드 이전(FPN)까지 shape 이상 없음.")
print(f"{'='*55}\n")
