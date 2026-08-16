import torch
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

from beatfcos.beat_transformer_encoder import BeatTransformerEncoder
from beatfcos.model_module import PyramidFeatures

B = 2       # batch size (작게)
T = 300     # train_length / hop = 153600 / 512
dmodel = 512
nhead = 8
d_hid = 2048

print("=" * 60)
print(f"Input: (B={B}, T={T}, mel=128)")

# --- Encoder ---
encoder = BeatTransformerEncoder(dmodel=dmodel, nhead=nhead, d_hid=d_hid)
encoder.eval()

audio_batch = torch.randn(B, T, 128)
print(f"\n[Encoder input]  {tuple(audio_batch.shape)}")

with torch.no_grad():
    C1, C2, C3 = encoder(audio_batch)

print(f"[Encoder output] C1={tuple(C1.shape)}  (tap at layer 2, T)")
print(f"                 C2={tuple(C2.shape)}  (tap at layer 5, T)")
print(f"                 C3={tuple(C3.shape)}  (tap at layer 8, T)")

# --- transpose (B,T,d) → (B,d,T) for conv ---
C1 = C1.transpose(1, 2)
C2 = C2.transpose(1, 2)
C3 = C3.transpose(1, 2)
print(f"\n[After transpose] C1={tuple(C1.shape)}, C2={tuple(C2.shape)}, C3={tuple(C3.shape)}")

# --- rho layers ---
import torch.nn as nn
rho1 = nn.Identity()
rho2 = nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1)
rho3 = nn.Sequential(
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1)
)

with torch.no_grad():
    C1 = rho1(C1)
    C2 = rho2(C2)
    C3 = rho3(C3)

print(f"\n[After rho]  C1={tuple(C1.shape)}  stride=1 → T")
print(f"             C2={tuple(C2.shape)}  stride=2 → T/2")
print(f"             C3={tuple(C3.shape)}  stride=4 → T/4")

expected_T   = T
expected_T2  = (T + 1) // 2
expected_T4  = ((T + 1) // 2 + 1) // 2

ok1 = C1.shape == (B, dmodel, expected_T)
ok2 = C2.shape[2] == expected_T2
ok3 = C3.shape[2] == expected_T4
print(f"\n  C1 shape OK: {ok1}  (expected T={expected_T})")
print(f"  C2 shape OK: {ok2}  (expected T/2={expected_T2})")
print(f"  C3 shape OK: {ok3}  (expected T/4={expected_T4})")

# --- FPN ---
fpn = PyramidFeatures(dmodel, dmodel, dmodel)
fpn.eval()

with torch.no_grad():
    P1, P2, P3 = fpn([C1, C2, C3])

print(f"\n[FPN output]  P1={tuple(P1.shape)}  (finest, T,   256ch)")
print(f"              P2={tuple(P2.shape)}  (T/2,  256ch)")
print(f"              P3={tuple(P3.shape)}  (coarsest, T/4, 256ch)")

ok_P1 = P1.shape == (B, 256, C1.shape[2])
ok_P2 = P2.shape == (B, 256, C2.shape[2])
ok_P3 = P3.shape == (B, 256, C3.shape[2])
print(f"\n  P1 shape OK: {ok_P1}")
print(f"  P2 shape OK: {ok_P2}")
print(f"  P3 shape OK: {ok_P3}")

print("\n" + "=" * 60)
if all([ok1, ok2, ok3, ok_P1, ok_P2, ok_P3]):
    print("✓ All shapes correct up to FPN output.")
    print("  Heads receive: (B, 256, T), (B, 256, T/2), (B, 256, T/4)")
else:
    print("✗ Shape mismatch found!")
print("=" * 60)
