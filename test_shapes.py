import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/taegum/mnt/BeatFCOS')

from beatfcos.beat_transformer_encoder import BeatTransformerEncoder
from beatfcos.model_module import PyramidFeatures

B, T, dmodel = 2, 500, 128

print("=== Input ===")
mel = torch.randn(B, T, 128)
print(f"mel-spec:     {mel.shape}")

print("\n=== BeatTransformerEncoder ===")
encoder = BeatTransformerEncoder(dmodel=dmodel, nhead=8, d_hid=512, nlayers=9)
with torch.no_grad():
    C1, C2, C3 = encoder(mel)
print(f"C1 (layer 2): {C1.shape}")
print(f"C2 (layer 5): {C2.shape}")
print(f"C3 (layer 8): {C3.shape}")

print("\n=== Transpose ===")
C1 = C1.transpose(1, 2)
C2 = C2.transpose(1, 2)
C3 = C3.transpose(1, 2)
print(f"C1: {C1.shape}")
print(f"C2: {C2.shape}")
print(f"C3: {C3.shape}")

print("\n=== rho Downsampling ===")
rho1 = nn.Identity()
rho2 = nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1)
rho3 = nn.Sequential(
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
    nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
)
with torch.no_grad():
    C1 = rho1(C1)
    C2 = rho2(C2)
    C3 = rho3(C3)
print(f"C1 (rho1): {C1.shape}")
print(f"C2 (rho2): {C2.shape}")
print(f"C3 (rho3): {C3.shape}")

print("\n=== FPN (PyramidFeatures) ===")
fpn = PyramidFeatures(dmodel, dmodel, dmodel)
with torch.no_grad():
    P1, P2, P3 = fpn([C1, C2, C3])
print(f"P1: {P1.shape}")
print(f"P2: {P2.shape}")
print(f"P3: {P3.shape}")

print("\nAll shapes OK!")
