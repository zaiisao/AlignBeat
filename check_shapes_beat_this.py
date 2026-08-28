"""
Shape + gradient check for the beat_this backbone path (encoder -> progressive
downsample -> subset head), matching model_module.py's backbone_type="beat_this"
branch. Prints each stage's shape next to its paper notation for narration.

Run: python check_shapes_beat_this.py
GPU:  CUDA_VISIBLE_DEVICES=<idx> python check_shapes_beat_this.py   (auto-uses cuda if available)
"""
import torch
import sys
sys.path.insert(0, '/disk1/taegum/mnt/BeatFCOS')

from beatfcos.beat_this_encoder import BeatThisEncoder
from beatfcos.progressive_downsample import ProgressiveDownsample, compute_schedule
from beatfcos.subset_head import SubsetSelectionHead, SubsetCriterion

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B = 3               # batch size (kept != 2 on purpose; see note at bottom)
T = 1500            # 30s @ 50fps -- dp-short-1.pdf's own worked example
SPECT_DIM = 128
TRANSFORMER_DIM = 512
N_MIN = 100

print("=" * 70)
print("STAGE 0: input")
print("=" * 70)
print(f"device: {device}")
print(f"x: (B={B}, T={T}, mel={SPECT_DIM})   <- x_t, dp-short-1.pdf Section 1")

# ---------------------------------------------------------------------------
# 1. Encoder: Beat This Stem + FrontendBlock x3 + Concat+Linear + Transformer x6
# ---------------------------------------------------------------------------
encoder = BeatThisEncoder(spect_dim=SPECT_DIM, transformer_dim=TRANSFORMER_DIM).to(device)
encoder.eval()

x = torch.randn(B, T, SPECT_DIM, device=device)
with torch.no_grad():
    h = encoder(x)

breakpoint()  # <- 여기서 터미널이 멈추고 (Pdb) 프롬프트가 뜸

print("\n" + "=" * 70)
print("STAGE 1: BeatThisEncoder  (beat_this_encoder.py)")
print("=" * 70)
print(f"h = encoder(x)  -> {tuple(h.shape)}")
print("  <- h_t / h^(Lenc), Beat This paper Section 3.1.2 & dp-short-1.pdf Section 2")
print(f"  params: {sum(p.numel() for p in encoder.parameters()):,}  (paper: ~20M)")

# ---------------------------------------------------------------------------
# 2. Progressive downsample: eq.(1)-(3)
# ---------------------------------------------------------------------------
schedule = compute_schedule(T, N_MIN)
print("\n" + "=" * 70)
print("STAGE 2: ProgressiveDownsample  (progressive_downsample.py)")
print("=" * 70)
print(f"compute_schedule(T={T}, N_min={N_MIN})  -> {schedule}")
print(f"  <- eq.(1): T_0={T} -> ... -> T_S={schedule[-1]}, S={len(schedule)} steps")

downsample = ProgressiveDownsample(d_model=TRANSFORMER_DIM, T=T, N_min=N_MIN).to(device)
downsample.eval()
with torch.no_grad():
    z = downsample(h)
N = downsample.N
print(f"z = downsample(h)  -> {tuple(z.shape)}")
print(f"  <- z_j = g_j^(S), eq.(2)-(3), N={N}")

# ---------------------------------------------------------------------------
# 3. Subset selection head (pre-existing, unmodified)
# ---------------------------------------------------------------------------
head = SubsetSelectionHead(feature_size=TRANSFORMER_DIM, num_candidates=N, level_strides=(1,)).to(device)
head.eval()

feature_maps = [z.transpose(1, 2)]   # SubsetSelectionHead expects a list of (B,C,T_l)
with torch.no_grad():
    class_logits, t_hat = head(feature_maps)

print("\n" + "=" * 70)
print("STAGE 3: SubsetSelectionHead  (subset_head.py, unmodified)")
print("=" * 70)
print(f"class_logits = head(feature_maps)  -> {tuple(class_logits.shape)}")
print("  <- p_hat_j over {DB, B, empty}")
print(f"t_hat                               -> {tuple(t_hat.shape)}")
print("  <- t_hat_j, eq.(4) cumulative-softplus reparameterization")
print(f"t_hat[0, :5] = {[round(v, 4) for v in t_hat[0, :5].tolist()]}")
monotonic = bool((t_hat[:, 1:] > t_hat[:, :-1]).all())
print(f"strictly increasing (eq.4 guarantee): {monotonic}")

# ---------------------------------------------------------------------------
# 4. Gradient flow check: loss -> backward, across 2 optimizer steps
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 4: gradient flow (train mode, loss.backward(), 2 steps)")
print("=" * 70)
print("subset_head.py deliberately zero-initializes class_head.weight and")
print("regression_head.weight (uniform-spacing / class-prior starting point).")
print("Linear backward is dL/dx = W^T @ dL/dy, so W=0 makes dL/dx exactly 0 on")
print("step 0 -- gradient into the backbone is 0 by design until one optimizer")
print("step moves those weights off zero. Both steps are shown below.")

encoder.train(); downsample.train(); head.train()

# Minimal synthetic targets, built directly in the format SubsetCriterion.forward
# expects ({'classes': (M,) long, 'times': (M,) float in [0,1]}) -- bypasses
# intervals_to_events's interval-chain format since we're not running through a
# real dataset batch here.
targets = [
    {'classes': torch.tensor([1, 0, 1, 0], dtype=torch.long, device=device),   # BEAT, DOWNBEAT, BEAT, DOWNBEAT
     'times': torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=torch.float32, device=device)}
    for _ in range(B)
]
criterion = SubsetCriterion()

params = list(encoder.parameters()) + list(downsample.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=3e-4)   # matches train.py's subset-head default lr

checks = [
    ("encoder.frontend.linear.weight", encoder.frontend.linear.weight),
    ("downsample.steps[0].merge.weight", downsample.steps[0].merge.weight),
    ("encoder.transformer_blocks (last FF layer)",
     encoder.transformer_blocks.layers[-1][1].net[4].weight),
]

for step in range(2):
    x = torch.randn(B, T, SPECT_DIM, device=device)
    h = encoder(x)
    z = downsample(h)
    class_logits, t_hat = head([z.transpose(1, 2)])

    losses, stats = criterion(class_logits, t_hat, targets)
    total = losses['class'] + losses['time'] + losses['background']

    opt.zero_grad()
    total.backward()

    print(f"\n--- step {step} ---")
    print(f"losses: class={losses['class'].item():.4f}  time={losses['time'].item():.4f}  "
          f"background={losses['background'].item():.4f}  total={total.item():.4f}")
    for name, p in checks:
        grad_sum = 0.0 if p.grad is None else p.grad.abs().sum().item()
        print(f"  grad reaches {name}: {grad_sum > 0}  (abs sum = {grad_sum:.4f})")

    opt.step()

print("\nAll stages ran; shapes match the paper notation above. Gradient into the "
      "backbone (encoder/downsample) is 0 at step 0 by the head's own zero-init "
      "design, and nonzero from step 1 onward once the head's final layers move "
      "off zero -- both are expected, not a bug.")
print("\nNote: batch size is kept != 2 above because BeatFCOS.forward() (the real "
      "model class, not used directly in this script) infers train/eval mode from "
      "len(inputs) == 2, which coincidentally collides with a batch size of 2 if "
      "you call it with a bare tensor.")
