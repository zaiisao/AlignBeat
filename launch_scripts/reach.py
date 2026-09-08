"""Reach vs need, and neighbour co-movement, of the clocks on one checkpoint.

Per matched (nearest) candidate: how far its clock needed to move from the initial
grid to reach the beat, and how far it did, binned by need. Per clock that moved more
than 30 ms: the signed shift of its two neighbours along the mover's direction.
The cumsum form of eq. (1) saturates at ~68 ms and taxes the next neighbour ~17 ms
(corr -0.61, T_aligned); a bounded per-candidate offset should track the need up to
its cap and leave neighbours at 0.
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.dp import subset_select_dp
from alignbeat.head import monotonic_times
from launch_scripts.oracle_ceiling import load
MS = 30000.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50, num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={}, hung_data=False, no_val=False, fold=a.fold)
    dm.setup(stage="fit"); dev = f"cuda:{a.gpu}"
    m = load(sorted(glob.glob(a.checkpoint))[0], dev)
    ts = m.model.task_heads.downsample.time_scale(1500)
    need, learn, own, prv, nxt = [], [], [], [], []
    with torch.no_grad():
        for b in dm.val_dataloader():
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16): p = m.model(b["spect"])
            th = p["t_hat"][0].float(); N = len(th)
            init = monotonic_times(torch.zeros(1, N, device=dev))[0] * ts      # the head's own r=0 grid
            gt = m._subset_targets(b)[0]["times"]
            if len(gt) < 2: continue
            n = subset_select_dp((gt[:, None] - th[None, :]).abs().cpu().numpy())
            sh = ((th - init) * MS).cpu().numpy()
            for i, j in enumerate(n):
                need.append(float((gt[i] - init[j]) * MS)); learn.append(sh[j])
                if 0 < j < N - 1: own.append(sh[j]); prv.append(sh[j - 1]); nxt.append(sh[j + 1])
    need, learn, own, prv, nxt = map(np.array, (need, learn, own, prv, nxt))
    print(f"\n{os.path.basename(a.checkpoint)[:30]}: reach vs need on {len(need)} matched candidates")
    print(f"   {'need':>8} {'n':>6} {'mean need':>10} {'learned':>8} {'ratio':>6}")
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 140), (140, 200)):
        k = (np.abs(need) >= lo) & (np.abs(need) < hi)
        if k.sum() < 50: continue
        proj = learn[k] * np.sign(need[k])
        print(f"   {f'{lo}-{hi}':>8} {k.sum():6d} {np.abs(need[k]).mean():10.1f} {proj.mean():8.1f} {proj.mean() / np.abs(need[k]).mean():6.2f}")
    big = np.abs(own) > 30
    print(f"   neighbours of clocks moved > 30 ms (n={big.sum()}): prev {np.mean(prv[big] * np.sign(own[big])):+.1f} ms (corr {np.corrcoef(own, prv)[0, 1]:+.2f}),"
          f" next {np.mean(nxt[big] * np.sign(own[big])):+.1f} ms (corr {np.corrcoef(own, nxt)[0, 1]:+.2f})")
    print(f"   learned shift over all candidates: mean {learn.mean():+.1f}, p5 {np.percentile(learn, 5):+.1f}, p95 {np.percentile(learn, 95):+.1f} ms")

if __name__ == "__main__":
    main()
