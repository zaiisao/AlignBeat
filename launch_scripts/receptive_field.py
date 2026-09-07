"""Where does candidate j's fire decision actually come from, and does that agree with t_hat?

For each candidate j: gradient of log p_j(event) with respect to the frozen encoder's
output frames (T, d), collapsed to |grad| per frame. From it: the centroid frame, the
spread, and the mass inside the nominal cell [8j, 8j+8). Compared with t_hat_j * T and
with the nominal centre 8j + 3.5.

Then per annotated event: which candidate is nearest by t_hat, by receptive-field
centroid, by nominal cell -- and which one the legacy E-step picked.
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import BACKGROUND, BEAT, DOWNBEAT
from alignbeat.criterion import EPS
from alignbeat.dp import subset_select_dp
from launch_scripts.oracle_ceiling import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fragments", type=int, default=30)
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    model = load(sorted(glob.glob(args.checkpoint))[0], device)
    net, crit = model.model, model.subset_criterion

    cent_rel, spread, in_cell, that_rel = [], [], [], []
    agree = dict(t_hat=0, rf=0, cell=0, rf_when_t_differs=0, cell_when_t_differs=0,
                 t_differs=0, events=0, off_tol=0, off_tol_rf_nearest=0)
    torch.manual_seed(0)
    done = 0
    for batch in dm.val_dataloader():
        if done >= args.fragments:
            break
        spect = batch["spect"].to(device).float()
        with torch.no_grad():
            x = net.transformer_blocks(net.frontend(spect)).float()      # (1, T, d)
        T = x.shape[1]
        x = x.detach().requires_grad_(True)
        out = net.task_heads(x)
        logits, t_hat, b_hat = out["class_logits"][0], out["t_hat"][0], out["b_hat"][0]
        N = logits.shape[0]
        stride = T / N
        logp = F.log_softmax(logits, -1)
        log_event = torch.logsumexp(logp[:, [BEAT, DOWNBEAT]], -1)         # (N,)

        centroid = np.zeros(N)
        for j in range(N):
            g, = torch.autograd.grad(log_event[j], x, retain_graph=True)
            w = g[0].abs().sum(-1)                                         # (T,)
            w = w / w.sum().clamp_min(1e-12)
            frames = torch.arange(T, device=device, dtype=w.dtype)
            c = float((w * frames).sum())
            centroid[j] = c
            nominal = stride * j + stride / 2
            cent_rel.append((c - nominal) / stride)
            spread.append(float(torch.sqrt((w * (frames - c) ** 2).sum())) / stride)
            lo, hi = int(stride * j), int(stride * (j + 1))
            in_cell.append(float(w[lo:hi].sum()))
            that_rel.append((float(t_hat[j]) * T - nominal) / stride)

        target = model._subset_targets(batch)[0]
        gt_t, gt_c = target["times"].to(device), target["classes"].to(device)
        M = len(gt_t)
        if 2 <= M <= N:
            with torch.no_grad():
                cost = crit.build_cost(logp, t_hat, gt_c, gt_t)
            sigma = subset_select_dp(cost.cpu().numpy())
            f = gt_t.cpu().numpy() * T                                     # onset frames
            th = t_hat.detach().cpu().numpy() * T
            d_t = np.abs(f[:, None] - th[None, :])
            n_t = subset_select_dp(d_t)
            n_rf = np.abs(f[:, None] - centroid[None, :]).argmin(1)
            n_cell = np.clip((f // stride).astype(int), 0, N - 1)
            ev = np.arange(M)
            agree["events"] += M
            agree["t_hat"] += int((sigma == n_t).sum())
            agree["rf"] += int((sigma == n_rf).sum())
            agree["cell"] += int((sigma == n_cell).sum())
            diff = sigma != n_t
            agree["t_differs"] += int(diff.sum())
            agree["rf_when_t_differs"] += int((sigma[diff] == n_rf[diff]).sum())
            agree["cell_when_t_differs"] += int((sigma[diff] == n_cell[diff]).sum())
            off = d_t[ev, sigma] > EPS * T
            agree["off_tol"] += int(off.sum())
            agree["off_tol_rf_nearest"] += int((sigma[off] == n_rf[off]).sum())
        done += 1

    cent_rel, spread = np.array(cent_rel), np.array(spread)
    in_cell, that_rel = np.array(in_cell), np.array(that_rel)
    print(f"\n{os.path.basename(args.checkpoint)[:30]}  N={N}  cell={stride:.0f} frames  "
          f"{done} fragments, {len(cent_rel)} candidates")
    print("\nreceptive field of log p_j(event), in cells relative to nominal centre 8j+3.5")
    print(f"  centroid offset : mean {cent_rel.mean():+.2f}  median {np.median(cent_rel):+.2f}"
          f"  p10 {np.percentile(cent_rel,10):+.2f}  p90 {np.percentile(cent_rel,90):+.2f}")
    print(f"  spread (std)    : mean {spread.mean():.2f}  median {np.median(spread):.2f}")
    print(f"  mass in own cell: mean {in_cell.mean():.2f}  median {np.median(in_cell):.2f}")
    print(f"  t_hat offset    : mean {that_rel.mean():+.2f}  median {np.median(that_rel):+.2f}"
          f"  p10 {np.percentile(that_rel,10):+.2f}  p90 {np.percentile(that_rel,90):+.2f}")
    print(f"  corr(centroid offset, t_hat offset) = {np.corrcoef(cent_rel, that_rel)[0,1]:.2f}")
    E = agree["events"]; D = max(agree["t_differs"], 1); O = max(agree["off_tol"], 1)
    print(f"\n{E} events: legacy pick == nearest by  t_hat {agree['t_hat']/E:.1%}  "
          f"RF-centroid {agree['rf']/E:.1%}  nominal cell {agree['cell']/E:.1%}")
    print(f"  when pick != nearest-by-t_hat ({agree['t_differs']/E:.1%}): "
          f"pick == RF-nearest {agree['rf_when_t_differs']/D:.1%}, "
          f"== cell {agree['cell_when_t_differs']/D:.1%}")
    print(f"  off-tolerance picks ({agree['off_tol']/E:.1%}): "
          f"RF-nearest {agree['off_tol_rf_nearest']/O:.1%}")


if __name__ == "__main__":
    main()
