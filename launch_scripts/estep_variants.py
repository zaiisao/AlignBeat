"""Evaluate E-step cost variants on an existing checkpoint, without training.

For each variant the DP runs over the same class logits and t_hat, and we report how
the resulting assignment compares with the nearest-candidate (time-only) one:

  off-tol      events whose assigned candidate is > 70 ms from the event
  differ       events whose pick differs from the nearest candidate
  saved/paid   on differing events, class cost saved vs time cost paid by the pick
  label->BG    events whose ASSIGNED candidate the classifier currently calls
               background, i.e. where the M-step would push a corrective gradient

Variants:
  legacy     eps-insensitive L1 / per-candidate b_j            (pre eq.(5) code)
  eq5raw     plain L1 / b, b = mean raw matched residual        (E_globalb)
  eq5excess  eps-insensitive L1 / b, b = mean excess residual   (proposed)
  gate2      eps-insensitive L1 / b_j, +inf beyond 2 x tolerance
  timeonly   class term dropped entirely
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import BACKGROUND
from alignbeat.criterion import EPS
from alignbeat.dp import subset_select_dp
from launch_scripts.oracle_ceiling import load

VARIANTS = ("legacy", "eq5raw", "eq5excess", "gate2", "gate1",
            "detr_p", "smooth_1e-2", "smooth_1e-3", "timeonly")


@torch.no_grad()
def collect(model, loader, device):
    """One pass: everything the variants need, per fragment, on CPU."""
    crit = model.subset_criterion
    frags = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        for i, target in enumerate(model._subset_targets(batch)):
            gt_t, gt_c = target["times"], target["classes"]
            M, N = len(gt_t), pred["t_hat"].shape[1]
            if M < 2 or M > N:
                continue
            logp = F.log_softmax(pred["class_logits"][i], -1)
            frags.append(dict(
                t_hat=pred["t_hat"][i].cpu().numpy(),
                gt_t=gt_t.cpu().numpy(),
                b_j=(crit.b_min + pred["b_hat"][i]).cpu().numpy(),
                class_cost=crit.class_nll(logp, gt_c).cpu().numpy(),        # (M, N)
                bg_nll=(-logp[:, BACKGROUND]).cpu().numpy(),
                argmax=pred["class_logits"][i].argmax(-1).cpu().numpy(),
            ))
    return frags, float(crit.gamma), float(crit.b_min)


def run_variant(name, frags, gamma, b_raw, b_excess):
    E = differ = off = to_bg = 0
    saved = paid = 0.0
    for f in frags:
        d = np.abs(f["gt_t"][:, None] - f["t_hat"][None, :])         # (M, N), window units
        excess = np.clip(d - EPS, 0.0, None)
        if name == "legacy":
            time_cost = excess / f["b_j"][None, :]
        elif name == "eq5raw":
            time_cost = d / b_raw
        elif name == "eq5excess":
            time_cost = excess / b_excess
        elif name == "gate2":
            time_cost = np.where(d > 2 * EPS, 1e6, excess / f["b_j"][None, :])
        elif name == "gate1":
            time_cost = np.where(d > EPS, 1e6, 0.0)
        elif name in ("detr_p", "smooth_1e-2", "smooth_1e-3"):
            time_cost = excess / f["b_j"][None, :]                 # legacy time term
        elif name == "timeonly":
            time_cost = d
        if name == "timeonly":
            cls = 0.0
        elif name == "detr_p":
            # detection convention: -p(c_i) for the class, -p(bg) for the correction
            cls = -np.exp(-f["class_cost"]) + gamma * np.exp(-f["bg_nll"])[None, :]
        elif name.startswith("smooth"):
            eps_s = float(name.split("_")[1])
            pc = (1 - eps_s) * np.exp(-f["class_cost"]) + eps_s / 3
            pb = (1 - eps_s) * np.exp(-f["bg_nll"]) + eps_s / 3
            cls = -np.log(pc) + gamma * np.log(pb)[None, :]
        else:
            cls = f["class_cost"] - gamma * f["bg_nll"][None, :]
        cost = cls + time_cost
        sigma = subset_select_dp(cost)
        nearest = subset_select_dp(d)
        M = len(sigma); ev = np.arange(M)
        E += M
        diff = sigma != nearest
        differ += int(diff.sum())
        off += int((d[ev, sigma] > EPS).sum())
        to_bg += int((f["argmax"][sigma] == BACKGROUND).sum())
        if diff.any() and name != "timeonly":
            e = ev[diff]
            saved += float((cls[e, nearest[e]] - cls[e, sigma[e]]).sum())
            paid += float((time_cost[e, sigma[e]] - time_cost[e, nearest[e]]).sum())
    n = max(differ, 1)
    return dict(off=off / E, differ=differ / E, saved=saved / n, paid=paid / n, to_bg=to_bg / E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=args.num_workers, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    frags, gamma, b_min = collect(load(sorted(glob.glob(args.checkpoint))[0], device),
                                  dm.val_dataloader(), device)

    # eq. (5) scales as an EMA would settle on them, from the nearest-candidate matching
    raw, exc = [], []
    for f in frags:
        d = np.abs(f["gt_t"][:, None] - f["t_hat"][None, :])
        r = d[np.arange(len(f["gt_t"])), subset_select_dp(d)]
        raw.append(r); exc.append(np.clip(r - EPS, 0, None))
    b_raw = max(float(np.concatenate(raw).mean()), b_min)
    b_excess = max(float(np.concatenate(exc).mean()), b_min)
    print(f"\n{len(frags)} fragments; eq.(5) b: raw {b_raw*30000:.1f} ms, "
          f"excess {b_excess*30000:.1f} ms (floor {b_min*30000:.1f} ms)")
    print(f"\n{'variant':<11}{'off-tol':>9}{'differ':>9}{'saved':>8}{'paid':>8}{'label->BG':>11}")
    for v in VARIANTS:
        r = run_variant(v, frags, gamma, b_raw, b_excess)
        print(f"{v:<11}{r['off']:>9.1%}{r['differ']:>9.1%}{r['saved']:>8.2f}{r['paid']:>8.2f}"
              f"{r['to_bg']:>11.1%}")


if __name__ == "__main__":
    main()
