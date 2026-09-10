"""Measure the two premises of the resolution hypothesis on an existing checkpoint.

  (a) Are neighbouring candidates' classifier inputs near-identical? Cosine similarity
      between z_j and z_{j+k} for k = 1, 2, 4, against random pairs from the same
      fragment.
  (b) How far, in candidate cells and in ms, does the legacy E-step's pick land from
      the candidate nearest the annotation, and how often is that beyond 70 ms?

Run on N=188 (E_ctl2) and N=376 (E_bigN): if bigN's gain came from resolution, (b)
should shrink with the cell size and (a) should be lower at the same k.
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

MS = 30000.0   # window units -> ms


@torch.no_grad()
def collect(model, loader, device):
    crit = model.subset_criterion
    head = model.model.task_heads if hasattr(model.model, "task_heads") else None
    captured = {}
    # The classifier's actual input: the tensor that goes into class_head.
    handle = None
    for name, mod in model.model.named_modules():
        if name.endswith("class_head"):
            handle = mod.register_forward_hook(
                lambda m, inp, out: captured.__setitem__("z", inp[0].detach()))
    assert handle is not None, "no class_head found"

    frags = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        z_all = captured["z"].float()
        for i, target in enumerate(model._subset_targets(batch)):
            gt_t, gt_c = target["times"], target["classes"]
            M, N = len(gt_t), pred["t_hat"].shape[1]
            if M < 2 or M > N:
                continue
            logp = F.log_softmax(pred["class_logits"][i], -1)
            cost = crit.build_cost(logp, pred["t_hat"][i], gt_c, gt_t)
            frags.append(dict(
                time_scale=model.model.task_heads.downsample.time_scale(batch["spect"].shape[1]),
                z=F.normalize(z_all[i], dim=-1).cpu().numpy(),          # (N, C)
                t_hat=pred["t_hat"][i].cpu().numpy(),
                gt_t=gt_t.cpu().numpy(),
                p_event=(1.0 - logp[:, BACKGROUND].exp()).cpu().numpy(),
                sigma=subset_select_dp(cost.cpu().numpy()),
            ))
    handle.remove()
    return frags


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
    frags = collect(load(sorted(glob.glob(args.checkpoint))[0], device),
                    dm.val_dataloader(), device)
    N = frags[0]["z"].shape[0]
    print(f"\n{os.path.basename(args.checkpoint)[:40]}  N={N}  cell={MS/N:.0f} ms  "
          f"{len(frags)} fragments")

    # (a) feature similarity by candidate offset
    rng = np.random.default_rng(0)
    print("\n(a) cosine similarity of classifier inputs z_j . z_{j+k}")
    print(f"{'k':>6}{'ms':>7}{'mean':>8}{'median':>8}{'p10':>8}")
    for k in (1, 2, 4, 8):
        s = np.concatenate([(f["z"][:-k] * f["z"][k:]).sum(-1) for f in frags])
        print(f"{k:>6}{k*MS/N:>7.0f}{s.mean():>8.3f}{np.median(s):>8.3f}"
              f"{np.percentile(s, 10):>8.3f}")
    s = np.concatenate([(f["z"][rng.permutation(N)] * f["z"]).sum(-1) for f in frags])
    print(f"{'random':>6}{'':>7}{s.mean():>8.3f}{np.median(s):>8.3f}{np.percentile(s, 10):>8.3f}")

    # (b) where the legacy pick and the classifier's local argmax land
    cells, ms_off, argmax_cells, near_ms = [], [], [], []
    for f in frags:
        d = np.abs(f["gt_t"][:, None] - f["t_hat"][None, :])            # (M, N)
        nearest = subset_select_dp(d)
        ev = np.arange(len(nearest))
        cells.append(f["sigma"] - nearest)
        ms_off.append(d[ev, f["sigma"]] * MS)
        near_ms.append(d[ev, nearest] * MS)
        # local argmax of p(event) within +-4 cells of the nearest candidate
        for e, n in enumerate(nearest):
            lo, hi = max(0, n - 4), min(N, n + 5)
            argmax_cells.append(lo + int(np.argmax(f["p_event"][lo:hi])) - n)
    cells = np.concatenate(cells); ms_off = np.concatenate(ms_off)
    near_ms = np.concatenate(near_ms); argmax_cells = np.array(argmax_cells)
    E = len(cells)
    print(f"\n(b) {E} events; nearest candidate is {near_ms.mean():.0f} ms from the "
          f"annotation on average, {np.mean(near_ms > 70):.1%} beyond 70 ms")
    print(f"    legacy pick == nearest: {np.mean(cells == 0):.1%}; "
          f"|offset| = 1 cell: {np.mean(np.abs(cells) == 1):.1%}; "
          f"2: {np.mean(np.abs(cells) == 2):.1%}; >=3: {np.mean(np.abs(cells) >= 3):.1%}")
    print(f"    legacy pick beyond 70 ms: {np.mean(ms_off > 70):.1%}   "
          f"(when pick != nearest: {np.mean(ms_off[cells != 0] > 70):.1%})")

    # (c) does the regression move the clock toward the onset? Initial grid is
    # (j+1)/N by eq. (1) over the PADDED candidate window, which SubsetHead rescales by
    # padded_length / input_frames (1504/1500 at N=188) before anyone reads t_hat. Build
    # the init the same way, or the rescale shows up as a 0 -> +80 ms ramp across the
    # window and reads as a learned drift (it did, for one afternoon).
    init = (np.arange(N) + 1.0) / N * frags[0]["time_scale"]
    shift, need, d_init, d_hat = [], [], [], []
    for f in frags:
        d = np.abs(f["gt_t"][:, None] - f["t_hat"][None, :])
        n = subset_select_dp(d)
        shift.append(f["t_hat"][n] - init[n]); need.append(f["gt_t"] - init[n])
        d_init.append(np.abs(f["gt_t"] - init[n])); d_hat.append(d[np.arange(len(n)), n])
    shift, need = np.concatenate(shift) * MS, np.concatenate(need) * MS
    d_init, d_hat = np.concatenate(d_init) * MS, np.concatenate(d_hat) * MS
    all_shift = np.concatenate([f["t_hat"] - init for f in frags]) * MS
    slope = np.polyfit(need, shift, 1)[0]
    print(f"\n(c) regression on the matched (nearest-by-t_hat) candidate, ms")
    print(f"    all candidates: t_hat - init  mean {all_shift.mean():+.1f}  "
          f"std {all_shift.std():.1f}  p5 {np.percentile(all_shift,5):+.1f}  "
          f"p95 {np.percentile(all_shift,95):+.1f}")
    print(f"    needed shift (onset - init): std {need.std():.1f};  learned shift std {shift.std():.1f}")
    print(f"    corr(learned, needed) = {np.corrcoef(need, shift)[0,1]:.2f}, slope = {slope:.2f}")
    print(f"    |onset - clock|: at init {d_init.mean():.1f} ms  ->  learned {d_hat.mean():.1f} ms; "
          f"beyond 70 ms: init {np.mean(d_init > 70):.1%} -> learned {np.mean(d_hat > 70):.1%}")


if __name__ == "__main__":
    main()
