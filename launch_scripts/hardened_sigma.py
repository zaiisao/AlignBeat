"""Section 4.2's condition: is the hardened sigma actually a reasonable simplification?

The E-step resolves sigma once, by minimum-cost matching, rather than marginalising
over it. Section 4.1 is explicit that this is a CHOICE, not a tractability necessity,
and that the resulting point mass is not in general a tight bound -- the PDF's own
diffuse example loses 2.02 nats. Section 4.2 then gives the condition under which the
choice is safe, as a directly computable quantity:

    m(theta, x) = P_1(sigma_hat | theta, x)
                = Laplace product at sigma_hat / sum over ALL sigma of the same

"When m(theta, x) -> 1 the point estimate discards almost nothing." The stated failure
mode is "several candidates cluster near the same true event", which is worth checking
rather than assuming: we run N candidates against M events with N/M around 3.

The gap this costs the surrogate is exactly -log m, so both are reported. Timing
evidence ONLY -- P_1 is defined under the timing term alone, so the class NLL that
build_cost adds for the E-step's own purposes is excluded here.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignbeat.dp import subset_select_dp, subset_select_logz
from launch_scripts.oracle_ceiling import load


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    model = load(sorted(glob.glob(args.checkpoint))[0], device)
    crit = model.subset_criterion
    loader = dm.train_dataloader() if args.split == "train" else dm.val_dataloader()

    rows = []
    for batch in loader:
        if len(rows) >= args.limit:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        for i, target in enumerate(model._subset_targets(batch)):
            gt_t = target["times"]
            M = int(gt_t.numel())
            if M < 2 or M > pred["t_hat"].shape[1]:
                continue
            # Timing evidence alone: eq. (1)'s Laplace, in nats, is lambda_L1 * |dt|.
            cost = crit.l1(pred["t_hat"][i][None, :], gt_t[:, None]).double().cpu().numpy()
            sigma = subset_select_dp(cost)
            best = float(cost[np.arange(M), sigma].sum())
            log_m = -best - subset_select_logz(cost)
            rows.append((float(np.exp(log_m)), -log_m, M,
                         batch["dataset"][i] if "dataset" in batch else "?"))

    m = np.array([r[0] for r in rows])
    gap = np.array([r[1] for r in rows])
    print(f"\n  {args.split}, {len(rows)} fragments, b = 1/lambda_L1 = "
          f"{1.0 / crit.lambda_l1:.6f} window units")
    print(f"\n  m(theta, x):  mean {m.mean():.4f}   median {np.median(m):.4f}"
          f"   min {m.min():.4f}   max {m.max():.4f}")
    print(f"  gap = -log m: mean {gap.mean():.4f} nats   median {np.median(gap):.4f}"
          f"   max {gap.max():.4f}")
    print(f"\n  {'m band':>16}{'share':>9}{'mean gap (nats)':>18}")
    for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
        sel = (m >= lo) & (m < hi)
        if sel.any():
            print(f"  [{lo:.2f}, {hi:.2f}){sel.mean():>13.3f}{gap[sel].mean():>18.4f}")
    print(f"\n  {'dataset':<16}{'n':>5}{'mean m':>9}{'mean gap':>10}")
    for ds in sorted({r[3] for r in rows}):
        sel = [r for r in rows if r[3] == ds]
        print(f"  {ds:<16}{len(sel):>5}{np.mean([r[0] for r in sel]):>9.4f}"
              f"{np.mean([r[1] for r in sel]):>10.4f}")


if __name__ == "__main__":
    main()
