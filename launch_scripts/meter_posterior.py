"""Does the head's implied meter match the annotated one?  Eq. (33) on a checkpoint.

For every validation fragment: run the E-step to get sigma, take the matched candidates'
class log-probabilities, and form P(L | x) over the candidate meters (eq. (33), uniform
prior). On downbeat-labelled fragments the annotated meter L_true is the modal gap
between downbeats, so report

  1  accuracy of argmax_L P(L | x) against L_true, and the confusion between meters;
  2  mean P(L_true | x) and mean -log P(L_true | x): the meter term's own value, so a
     checkpoint trained with --lambda_meter can be compared with one trained without;
  3  the share of fragments where P(L | x) is confident (max >= 0.9) and right / wrong.

On beat-only fragments there is no L_true; report the argmax histogram, which is what
the beat-only E-step actually feeds the class term as r_i.

No training, one forward pass over the validation set.
"""
import argparse
import collections
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alignbeat.classes import CLASS_UNKNOWN
from launch_scripts.oracle_ceiling import load

CANDIDATES = (2, 3, 4, 5, 6, 8)


@torch.no_grad()
def collect(model, loader, device, candidates):
    crit = model.subset_criterion
    crit.meter_candidates = tuple(candidates)
    crit.meter_prior = None
    rows = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        for i, target in enumerate(model._subset_targets(batch)):
            gt_c, gt_t = target["classes"], target["times"]
            M = int(gt_c.numel())
            if M < 2:
                continue
            log_p = torch.log_softmax(pred["class_logits"][i], dim=-1)
            match = crit._e_step(log_p, pred["t_hat"][i], gt_c, gt_t)
            span = log_p[torch.from_numpy(match.sigma).to(device)]
            post = crit._meter_log_posterior(span)
            if post is None:
                continue
            probs = {L: math.exp(float(v)) for L, v in post.items()}
            beat_only = bool((gt_c == CLASS_UNKNOWN).all())
            rows.append(dict(
                dataset=batch["dataset"][i] if "dataset" in batch else "?",
                M=M, beat_only=beat_only,
                L_true=0 if beat_only else int(match.meter),
                L_hat=max(probs, key=probs.get), p_max=max(probs.values()),
                p_true=(probs.get(int(match.meter), float("nan")) if not beat_only else float("nan")),
                p4=probs.get(4, 0.0)))
    return rows


def report(name, rows, candidates):
    lab = [r for r in rows if not r["beat_only"] and r["L_true"] > 1]
    unl = [r for r in rows if r["beat_only"]]
    print(f"\n{name}: {len(rows)} fragments, {len(lab)} downbeat-labelled with a meter, "
          f"{len(unl)} beat-only")

    sup = [r for r in lab if r["L_true"] in candidates]
    right = [r for r in sup if r["L_hat"] == r["L_true"]]
    print(f"\n1  argmax P(L | x) == annotated L on {len(sup)} fragments whose L is a candidate: "
          f"{100 * len(right) / max(len(sup), 1):.1f}%")
    conf = collections.Counter((r["L_true"], r["L_hat"]) for r in sup)
    trues = sorted({r["L_true"] for r in sup})
    print("   true \\ predicted " + "".join(f"{L:>6d}" for L in candidates) + "      n   acc")
    for Lt in trues:
        n = sum(1 for r in sup if r["L_true"] == Lt)
        acc = 100 * conf[(Lt, Lt)] / n
        print(f"   {Lt:>16d} " + "".join(f"{conf[(Lt, Lh)]:>6d}" for Lh in candidates)
              + f"   {n:>5d}  {acc:5.1f}%")

    p_true = np.array([r["p_true"] for r in sup])
    nll = -np.log(np.clip(p_true, 1e-12, None))
    print(f"\n2  mean P(L_true | x) = {p_true.mean():.3f};  mean -log P(L_true | x) = {nll.mean():.3f} "
          f"(median {np.median(nll):.3f})")
    for Lt in trues:
        sel = np.array([r["L_true"] == Lt for r in sup])
        print(f"   L={Lt}: mean P(L_true|x) {p_true[sel].mean():.3f}   -log {nll[sel].mean():.3f}")

    conf_right = sum(1 for r in sup if r["p_max"] >= 0.9 and r["L_hat"] == r["L_true"])
    conf_wrong = sum(1 for r in sup if r["p_max"] >= 0.9 and r["L_hat"] != r["L_true"])
    print(f"\n3  confident (max P >= 0.9): right {100 * conf_right / max(len(sup), 1):.1f}%, "
          f"wrong {100 * conf_wrong / max(len(sup), 1):.1f}%, "
          f"undecided {100 * (len(sup) - conf_right - conf_wrong) / max(len(sup), 1):.1f}%")

    if unl:
        hist = collections.Counter(r["L_hat"] for r in unl)
        print(f"\n4  beat-only fragments ({len(unl)}): argmax histogram "
              + ", ".join(f"L={L}: {100 * hist[L] / len(unl):.1f}%" for L in candidates)
              + f";  mean P(4 | x) = {np.mean([r['p4'] for r in unl]):.3f}")

    by_ds = collections.defaultdict(list)
    for r in sup:
        by_ds[r["dataset"]].append(r["L_hat"] == r["L_true"])
    print("\n5  per dataset, argmax accuracy:")
    for ds in sorted(by_ds):
        v = by_ds[ds]
        print(f"   {ds:<16s} {100 * np.mean(v):5.1f}%  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--candidates", default="2,3,4,5,6,8")
    args = ap.parse_args()
    candidates = tuple(int(v) for v in args.candidates.split(","))

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    path = sorted(glob.glob(args.checkpoint))[0]
    model = load(path, device)
    rows = collect(model, dm.val_dataloader(), device, candidates)
    report(Path(path).name, rows, candidates)


if __name__ == "__main__":
    main()
