"""Why does the head call background on candidates that sit on a beat?

A miss here is an annotated event whose nearest candidate (by t_hat) is within 70 ms
and whose class argmax is BACKGROUND -- the oracle split's C stage, the largest beat
loss. Four questions, each separating one cause, on one checkpoint (plus earlier
snapshots of the same arm for the label history):

  1  threshold   p(event) on the missed candidate: just under decode's threshold, or
                 confidently background?
  2  labels      was the missed candidate ever the E-step's pick for its event, at
                 epochs 4 / 9 / 19?  Never labelled vs lost.  Split by tempo, with the
                 background weight gamma (N-M)/M the fragment trained under.
  3  features    does the frozen encoder's own dense head fire at that onset?  If it
                 does, the information reaches the encoder output and the subset head
                 loses it; if not, no head fix helps.
  4  geometry    where in its 160 ms cell the onset falls, missed vs hit.
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

MS = 30000.0
FPS = 50.0


def tempo_bpm(gt_t):
    ibi = np.diff(np.sort(gt_t)) * 30.0
    return 60.0 / np.median(ibi) if len(ibi) else float("nan")


def tempo_bin(bpm):
    for hi, name in ((70, "<70"), (100, "70-100"), (130, "100-130"), (160, "130-160")):
        if bpm < hi:
            return name
    return "160+"


@torch.no_grad()
def collect(model, dense, loader, device, snapshots):
    """Per event: nearest candidate, its p(event), decode fire, label history, dense fire."""
    crit = model.subset_criterion
    rows = []
    for k, batch in enumerate(loader):
        batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
            dense_out = dense.model(batch["spect"]) if dense is not None else None
        pred = {kk: v.float() for kk, v in pred.items()}
        target = model._subset_targets(batch)[0]
        gt_t, gt_c = target["times"], target["classes"]
        M = len(gt_t)
        if M < 2:
            continue
        t_hat = pred["t_hat"][0]
        N = len(t_hat)
        prob = F.softmax(pred["class_logits"][0], -1)
        p_event = 1.0 - prob[:, BACKGROUND]
        argmax = prob.argmax(-1)
        d = (gt_t[:, None] - t_hat[None, :]).abs()
        nearest = subset_select_dp(d.cpu().numpy())
        ev = np.arange(M)
        d_near = d.cpu().numpy()[ev, nearest] * MS

        # label history: was `nearest` the E-step's pick at each snapshot?
        labelled = {}
        for name, snap in snapshots.items():
            with torch.autocast("cuda", dtype=torch.float16):
                sp = snap.model(batch["spect"])
            sp = {kk: v.float() for kk, v in sp.items()}
            logp = F.log_softmax(sp["class_logits"][0], -1)
            sigma = subset_select_dp(snap.subset_criterion.build_cost(
                logp, sp["t_hat"][0], gt_c, gt_t).cpu().numpy())
            labelled[name] = sigma == nearest

        # dense head: sigmoid of the frozen encoder's own beat logit, max within +-3
        # frames of the onset (the dense metric's own tolerance is +-70 ms = 3.5 frames)
        dense_fire = np.full(M, np.nan)
        if dense_out is not None:
            beat = torch.sigmoid(dense_out["beat"][0].float())
            T = beat.shape[0]
            frames = (gt_t.cpu().numpy() * T).round().astype(int)
            for i, f in enumerate(frames):
                lo, hi = max(0, f - 3), min(T, f + 4)
                dense_fire[i] = float(beat[lo:hi].max())

        bpm = tempo_bpm(gt_t.cpu().numpy())
        frame = gt_t.cpu().numpy() * 1504.0
        for i in range(M):
            n = nearest[i]
            rows.append(dict(
                frag=k, bpm=bpm, tbin=tempo_bin(bpm), M=M, N=N,
                bg_weight=float(crit.gamma) * (N - M) / M,
                near_ms=d_near[i], in_tol=d_near[i] <= 70.0,
                p_event=float(p_event[n]), argmax_bg=bool(argmax[n] == BACKGROUND),
                fires=bool(argmax[n] != BACKGROUND and prob[n].max() >= 0.2),
                dense=dense_fire[i], cell_pos=(frame[i] % 8.0) / 8.0,
                **{f"lab_{nm}": bool(v[i]) for nm, v in labelled.items()},
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="e.g. E_bhat")
    ap.add_argument("--epochs", default="004,009,019")
    ap.add_argument("--dense", default="../Analyze-SMC/third-party/beat_this/checkpoints/"
                    "vanilla_f0 S0 fold0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    epochs = args.epochs.split(",")
    snaps = {e: load(sorted(glob.glob(f"checkpoints/{args.arm} *epoch={e}.ckpt"))[0], device)
             for e in epochs}
    final = snaps[epochs[-1]]
    dense = load(args.dense, device) if os.path.exists(args.dense) else None
    rows = collect(final, dense, dm.val_dataloader(), device, snaps)

    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.in_tol]                       # events the timing side has delivered
    miss = ok[~ok.fires]
    hit = ok[ok.fires]
    print(f"\n{args.arm} ep{epochs[-1]}: {len(df)} events; {len(ok)} with a candidate inside "
          f"70 ms; of those {len(miss)} ({len(miss)/len(ok):.1%}) do not fire  <- the C loss")

    print("\n1  p(event) on the missed candidate")
    edges = [0, .01, .05, .1, .15, .2, 1.0]
    h = np.histogram(miss.p_event, bins=edges)[0] / max(len(miss), 1)
    for lo, hi, v in zip(edges[:-1], edges[1:], h):
        print(f"   [{lo:.2f},{hi:.2f}): {v:5.1%}")
    print(f"   argmax already non-background but score < 0.2: "
          f"{(~miss.argmax_bg).mean():.1%} of misses")
    print(f"   hits, for reference: p(event) median {hit.p_event.median():.2f}, p10 {hit.p_event.quantile(.1):.2f}")

    print("\n2  label history of the missed candidate (was it the E-step's pick?)")
    labs = [f"lab_{e}" for e in epochs]
    for name, sub in (("missed", miss), ("hit", hit)):
        never = (~sub[labs]).all(axis=1).mean()
        always = sub[labs].all(axis=1).mean()
        print(f"   {name:6s}: " + "  ".join(f"ep{e} {sub[l].mean():5.1%}" for e, l in zip(epochs, labs))
              + f"   never {never:5.1%}  always {always:5.1%}")
    print("   by tempo:  miss rate | labelled at final epoch (missed) | bg weight gamma(N-M)/M")
    for tb in ("<70", "70-100", "100-130", "130-160", "160+"):
        s = ok[ok.tbin == tb]; m = s[~s.fires]
        if len(s) == 0: continue
        print(f"   {tb:8s} n={len(s):6d}  miss {len(m)/len(s):5.1%} | {m[labs[-1]].mean() if len(m) else float('nan'):5.1%} | {s.bg_weight.mean():4.1f}")

    if dense is not None:
        print("\n3  the frozen encoder's own dense head at the onset (max sigmoid within +-3 frames)")
        for name, sub in (("missed", miss), ("hit", hit)):
            print(f"   {name:6s}: dense >= 0.5 for {(sub.dense >= 0.5).mean():5.1%};  median {sub.dense.median():.2f}")
        print(f"   missed AND dense >= 0.5 (encoder sees it, subset head drops it): "
              f"{((miss.dense >= 0.5)).sum()} = {(miss.dense >= 0.5).mean():.1%} of misses, "
              f"{(miss.dense >= 0.5).sum()/len(ok):.1%} of events")

    print("\n4  onset position within its 160 ms cell (0 = cell start, 1 = end)")
    edges = np.linspace(0, 1, 5)
    for name, sub in (("missed", miss), ("hit", hit)):
        h = np.histogram(sub.cell_pos, bins=edges)[0] / max(len(sub), 1)
        print(f"   {name:6s}: " + "  ".join(f"{v:5.1%}" for v in h))
    print(f"   miss rate by quarter: " + "  ".join(
        f"{(~ok[(ok.cell_pos >= lo) & (ok.cell_pos < hi)].fires).mean():5.1%}"
        for lo, hi in zip(edges[:-1], edges[1:])))


if __name__ == "__main__":
    main()
