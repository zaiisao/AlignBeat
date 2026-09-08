"""Why does the head call a downbeat a beat, or a beat a downbeat?

On fired candidates matched (nearest, within 70 ms) to an annotated event in a fragment
with downbeat annotations, the oracle split says the DB-vs-B call alone costs ~7 points
of downbeat F. Five questions, one checkpoint, no training:

  1  confusion   DB->B and B->DB rates, and how confident the wrong calls are.
  2  phase       per fragment, are the errors scattered, or is the whole fragment's
                 downbeat pattern shifted by k beats (a phase error)?  A phase error
                 is a bar-level inference failure; scattered errors are local.
  3  position    error rate by position in the bar (beat 1 = downbeat, 2, 3, ...) and
                 by meter L; are some meters or bar positions systematically wrong?
  4  features    linear probe on the trunk features z at those candidates for DB-vs-B,
                 held out by fragment, against (a) the head's own call and (b) the
                 frozen dense head's downbeat logit at the same frame.  Says whether
                 the information reaches z and the head loses it, or never arrives.
  5  datasets    all of the above per dataset for the worst ones.
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import BACKGROUND, BEAT, DOWNBEAT
from alignbeat.dp import subset_select_dp
from launch_scripts.oracle_ceiling import load

MS = 30000.0


@torch.no_grad()
def collect(model, dense, loader, device):
    cap = {}
    h = model.model.task_heads.head.trunk.register_forward_hook(
        lambda m, i, o: cap.__setitem__("z", o.detach()))
    rows, Z = [], []
    for k, batch in enumerate(loader):
        batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
        if not bool(batch["downbeat_mask"][0]):
            continue
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
            dn = dense.model(batch["spect"]) if dense is not None else None
        pred = {kk: v.float() for kk, v in pred.items()}
        z = cap["z"][0].float()
        target = model._subset_targets(batch)[0]
        gt_t, gt_c = target["times"], target["classes"]
        M = len(gt_t)
        if M < 4:
            continue
        t_hat = pred["t_hat"][0]
        prob = F.softmax(pred["class_logits"][0], -1)
        d = (gt_t[:, None] - t_hat[None, :]).abs()
        nearest = subset_select_dp(d.cpu().numpy())
        ev = np.arange(M)
        d_near = d.cpu().numpy()[ev, nearest] * MS
        cls = gt_c.cpu().numpy()
        # position in bar: 0 for downbeat, then 1, 2, ... until the next downbeat
        pos = np.full(M, -1); L = np.full(M, 0)
        db_idx = np.where(cls == DOWNBEAT)[0]
        for a, b in zip(db_idx[:-1], db_idx[1:]):
            pos[a:b] = np.arange(b - a); L[a:b] = b - a
        dense_db = np.full(M, np.nan)
        if dn is not None:
            dbp = torch.sigmoid(dn["downbeat"][0].float()); T = dbp.shape[0]
            fr = (gt_t.cpu().numpy() * T).round().astype(int)
            for i, f in enumerate(fr):
                dense_db[i] = float(dbp[max(0, f - 3):f + 4].max())
        name = batch.get("dataset", [""])[0] if "dataset" in batch else ""
        for i in range(M):
            j = nearest[i]
            p = prob[j]
            fired = bool(p.argmax() != BACKGROUND and p.max() >= 0.2)
            rows.append(dict(
                frag=k, ds=name, i=i, cls=int(cls[i]), pos=int(pos[i]), L=int(L[i]),
                in_tol=d_near[i] <= 70.0, fired=fired,
                p_db=float(p[DOWNBEAT]), p_b=float(p[BEAT]),
                pred=int(DOWNBEAT if p[DOWNBEAT] > p[BEAT] else BEAT),
                dense_db=dense_db[i]))
            Z.append(z[j].cpu())
    h.remove()
    return rows, torch.stack(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
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
    model = load(sorted(glob.glob(args.checkpoint))[0], device)
    dense = load(args.dense, device) if os.path.exists(args.dense) else None
    rows, Z = collect(model, dense, dm.val_dataloader(), device)

    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.in_tol & df.fired].copy()
    ok["wrong"] = ok.pred != ok.cls
    db, bt = ok[ok.cls == DOWNBEAT], ok[ok.cls == BEAT]
    print(f"\n{os.path.basename(args.checkpoint)[:30]}: {len(df)} events in downbeat-annotated "
          f"fragments; {len(ok)} fired within tolerance ({len(db)} DB, {len(bt)} B)")

    print("\n1  confusion on fired matched candidates")
    print(f"   DB called B : {db.wrong.mean():.1%}   of which confident (p_B > 0.8): {(db.wrong & (db.p_b > 0.8)).sum() / max(db.wrong.sum(), 1):.0%}")
    print(f"   B called DB : {bt.wrong.mean():.1%}   of which confident (p_DB > 0.8): {(bt.wrong & (bt.p_db > 0.8)).sum() / max(bt.wrong.sum(), 1):.0%}")
    print(f"   margin |p_DB - p_B| on wrong calls: median {np.abs(ok[ok.wrong].p_db - ok[ok.wrong].p_b).median():.2f};  on right calls: {np.abs(ok[~ok.wrong].p_db - ok[~ok.wrong].p_b).median():.2f}")

    print("\n2  phase: per fragment, is the predicted downbeat pattern the true one shifted by k beats?")
    shifts = []
    for k, g in ok.groupby("frag"):
        g = g.sort_values("i")
        true = (g.cls.values == DOWNBEAT).astype(int); pr = (g.pred.values == DOWNBEAT).astype(int)
        Ls = g.L[g.L > 0]
        if len(Ls) == 0 or true.sum() < 2: continue
        Lm = int(Ls.mode().iloc[0])
        best = max(range(Lm), key=lambda s: (np.roll(true, s) == pr).mean())
        agree0 = (true == pr).mean(); agree_best = (np.roll(true, best) == pr).mean()
        shifts.append(dict(frag=k, ds=g.ds.iloc[0], L=Lm, agree0=agree0, best_shift=best, agree_best=agree_best, err=g.wrong.mean()))
    sh = pd.DataFrame(shifts)
    phase = sh[(sh.best_shift != 0) & (sh.agree_best - sh.agree0 > 0.15)]
    print(f"   fragments: {len(sh)};  with a whole-fragment phase shift (k != 0 fits >= 15 pts better): {len(phase)} = {len(phase)/len(sh):.1%}")
    print(f"   error rate in phase-shifted fragments {phase.err.mean() if len(phase) else float('nan'):.1%} vs others {sh[~sh.index.isin(phase.index)].err.mean():.1%}")
    print(f"   share of all wrong calls that sit in phase-shifted fragments: "
          f"{ok[ok.frag.isin(phase.frag)].wrong.sum() / max(ok.wrong.sum(),1):.0%}")
    print("   by meter L (fragments / phase-shifted):", {int(L): f"{(sh.L==L).sum()}/{(phase.L==L).sum()}" for L in sorted(sh.L.unique())})

    print("\n3  error rate by position in bar (0 = downbeat) and by meter")
    for Lm in (3, 4):
        s = ok[ok.L == Lm]
        if len(s) == 0: continue
        print(f"   L={Lm}: " + "  ".join(f"pos{p} {s[s.pos==p].wrong.mean():5.1%} (n={len(s[s.pos==p])})" for p in range(Lm)))
    print("   by meter: " + "  ".join(f"L={int(L)} err {ok[ok.L==L].wrong.mean():.1%} (n={len(ok[ok.L==L])})" for L in sorted(ok.L.unique()) if L > 0 and len(ok[ok.L==L]) > 100))

    print("\n4  linear probe on trunk z for DB-vs-B (held out by fragment) vs the head vs the dense head")
    idx = ok.index.values; Zo = Z[idx]; y = torch.tensor((ok.cls.values == DOWNBEAT).astype(np.float32))
    test = torch.tensor(ok.frag.values % 5 == 0); train = ~test
    W = torch.zeros(Zo.shape[1], requires_grad=True); b0 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([W, b0], lr=0.02)
    for _ in range(2000):
        loss = F.binary_cross_entropy_with_logits(Zo[train] @ W + b0, y[train]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        probe = ((Zo[test] @ W + b0) > 0).float()
    head_acc = float((torch.tensor(ok.pred.values == DOWNBEAT).float()[test] == y[test]).float().mean())
    probe_acc = float((probe == y[test]).float().mean())
    print(f"   head accuracy {head_acc:.1%}   probe accuracy {probe_acc:.1%}   (chance: always-B {1 - float(y[test].mean()):.1%})")
    if dense is not None:
        dd = ok[test.numpy()]
        dense_acc = float(((dd.dense_db >= 0.5) == (dd.cls == DOWNBEAT)).mean())
        print(f"   dense head (downbeat sigmoid >= 0.5 within +-3 frames) accuracy {dense_acc:.1%}")
        print(f"   head wrong AND dense right: {((dd.pred != dd.cls) & ((dd.dense_db >= 0.5) == (dd.cls == DOWNBEAT))).mean():.1%} of events;  "
              f"both wrong: {((dd.pred != dd.cls) & ((dd.dense_db >= 0.5) != (dd.cls == DOWNBEAT))).mean():.1%}")

    print("\n5  per dataset: DB->B, B->DB, phase-shifted fragment share")
    for ds, g in ok.groupby("ds"):
        gd, gb = g[g.cls == DOWNBEAT], g[g.cls == BEAT]
        ph = phase[phase.ds == ds]; tot = sh[sh.ds == ds]
        print(f"   {str(ds):14s} n={len(g):5d}  DB->B {gd.wrong.mean():5.1%}  B->DB {gb.wrong.mean():5.1%}  phase-shifted frags {len(ph)}/{len(tot)}")


if __name__ == "__main__":
    main()
