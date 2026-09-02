"""Score subset-head snapshots on held-out fold 0, the way the vanilla baseline was scored.

A port of Analyze-SMC/scripts/sa3/score_fold0_snapshots.py. Everything that decides a
number is kept byte-for-byte: the same fold-0 val_dataloader at batch_size 1, the same
autocast, the same truth_orig_beat reference, the same "<3 beats -> skip" rule, the same
model.metrics(step="test") call, the same per-piece averaging and subgroup definitions,
and the same arm,epoch,group,n,F,CMLt,AMLt output schema so the rows concatenate with
Analyze-SMC/cache/*.csv.

The one thing that cannot be shared is the prediction step: the vanilla script calls
model.postprocessor on the dense head's framewise activations, and the alignment head has
none. It decodes candidates instead, via the same _subset_predict_piece the training-time
test loop uses. That difference IS the experiment; everything around it is held fixed.
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_loader(num_workers, fold):
    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(
        Path("data"), batch_size=1, train_length=1500, spect_fps=50,
        num_workers=num_workers, test_dataset="gtzan",
        length_based_oversampling_factor=0.65, augmentations={},
        hung_data=False,
        no_val=False,          # keep the fold OUT of training, i.e. a real val set
        fold=fold,
    )
    dm.setup(stage="fit")
    return dm.val_dataloader()


def load_model(ckpt_path, device):
    """Rebuild from the checkpoint's own hyper_parameters, tolerating retired knobs.

    Older checkpoints carry arguments that no longer exist (predict_precision, b_scale,
    marginal, ...). Drop those rather than refusing to load. Any parameter the current
    architecture does not provide is reported loudly: strict=False would otherwise leave
    it randomly initialised and score confident nonsense.
    """
    import inspect
    from beat_this.model.pl_module import PLBeatThis
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    accepted = set(inspect.signature(PLBeatThis.__init__).parameters)
    dropped = sorted(k for k in hp if k not in accepted)
    if dropped:
        print(f"    ignoring retired hyper_parameters: {', '.join(dropped)}")
        hp = {k: v for k, v in hp.items() if k in accepted}
    model = PLBeatThis(**hp)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if "criterion" not in k]
    if real_missing:
        print(f"    !! {len(real_missing)} MODEL weights absent from this checkpoint: "
              f"{real_missing[:4]} -- architecture has changed, scores would be "
              f"meaningless. Skipping.")
        return None
    if unexpected:
        print(f"    note: {len(unexpected)} unused keys in checkpoint, e.g. {unexpected[:2]}")
    return model.eval().to(device)


@torch.no_grad()
def score(model, loader, device):
    rows = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        # The alignment head's own whole-piece decode, in place of the dense
        # postprocessor. Returns times in seconds, exactly what the metrics want.
        beats, downbeats = model._subset_decode(batch, pred)
        for i, est in enumerate(beats):
            truth = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth) < 3:
                continue
            met = model.metrics(truth, est, step="test")
            path = batch["spect_path"][i]
            corpus = str(path).split("/", 1)[0]
            ibis = np.diff(truth)
            row = dict(
                path=str(path), corpus=corpus,
                bpm=float(60.0 / np.median(ibis)),
                F=met["F-measure"], CMLt=met["CMLt"], AMLt=met["AMLt"],
            )
            # Downbeats only where they were annotated: mir_eval scores 0.0 against an
            # empty reference, and smc/simac (102 of 571 fold-0 pieces) have none, so
            # including them would measure the corpus rather than the model.
            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            if bool(batch["downbeat_mask"][i]) and len(truth_db) >= 3:
                met_d = model.metrics(truth_db, downbeats[i], step="test")
                row.update(dbF=met_d["F-measure"], dbCMLt=met_d["CMLt"],
                           dbAMLt=met_d["AMLt"])
            rows.append(row)
    return rows


def summarize(rows):
    groups = {
        "ALL": lambda r: True,
        "SMC": lambda r: r["corpus"] == "smc",
        "<70 bpm": lambda r: r["bpm"] < 70,
        "SMC <70": lambda r: r["corpus"] == "smc" and r["bpm"] < 70,
        ">=70 bpm": lambda r: r["bpm"] >= 70,
    }
    # Tempo bands, so a deficit can be attributed to a range rather than to "slow".
    for lo, hi in ((0, 70), (70, 100), (100, 130), (130, 160), (160, 1e9)):
        name = f"bpm {lo}-{hi:.0f}" if hi < 1e9 else f"bpm {lo}+"
        groups[name] = (lambda l, h: lambda r: l <= r["bpm"] < h)(lo, hi)
    # ...and per corpus, which is what a reader asks for first.
    for corpus in sorted({r["corpus"] for r in rows}):
        groups[f"ds:{corpus}"] = (lambda c: lambda r: r["corpus"] == c)(corpus)
    out = {}
    for name, fn in groups.items():
        sel = [r for r in rows if fn(r)]
        if not sel:
            continue
        db = [r for r in sel if "dbF" in r]
        mean = lambda k, xs: float(np.mean([x[k] for x in xs])) if xs else float("nan")
        out[name] = (len(sel), mean("F", sel), mean("CMLt", sel), mean("AMLt", sel),
                     len(db), mean("dbF", db), mean("dbCMLt", db), mean("dbAMLt", db))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pattern", default="checkpoints/*.ckpt")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="cache/fold0_subset_scores.csv")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    loader = build_loader(args.num_workers, args.fold)
    ckpts = sorted(glob.glob(args.pattern))
    print(f"  scoring {len(ckpts)} checkpoints on fold {args.fold}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    fh = open(args.out, "a")
    if write_header:
        fh.write("arm,epoch,group,n,F,CMLt,AMLt,n_db,dbF,dbCMLt,dbAMLt\n")

    for c in ckpts:
        stem = Path(c).stem
        m = re.search(r"epoch=(\d+)", stem)
        epoch = int(m.group(1)) if m else -1
        arm = stem.split()[0]
        model = load_model(c, device)
        if model is None:
            continue
        summary = summarize(score(model, loader, device))
        for group, v in summary.items():
            fh.write(f"{arm},{epoch},{group},{v[0]},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f},"
                     f"{v[4]},{v[5]:.6f},{v[6]:.6f},{v[7]:.6f}\n")
            if group == "ALL":
                print(f"  {arm:>12} ep{epoch:>3}  beat F={v[1]:.4f} CMLt={v[2]:.4f} | "
                      f"downbeat F={v[5]:.4f} CMLt={v[6]:.4f}  (n={v[0]}, n_db={v[4]})")
        fh.flush()
        del model
        torch.cuda.empty_cache()
    fh.close()
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
