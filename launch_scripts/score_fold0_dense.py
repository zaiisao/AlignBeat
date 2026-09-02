"""Score dense (vanilla Beat This) snapshots on held-out fold 0, beat AND downbeat.

Analyze-SMC/scripts/sa3/score_fold0_snapshots.py records beat metrics only, so the
comparison table has had no vanilla downbeat column. This is that script with the
downbeat half added, and nothing else changed: same fold-0 val_dataloader at batch_size
1, same float16 autocast, same model.postprocessor, same truth_orig_* references, same
"<3 events -> skip" rule, same model.metrics(step="test"), same per-piece averaging and
subgroups.

Pieces whose corpus has no downbeat annotation (smc, simac -- 102 of 571) are EXCLUDED
from the downbeat average rather than scored as 0. mir_eval returns 0.0 against an empty
reference, which would drag the mean by ~18% and measure the corpus rather than the
model. Beat metrics use every piece, as before.
"""
import argparse, glob, os, re, sys
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
        hung_data=False, no_val=False, fold=fold,
    )
    dm.setup(stage="fit")
    return dm.val_dataloader()


def load_model(ckpt_path, device):
    import inspect
    from beat_this.model.pl_module import PLBeatThis
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    accepted = set(inspect.signature(PLBeatThis.__init__).parameters)
    hp = {k: v for k, v in hp.items() if k in accepted}
    hp.setdefault("head_type", "dense")
    model = PLBeatThis(**hp)
    missing, _ = model.load_state_dict(ckpt["state_dict"], strict=False)
    real = [k for k in missing if "criterion" not in k]
    if real:
        print(f"    !! {len(real)} model weights absent ({real[:3]}), skipping")
        return None
    return model.eval().to(device)


@torch.no_grad()
def score(model, loader, device):
    rows = []
    for batch in loader:
        spect = batch["spect"].to(device)
        pad = batch["padding_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(spect)
        pred = {k: v.float() for k, v in pred.items()}
        pb, pdb = model.postprocessor(pred["beat"], pred["downbeat"], pad)
        if not isinstance(pb, tuple):
            pb, pdb = (pb,), (pdb,)
        for i in range(len(pb)):
            truth_b = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth_b) < 3:
                continue
            met_b = model.metrics(truth_b, pb[i], step="test")
            row = dict(path=str(batch["spect_path"][i]),
                       corpus=str(batch["spect_path"][i]).split("/", 1)[0],
                       bpm=float(60.0 / np.median(np.diff(truth_b))),
                       F=met_b["F-measure"], CMLt=met_b["CMLt"], AMLt=met_b["AMLt"])
            # downbeats only where they were annotated at all
            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            if bool(batch["downbeat_mask"][i]) and len(truth_db) >= 3:
                met_d = model.metrics(truth_db, pdb[i], step="test")
                row.update(dbF=met_d["F-measure"], dbCMLt=met_d["CMLt"],
                           dbAMLt=met_d["AMLt"])
            rows.append(row)
    return rows


def summarize(rows):
    groups = {"ALL": lambda r: True,
              "SMC": lambda r: r["corpus"] == "smc",
              "<70 bpm": lambda r: r["bpm"] < 70,
              "SMC <70": lambda r: r["corpus"] == "smc" and r["bpm"] < 70,
              ">=70 bpm": lambda r: r["bpm"] >= 70}
    for lo, hi in ((0, 70), (70, 100), (100, 130), (130, 160), (160, 1e9)):
        name = f"bpm {lo}-{hi:.0f}" if hi < 1e9 else f"bpm {lo}+"
        groups[name] = (lambda l, h: lambda r: l <= r["bpm"] < h)(lo, hi)
    for corpus in sorted({r["corpus"] for r in rows}):
        groups[f"ds:{corpus}"] = (lambda c: lambda r: r["corpus"] == c)(corpus)
    out = {}
    for name, fn in groups.items():
        sel = [r for r in rows if fn(r)]
        if not sel:
            continue
        db = [r for r in sel if "dbF" in r]
        mean = lambda key, xs: float(np.mean([x[key] for x in xs])) if xs else float("nan")
        out[name] = (len(sel), mean("F", sel), mean("CMLt", sel), mean("AMLt", sel),
                     len(db), mean("dbF", db), mean("dbCMLt", db), mean("dbAMLt", db))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="cache/fold0_dense_scores.csv")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    loader = build_loader(args.num_workers, args.fold)
    ckpts = sorted(glob.glob(args.pattern))
    print(f"  scoring {len(ckpts)} checkpoints on fold {args.fold}\n")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    fh = open(args.out, "a")
    if new:
        fh.write("arm,epoch,group,n,F,CMLt,AMLt,n_db,dbF,dbCMLt,dbAMLt\n")
    for c in ckpts:
        stem = Path(c).stem
        m = re.search(r"_ep(\d+)$|epoch=(\d+)", stem)
        epoch = int(next(g for g in (m.groups() if m else ()) if g)) if m else -1
        arm = re.sub(r"_ep\d+$", "", stem).split()[0]
        model = load_model(c, device)
        if model is None:
            continue
        for g, v in summarize(score(model, loader, device)).items():
            fh.write(f"{arm},{epoch},{g},{v[0]},{v[1]:.6f},{v[2]:.6f},{v[3]:.6f},"
                     f"{v[4]},{v[5]:.6f},{v[6]:.6f},{v[7]:.6f}\n")
            if g == "ALL":
                print(f"  {arm:>14} ep{epoch:>3}  beat F={v[1]:.4f} CMLt={v[2]:.4f} | "
                      f"downbeat F={v[5]:.4f} CMLt={v[6]:.4f}  (n={v[0]}, n_db={v[4]})")
        fh.flush(); del model; torch.cuda.empty_cache()
    fh.close(); print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
