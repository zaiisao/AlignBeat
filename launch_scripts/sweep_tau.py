"""Sweep the decode thresholds tau_beat and tau_downbeat on a trained checkpoint.

Both thresholds act only inside decode_events, downstream of the encoder and the head,
so a sweep needs exactly one forward pass over the fold: cache (class_logits, t_hat)
per piece, then re-decode the cache at each (tau_beat, tau_downbeat) on the grid. That
turns an O(grid) GPU job into one pass plus cheap CPU decodes.

Scoring is otherwise byte-identical to score_fold0_subset.py -- same loader, same
autocast, same "<3 beats -> skip" rule, same downbeat-annotated-only restriction -- so
a row here is comparable to a row there.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from launch_scripts.score_fold0_subset import build_loader, load_model


@torch.no_grad()
def cache_forward(model, loader, device):
    """One pass over the fold. Keeps only what decode_events and the metrics need."""
    cache = []
    for batch in loader:
        spect = batch["spect"].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(spect)
        num_frames = batch["truth_beat"].shape[-1]
        window_seconds = num_frames / model.fps
        pad = batch.get("padding_mask")
        for i in range(len(batch["spect"])):
            truth = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth) < 3:
                continue
            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            has_db = bool(batch["downbeat_mask"][i]) and len(truth_db) >= 3
            cache.append(dict(
                class_logits=pred["class_logits"][i].float().cpu(),
                t_hat=pred["t_hat"][i].float().cpu(),
                window_seconds=window_seconds,
                valid_seconds=(float(pad[i].sum()) / model.fps) if pad is not None else None,
                truth=truth, truth_db=truth_db if has_db else None,
            ))
    return cache


def evaluate(model, cache, tau_beat, tau_downbeat):
    from alignbeat.classes import DOWNBEAT
    from alignbeat.decode import decode_events
    F = CMLt = dbF = dbCMLt = 0.0
    n = n_db = 0
    for c in cache:
        classes, times, _ = decode_events(c["class_logits"], c["t_hat"],
                                          tau_beat, tau_downbeat,
                                          db_margin=model.db_margin)
        seconds = (times * c["window_seconds"]).numpy()
        classes = classes.numpy()
        if c["valid_seconds"] is not None:
            keep = seconds < c["valid_seconds"]
            seconds, classes = seconds[keep], classes[keep]
        met = model.metrics(c["truth"], np.sort(seconds), step="test")
        F += met["F-measure"]; CMLt += met["CMLt"]; n += 1
        if c["truth_db"] is not None:
            met_d = model.metrics(c["truth_db"], np.sort(seconds[classes == DOWNBEAT]),
                                  step="test")
            dbF += met_d["F-measure"]; dbCMLt += met_d["CMLt"]; n_db += 1
    return F / n, CMLt / n, dbF / max(n_db, 1), dbCMLt / max(n_db, 1), n, n_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--taus", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5",
                    help="grid, applied to both thresholds")
    ap.add_argument("--out", default="cache/tau_sweep.csv")
    args = ap.parse_args()

    grid = [float(x) for x in args.taus.split(",")]
    device = f"cuda:{args.gpu}"
    loader = build_loader(args.num_workers, args.fold)
    model = load_model(args.ckpt, device)
    print(f"  baseline tau from checkpoint: beat={model.tau_beat} "
          f"downbeat={model.tau_downbeat}", flush=True)

    cache = cache_forward(model, loader, device)
    print(f"  cached {len(cache)} pieces; sweeping {len(grid)}x{len(grid)}\n", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("ckpt,tau_beat,tau_downbeat,n,F,CMLt,n_db,dbF,dbCMLt\n")
        best_b = best_db = None
        for tb in grid:
            for td in grid:
                F, CMLt, dbF, dbCMLt, n, n_db = evaluate(model, cache, tb, td)
                fh.write(f"{os.path.basename(args.ckpt)},{tb},{td},{n},{F:.6f},"
                         f"{CMLt:.6f},{n_db},{dbF:.6f},{dbCMLt:.6f}\n")
                fh.flush()
                print(f"  tau_b={tb:<5} tau_db={td:<5}  beat F={F:.4f} CMLt={CMLt:.4f} | "
                      f"downbeat F={dbF:.4f} CMLt={dbCMLt:.4f}", flush=True)
                if best_b is None or F > best_b[0]:
                    best_b = (F, tb, td)
                if best_db is None or dbF > best_db[0]:
                    best_db = (dbF, tb, td)
    print(f"\n  best beat F     {best_b[0]:.4f} at tau_beat={best_b[1]} tau_db={best_b[2]}")
    print(f"  best downbeat F {best_db[0]:.4f} at tau_beat={best_db[1]} tau_db={best_db[2]}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
