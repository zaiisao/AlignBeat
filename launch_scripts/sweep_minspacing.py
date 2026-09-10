"""Decode-time minimum-spacing (de-duplication) sweep on a trained checkpoint.

Section 9.2 argues no NMS or de-duplication is needed, on the grounds that exactly one
decision is made per candidate and equation (1) makes t_hat strictly increasing, so "no
two reported detections can ever coincide or cross in time". That establishes only that
detections do not COINCIDE. Two detections a fraction of a beat apart do not coincide
and are still a duplicate, and measurement on T_omega2 puts ~30% of insertions in that
category (offset < 0.15 IBI, with the true event's own candidate also fired).

This sweeps a minimum gap, expressed as a fraction of the model's OWN estimated inter-
beat interval (the median gap between its emitted events, so no ground truth leaks in),
and keeps the higher-scoring detection of any pair closer than that. One forward pass;
the decode is then re-run per grid point on the cache.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from launch_scripts.score_fold0_subset import build_loader, load_model


def suppress(sec, cls, score, min_gap):
    """Greedy highest-score-first suppression of detections closer than min_gap."""
    if len(sec) == 0 or min_gap <= 0:
        return sec, cls
    order = np.argsort(-score)
    keep = np.ones(len(sec), dtype=bool)
    taken = []
    for i in order:
        if any(abs(sec[i] - sec[j]) < min_gap for j in taken):
            keep[i] = False
        else:
            taken.append(i)
    return sec[keep], cls[keep]


@torch.no_grad()
def cache_forward(model, loader, device):
    from alignbeat.classes import DOWNBEAT
    from alignbeat.decode import decode_events
    out = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        n_frames = batch["truth_beat"].shape[-1]
        window = n_frames / model.fps
        pad = batch.get("padding_mask")
        for i in range(len(batch["spect"])):
            truth = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth) < 3:
                continue
            cls, times, score = decode_events(pred["class_logits"][i], pred["t_hat"][i], model.tau)
            sec = (times * window).cpu().numpy()
            cls = cls.cpu().numpy(); score = score.cpu().numpy()
            if pad is not None:
                valid = float(pad[i].sum()) / model.fps
                m = sec < valid
                sec, cls, score = sec[m], cls[m], score[m]
            tdb = np.frombuffer(batch["truth_orig_downbeat"][i])
            has_db = bool(batch["downbeat_mask"][i]) and len(tdb) >= 3
            out.append(dict(sec=sec, cls=cls, score=score, truth=truth,
                            truth_db=tdb if has_db else None))
    return out


def evaluate(model, cache, frac):
    from alignbeat.classes import DOWNBEAT
    F = C = dF = dC = 0.0; n = nd = 0
    for c in cache:
        sec, cls = c["sec"], c["cls"]
        if frac > 0 and len(sec) > 2:
            ibi = float(np.median(np.diff(np.sort(sec))))
            sec, cls = suppress(sec, cls, c["score"], frac * ibi)
        m = model.metrics(c["truth"], np.sort(sec), step="test")
        F += m["F-measure"]; C += m["CMLt"]; n += 1
        if c["truth_db"] is not None:
            md = model.metrics(c["truth_db"], np.sort(sec[cls == DOWNBEAT]), step="test")
            dF += md["F-measure"]; dC += md["CMLt"]; nd += 1
    return F/n, C/n, dF/max(nd,1), dC/max(nd,1), n, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--fracs", default="0,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--out", default="cache/minspacing.csv")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    loader = build_loader(a.num_workers, a.fold)
    model = load_model(a.ckpt, dev)
    cache = cache_forward(model, loader, dev)
    print(f"  cached {len(cache)} pieces\n", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("ckpt,min_gap_frac_of_ibi,n,F,CMLt,n_db,dbF,dbCMLt\n")
        for f in [float(x) for x in a.fracs.split(",")]:
            F, C, dF, dC, n, nd = evaluate(model, cache, f)
            fh.write(f"{os.path.basename(a.ckpt)},{f},{n},{F:.6f},{C:.6f},{nd},{dF:.6f},{dC:.6f}\n")
            fh.flush()
            print(f"  min_gap={f:<5} beat F={F:.4f} CMLt={C:.4f} | downbeat F={dF:.4f} CMLt={dC:.4f}", flush=True)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
