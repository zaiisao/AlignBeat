"""Sweep decode's threshold tau on a trained checkpoint.

tau has been 0.2 since decode.py was written and was never tuned. The oracle ladder
says why that matters: handing over the event SET takes beat F from 0.898 to 0.990, and
simac from 0.760 to 0.978 -- so which candidates get emitted is the single largest loss
in the pipeline, and tau is the one knob controlling it.

A first sweep showed tau is INERT: identical scores from 0.05 to 0.30, because the emit
decision is argmax over three classes and tau only fires when the winning probability
falls below it, which almost never happens. So the knob that actually controls how many
candidates are emitted is a bias on the CLASS_BACKGROUND logit -- subtract b and every
candidate whose event evidence is within b nats of background is emitted too. b = 0 is
current behaviour; this sweeps it to trace the detection precision/recall trade the
ladder says is worth 0.09 beat F.

Both knobs live entirely inside decode, downstream of the encoder and the head, so
nothing about training changes; this only asks what the trained model would score if
asked a different question at decode time. Scoring is byte-identical to
score_fold0_subset.py -- same loader, same autocast, same skip rules, same per-piece
averaging -- so a row here is comparable to a row there.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alignbeat.inference.decode
import alignbeat.inference.stitching
import beat_this.model.pl_module
from alignbeat.constants import CLASS_BACKGROUND
from launch_scripts.score_fold0_subset import build_loader, load_model, score

_decode = alignbeat.inference.decode.decode


def set_background_bias(bias):
    """Shift the background channel wherever decode is looked up. Patching the
    module attributes rather than the function keeps the shipped decode untouched."""
    def patched(class_logits, t_hat, tau=0.2):
        if bias:
            class_logits = class_logits.clone()
            class_logits[..., CLASS_BACKGROUND] = class_logits[..., CLASS_BACKGROUND] - bias
        return _decode(class_logits, t_hat, tau)
    alignbeat.inference.decode.decode = patched

BEAT_ONLY = ("simac", "smc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--taus", default="0.2")
    ap.add_argument("--bg-biases", default="-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,3.0",
                    help="nats subtracted from the background logit before the argmax")
    args = ap.parse_args()

    taus = [float(v) for v in args.taus.split(",")]
    device = f"cuda:{args.gpu}"
    loader = build_loader(args.num_workers, args.fold)
    model = load_model(sorted(glob.glob(args.checkpoint))[0], device)

    def mean(rows, key, keep=lambda r: True):
        v = [r[key] for r in rows if key in r and keep(r)]
        return np.mean(v) if v else float("nan")

    print(f"\n  {'tau':>6}{'bg bias':>9}{'ALL':>9}{'CMLt':>9}{'labelled':>10}"
          f"{'simac':>9}{'smc':>9}{'dbF':>9}")
    best = None
    for tau in taus:
      for bias in [float(v) for v in args.bg_biases.split(",")]:
        model.tau = tau
        set_background_bias(bias)
        rows = score(model, loader, device)
        lab = lambda r: r["corpus"] not in BEAT_ONLY
        allF = mean(rows, "F")
        print(f"  {tau:>6.2f}{bias:>9.2f}{allF:>9.4f}{mean(rows, 'CMLt'):>9.4f}"
              f"{mean(rows, 'F', lab):>10.4f}"
              f"{mean(rows, 'F', lambda r: r['corpus'] == 'simac'):>9.4f}"
              f"{mean(rows, 'F', lambda r: r['corpus'] == 'smc'):>9.4f}"
              f"{mean(rows, 'dbF'):>9.4f}", flush=True)
        if best is None or allF > best[1]:
            best = ((tau, bias), allF)
    print(f"\n  best ALL F at tau={best[0][0]}, bg bias={best[0][1]} ({best[1]:.4f})")


if __name__ == "__main__":
    main()
