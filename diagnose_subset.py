#!/usr/bin/env python3
"""Locate what caps the subset head's F-measure.

Answers three questions that F alone cannot:

  1. precision vs recall -- "we miss events" and "we emit spurious ones" both lower
     F but want opposite threshold changes.
  2. timing vs detection -- for every ground-truth event, is there a prediction
     NEAR it that simply falls outside the +-70 ms tolerance (a regression
     problem), or is there nothing nearby at all (a detection problem)?
  3. tau -- never swept. The FCOS path found beat 0.20 / downbeat 0.05 optimal, so
     the shared 0.2 default may be discarding correct downbeats outright.

Decoding is re-run from cached logits, so the tau sweep costs one forward pass.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beatfcos import model_module                                    # noqa: E402
from beatfcos.dataloader import BeatDataset, collater                # noqa: E402
from beatfcos.stitching import fragment_offsets                      # noqa: E402
from beatfcos.subset_head import BEAT, DOWNBEAT, BACKGROUND          # noqa: E402

TOL = 0.070


def gt_from_annot(annot, to_seconds):
    """Same convention the evaluator uses: beat chain includes downbeats."""
    beats, downbeats = [], []
    last_b = last_d = None
    for row in annot:
        label = int(row[2])
        if label < 0:
            continue
        left, right = int(row[0]), int(row[1])
        if label == DOWNBEAT:
            downbeats.append(left * to_seconds)
            last_d = right if last_d is None else max(last_d, right)
        else:                       # beat (1) or beat-only (2)
            beats.append(left * to_seconds)
            last_b = right if last_b is None else max(last_b, right)
    if last_b is not None:
        beats.append(last_b * to_seconds)
    if last_d is not None:
        downbeats.append(last_d * to_seconds)
    return np.sort(np.array(beats)), np.sort(np.array(downbeats))


def match(ref, est, tol=TOL):
    """Greedy nearest matching; returns (n_hit, signed errors of hits, nearest-dist per ref)."""
    if len(ref) == 0 or len(est) == 0:
        return 0, np.array([]), np.full(len(ref), np.inf)
    used = np.zeros(len(est), bool)
    hits, errs, nearest = 0, [], []
    for r in ref:
        d = np.abs(est - r)
        d[used] = np.inf
        j = int(np.argmin(d))
        nearest.append(float(np.min(np.abs(est - r))))
        if d[j] <= tol:
            hits += 1
            errs.append(float(est[j] - r))
            used[j] = True
    return hits, np.array(errs), np.array(nearest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num_candidates', type=int, default=160)
    ap.add_argument('--max_songs', type=int, default=40)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    SR, HOP = 22050, 512
    WIN = args.num_candidates * 8
    to_seconds = HOP / SR
    F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_folds')
    B = '/disk1/taegum/mnt/labeled_data'
    SETS = [('ballroom', f'{B}/ballroom/data'), ('hainsworth', f'{B}/hains/data'),
            ('rwc_popular', f'{B}/rwc_popular/data'), ('carnatic', '/disk4/taegum/carnatic/data'),
            ('harmonix', '/disk4/taegum/harmonix_griffinlim/audio'),
            ('smc', '/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio')]

    model = model_module.BeatFCOS(
        num_classes=2, clusters=torch.tensor([0.4, 0.7, 1.9]), head_type='subset',
        audio_downsampling_factor=HOP, audio_sample_rate=SR, dmodel=128, nhead=8,
        d_hid=512, nlayers=9, attn_len=5, dropout=0.1, num_candidates=args.num_candidates)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    model = model.cuda().eval()
    print(f"loaded {args.checkpoint} (missing={len(missing)})")

    TAUS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    agg = {t: {'b': [0, 0, 0], 'd': [0, 0, 0]} for t in TAUS}   # hits, n_ref, n_est
    all_err_b, all_err_d, near_b, near_d = [], [], [], []
    per_ds = {}

    with torch.no_grad():
        for name, audio_dir in SETS:
            ds = BeatDataset(audio_dir, f'{F}/{name}/label', dataset=name,
                             audio_sample_rate=SR, audio_downsampling_factor=HOP,
                             subset='val', length=2097152, spectral=True,
                             preload=False, augment=False, validation_fold=0)
            n = min(args.max_songs, len(ds.audio_files))
            ds_hits = [0, 0, 0]
            for i in range(n):
                item = ds[i]
                mel = item[0].cuda()
                annot = collater([item])[1][0]
                gt_b, gt_d = gt_from_annot(annot, to_seconds)

                # cache logits/times per fragment once, decode at every tau
                frags = fragment_offsets(mel.shape[0], WIN, 8)
                cached = []
                for off, ks, ke in frags:
                    f = mel[off:off + WIN]
                    if f.shape[0] < WIN:
                        f = torch.nn.functional.pad(f, (0, 0, 0, WIN - f.shape[0]))
                    cl, th = model(f.unsqueeze(0))
                    cached.append((cl[0].cpu(), th[0].cpu(), off, ks, ke))

                for tau in TAUS:
                    est_b, est_d = [], []
                    for cl, th, off, ks, ke in cached:
                        p = torch.softmax(cl, -1)
                        sc, pred = p.max(-1)
                        keep = (pred != BACKGROUND) & (sc >= tau)
                        absf = off + th * WIN
                        inside = (absf >= ks) & ((absf <= ke) if ke == mel.shape[0] else (absf < ke))
                        sel = keep & inside
                        t_sec = (absf[sel] * to_seconds).numpy()
                        c_sel = pred[sel].numpy()
                        est_b += list(t_sec)                      # beat list = B + DB
                        est_d += list(t_sec[c_sel == DOWNBEAT])
                    eb, ed = np.sort(np.array(est_b)), np.sort(np.array(est_d))
                    hb, errb, nb = match(gt_b, eb)
                    hd, errd, nd = match(gt_d, ed)
                    agg[tau]['b'][0] += hb; agg[tau]['b'][1] += len(gt_b); agg[tau]['b'][2] += len(eb)
                    agg[tau]['d'][0] += hd; agg[tau]['d'][1] += len(gt_d); agg[tau]['d'][2] += len(ed)
                    if abs(tau - 0.20) < 1e-9:
                        all_err_b += list(errb); all_err_d += list(errd)
                        near_b += list(nb); near_d += list(nd)
                        ds_hits[0] += hb; ds_hits[1] += len(gt_b); ds_hits[2] += len(eb)
            per_ds[name] = ds_hits
            print(f"  {name}: {n} songs done")

    def prf(h, nref, nest):
        p = h / nest if nest else 0.0
        r = h / nref if nref else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    print("\n=== 1+3. tau sweep: precision / recall / F ===")
    print(f"{'tau':>5} | {'BEAT  P':>8} {'R':>6} {'F':>6} | {'DOWNBEAT  P':>12} {'R':>6} {'F':>6}")
    for t in TAUS:
        pb, rb, fb = prf(*agg[t]['b']); pd, rd, fd = prf(*agg[t]['d'])
        print(f"{t:>5.2f} | {pb:8.3f} {rb:6.3f} {fb:6.3f} | {pd:12.3f} {rd:6.3f} {fd:6.3f}")

    print("\n=== 2. timing vs detection (at tau=0.20) ===")
    for label, errs, near in (('BEAT', np.array(all_err_b), np.array(near_b)),
                              ('DOWNBEAT', np.array(all_err_d), np.array(near_d))):
        if len(near) == 0:
            continue
        within = (near <= TOL).mean()
        near_miss = ((near > TOL) & (near <= 0.200)).mean()
        far = (near > 0.200).mean()
        print(f"{label}: of {len(near)} GT events -- "
              f"{within*100:5.1f}% have a prediction within +-70ms, "
              f"{near_miss*100:5.1f}% only within 70-200ms (MISTIMED), "
              f"{far*100:5.1f}% nothing within 200ms (MISSED)")
        if len(errs):
            print(f"   matched residuals: mean|e| {np.abs(errs).mean()*1000:5.1f} ms, "
                  f"median {np.median(np.abs(errs))*1000:5.1f} ms, "
                  f"p90 {np.percentile(np.abs(errs),90)*1000:5.1f} ms, bias {errs.mean()*1000:+.1f} ms")

    print("\n=== per-dataset at tau=0.20 (beat) ===")
    for k, v in per_ds.items():
        p, r, f = prf(*v)
        print(f"  {k:12s} P {p:.3f}  R {r:.3f}  F {f:.3f}")


if __name__ == '__main__':
    main()
