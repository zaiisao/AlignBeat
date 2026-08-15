#!/usr/bin/env python3
"""Sweep the decode threshold tau for the subset head, using mir_eval.

[왜 필요한가] tau 기본값 0.2는 FCOS 경로의 관례(beat 0.20 / downbeat 0.05)에서
그대로 가져온 값인데, subset head에서는 **아무 일도 하지 않는다**. 이 헤드의 decode는
3-class softmax의 argmax를 쓰므로 이긴 클래스의 확률은 항상 1/3 이상이다. 따라서
tau <= 0.333은 단 하나의 예측도 거르지 못한다 - 실측으로 0.05~0.30 구간의 결과가
완전히 동일했다. 의미 있는 구간은 (1/3, 1)뿐이다.

decode는 cached logits에서 다시 돌리므로 forward pass는 곡당 한 번뿐이고, tau를
아무리 많이 훑어도 추가 비용이 거의 없다.

평가는 학습/평가 경로와 동일하게 mir_eval(5초 trim, ±70ms)을 쓰고, beat 목록에는
downbeat으로 분류된 후보도 합친다(GT의 beat 사슬이 downbeat을 포함하므로).
"""
import argparse
import os
import sys

import mir_eval
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from beatfcos import model_module                             # noqa: E402
from beatfcos.dataloader import BeatDataset, collater, BEAT_ONLY_DATASETS  # noqa: E402
from beatfcos.stitching import fragment_offsets               # noqa: E402
from beatfcos.subset_head import BACKGROUND, DOWNBEAT         # noqa: E402

SR, HOP = 22050, 512
B = '/disk1/taegum/mnt/labeled_data'
SETS = [('ballroom', f'{B}/ballroom/data'), ('hainsworth', f'{B}/hains/data'),
        ('rwc_popular', f'{B}/rwc_popular/data'), ('carnatic', '/disk4/taegum/carnatic/data'),
        ('harmonix', '/disk4/taegum/harmonix_griffinlim/audio'),
        ('smc', '/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio')]


def gt_seconds(annot, to_sec):
    beats, downs, lb, ld = [], [], None, None
    for row in annot:
        lab = int(row[2])
        if lab < 0:
            continue
        l, r = int(row[0]), int(row[1])
        if lab == DOWNBEAT:
            downs.append(l * to_sec); ld = r if ld is None else max(ld, r)
        else:
            beats.append(l * to_sec); lb = r if lb is None else max(lb, r)
    if lb is not None:
        beats.append(lb * to_sec)
    if ld is not None:
        downs.append(ld * to_sec)
    return np.sort(np.array(beats)), np.sort(np.array(downs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num_candidates', type=int, default=160)
    ap.add_argument('--max_songs', type=int, default=100)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    WIN = args.num_candidates * 8
    to_sec = HOP / SR
    F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_folds')

    model = model_module.BeatFCOS(
        num_classes=2, clusters=torch.tensor([0.4, 0.7, 1.9]), head_type='subset',
        audio_downsampling_factor=HOP, audio_sample_rate=SR, dmodel=128, nhead=8,
        d_hid=512, nlayers=9, attn_len=5, dropout=0.1, num_candidates=args.num_candidates)
    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in sd.items()}, strict=False)
    model = model.cuda().eval()

    # tau below 1/3 cannot reject anything (3-way argmax), so start just above it
    TAUS = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    # per (tau_beat, tau_downbeat) we accumulate per-dataset F, so the two can differ
    scores = {(tb, td): {n: [[], []] for n, _ in SETS} for tb in TAUS for td in TAUS}

    with torch.no_grad():
        for name, audio_dir in SETS:
            ds = BeatDataset(audio_dir, f'{F}/{name}/label', dataset=name,
                             audio_sample_rate=SR, audio_downsampling_factor=HOP,
                             subset='val', length=2097152, spectral=True,
                             preload=False, augment=False, validation_fold=0)
            n = min(args.max_songs, len(ds.audio_files))
            for i in range(n):
                item = ds[i]
                mel = item[0].cuda()
                gt_b, gt_d = gt_seconds(collater([item])[1][0], to_sec)
                cache = []
                for off, ks, ke in fragment_offsets(mel.shape[0], WIN, 8):
                    f = mel[off:off + WIN]
                    if f.shape[0] < WIN:
                        f = torch.nn.functional.pad(f, (0, 0, 0, WIN - f.shape[0]))
                    cl, th = model(f.unsqueeze(0))
                    p = torch.softmax(cl[0], -1).cpu()
                    sc, pred = p.max(-1)
                    absf = off + th[0].cpu() * WIN
                    inside = (absf >= ks) & ((absf <= ke) if ke == mel.shape[0] else (absf < ke))
                    cache.append((sc[inside], pred[inside], (absf[inside] * to_sec)))
                for tb in TAUS:
                    for td in TAUS:
                        eb, ed = [], []
                        for sc, pred, t in cache:
                            thr = torch.where(pred == DOWNBEAT, torch.full_like(sc, td),
                                              torch.full_like(sc, tb))
                            k = (pred != BACKGROUND) & (sc >= thr)
                            tt = t[k].numpy(); cc = pred[k].numpy()
                            eb += list(tt); ed += list(tt[cc == DOWNBEAT])
                        fb = mir_eval.beat.evaluate(mir_eval.beat.trim_beats(gt_b),
                                                    mir_eval.beat.trim_beats(np.sort(np.array(eb))))['F-measure']
                        scores[(tb, td)][name][0].append(fb)
                        if name not in BEAT_ONLY_DATASETS:
                            fd = mir_eval.beat.evaluate(mir_eval.beat.trim_beats(gt_d),
                                                        mir_eval.beat.trim_beats(np.sort(np.array(ed))))['F-measure']
                            scores[(tb, td)][name][1].append(fd)
            print(f"  {name}: {n} songs", flush=True)

    def macro(d):
        b = np.mean([np.mean(v[0]) for v in d.values() if v[0]])
        dd = np.mean([np.mean(v[1]) for v in d.values() if v[1]])
        return b, dd, (b + dd) / 2

    print("\n=== tau_beat x tau_downbeat -> Joint (macro, mir_eval) ===")
    print("      " + "".join(f"{td:>7.2f}" for td in TAUS) + "   <- tau_downbeat")
    best = None
    for tb in TAUS:
        row = f"{tb:>5.2f} "
        for td in TAUS:
            _, _, j = macro(scores[(tb, td)])
            row += f"{j:>7.3f}"
            if best is None or j > best[0]:
                best = (j, tb, td)
        print(row)
    j, tb, td = best
    b, d, _ = macro(scores[(tb, td)])
    print(f"\nBEST: tau_beat={tb:.2f} tau_downbeat={td:.2f} -> Beat {b:.3f} Downbeat {d:.3f} Joint {j:.3f}")
    b0, d0, j0 = macro(scores[(0.34, 0.34)])
    print(f"CURRENT (tau=0.2 == 0.34, both no-ops below 1/3): Beat {b0:.3f} Downbeat {d0:.3f} Joint {j0:.3f}")
    print(f"gain: {j - j0:+.3f} Joint")


if __name__ == '__main__':
    main()
