#!/usr/bin/env python3
"""Reproduce Beat Transformer's published 8-fold splits for our datasets.

Why: the existing .folds files for ballroom/beatles/hainsworth turned out to be
byte-identical to the split published by Beat This! (Foscarin et al. 2024), while
carnatic/harmonix had no fold file at all and fell back to an arbitrary 80/10/10.
The two sources' fold assignments are entirely unrelated - on ballroom they agree
on 11.8% of songs, against 12.5% expected by chance for 8 folds. Which one is used
therefore changes the results, so it has to be unified on one.

Beat Transformer (Zhao et al. 2022) was chosen because its training data is nearly
the same as ours - of their seven (ballroom, hainsworth, rwc_popular, harmonix,
carnatic, smc, plus gtzan as test-only) only beatles is missing. Beat This! trains
on 18 datasets, so matching their folds would not make our numbers comparable to
their published ones anyway.

How: their splitting rule is fully reproducible (code/spectrogram_dataset.py
L328-355):
    FOLD_SIZE = len(data) // 8
    np.random.seed(0); np.random.shuffle(audio_root)
    fold i (i < 7) = audio_root[i*FOLD_SIZE : (i+1)*FOLD_SIZE]
    fold 7         = everything left over
The pre-shuffle order is the line order of data/audio_lists/{key}.txt, published
alongside the repository.

Confidence per source:
  ballroom / hainsworth / carnatic / smc
      The published audio_lists are used as-is; the stems match our filenames
      directly.
  harmonix
      They renamed the files to bare 000-911, so the published list alone does not
      say which song is which. Recovered by annotation signature (beat count, first
      and last time) - 910 of 912 are uniquely determined, and the remaining two
      (0250_sexyandiknowit / 0324_yeah3x) have identical signatures and were split
      arbitrarily. Worst case, those two songs swap folds.
  rwc_popular
      audio_lists/rwc.txt was never published (per the README, RWC audio cannot be
      redistributed for royalty reasons). Recovered by matching the npz annotations
      against our RM-P files with an assignment solver - 87 of 100 match times
      exactly (0.000s) and the remaining 13 fall out by elimination under the
      bijection constraint. Those 13 appear to be the "RWC pop annotations partially
      corrected by Böck" the Beat This! README mentions. This is inference, not
      verification, and should be stated as such in the paper.

Caveat: our ballroom copy is missing 13 of the 685 songs both sources list (e.g.
Albums-AnaBelen_Veneo-03, -11). The fold assignment is theirs, but the fold
membership is a subset - which applies equally whichever fold source is used.

Usage:
  python make_bt_folds.py            # preview
  python make_bt_folds.py --write    # write into dataset_folds/<ds>/label/
                                     #   (annotations are mirrored as symlinks)
"""
import argparse
import glob
import os
from collections import Counter

import numpy as np

BT_REPO = '/home/sogang/jaehoon/Beat-Transformer'
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_folds')
LABELED = '/disk1/taegum/mnt/labeled_data'
NUM_FOLDS = 8
SEED = 0

# dataset -> (Beat Transformer audio-list stem, our annotation dir, annotation glob)
DATASETS = {
    'ballroom':    ('ballroom',   f'{LABELED}/ballroom/label',    '*.beats'),
    'hainsworth':  ('hainsworth', f'{LABELED}/hains/label',       '*.txt'),
    'carnatic':    ('carnetic',   '/disk4/taegum/carnatic/label', '*.beats'),
    'harmonix':    ('harmonix',   '/disk4/taegum/harmonix_griffinlim/annotations_urinieto', '*.txt'),
    'rwc_popular': (None,         f'{LABELED}/rwc_popular/label', 'RM-P*.BEAT.TXT'),
    'smc':         ('smc',        None, '*.txt'),  # annotations resolved separately
}
SMC_AUDIO = '/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio'


def assign_folds(order):
    """Beat Transformer's exact recipe: seeded shuffle, contiguous chunks,
    remainder to the last fold."""
    shuffled = list(order)
    np.random.seed(SEED)
    np.random.shuffle(shuffled)
    size = len(shuffled) // NUM_FOLDS
    out = {}
    for i in range(NUM_FOLDS - 1):
        for stem in shuffled[i * size:(i + 1) * size]:
            out[stem] = i
    for stem in shuffled[(NUM_FOLDS - 1) * size:]:
        out[stem] = NUM_FOLDS - 1
    return out


def published_order(list_stem):
    path = os.path.join(BT_REPO, 'data', 'audio_lists', f'{list_stem}.txt')
    return [os.path.splitext(os.path.basename(x.strip()))[0]
            for x in open(path) if x.strip()]


def beat_times(path, kind):
    times = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if kind == 'rwc':
            parts = line.split('\t')
            if len(parts) >= 3:
                times.append(int(parts[0]) / 100.0)
        else:
            first = line.replace('\t', ' ').split()[0]
            try:
                times.append(float(first))
            except ValueError:
                pass
    return np.array(times)


def recover_harmonix_order(annot_dir):
    """BT renamed harmonix to bare indices, so match on annotation signature."""
    annotations = np.load(os.path.join(BT_REPO, 'data', 'full_beat_annotation.npz'),
                          allow_pickle=True)['harmonix']
    ours = {os.path.basename(f)[:-4]: beat_times(f, 'harmonix')
            for f in sorted(glob.glob(os.path.join(annot_dir, '*.txt')))}

    def signature(t):
        return (len(t), round(float(t[0]), 2), round(float(t[-1]), 2)) if len(t) else None

    buckets = {}
    for stem, t in ours.items():
        buckets.setdefault(signature(t), []).append(stem)
    order, used, ambiguous = [], set(), 0
    for entry in annotations:
        candidates = [s for s in buckets.get(signature(np.asarray(entry)[:, 0]), [])
                      if s not in used]
        if len(candidates) > 1:
            ambiguous += 1
        if candidates:
            order.append(candidates[0])
            used.add(candidates[0])
    return order, ambiguous


def recover_rwc_order(annot_dir):
    """No published list for RWC; recover by matching annotation times."""
    from scipy.optimize import linear_sum_assignment
    annotations = [np.asarray(x) for x in
                   np.load(os.path.join(BT_REPO, 'data', 'full_beat_annotation.npz'),
                           allow_pickle=True)['rwc']]
    ours = {os.path.basename(f).split('.')[0]: beat_times(f, 'rwc')
            for f in sorted(glob.glob(os.path.join(annot_dir, 'RM-P*.BEAT.TXT')))}
    names = sorted(ours)

    def cost(a, b):
        n = min(len(a), len(b))
        if n < 10:
            return 1e9
        return float(np.abs(a[:n] - b[:n]).mean()) + 0.01 * abs(len(a) - len(b))

    matrix = np.array([[cost(entry[:, 0], ours[n]) for n in names] for entry in annotations])
    rows, cols = linear_sum_assignment(matrix)
    exact = sum(1 for r, c in zip(rows, cols) if matrix[r, c] < 1e-6)
    return [names[c] for _, c in sorted(zip(rows, cols))], exact


def mirror_and_write(dataset, assignment, annot_dir, write):
    """Symlink the annotations next to a single fold file.

    The dataloader globs "*8-fold*.folds" in annot_dir and takes the first hit, so
    the mirror must contain exactly one - the original directories also hold
    Beat This! and 80/10/10 files, and which one won would otherwise be arbitrary.
    """
    on_disk = {os.path.splitext(f)[0] if not f.endswith('.BEAT.TXT') else f.split('.')[0]
               for f in os.listdir(annot_dir) if not f.endswith('.folds')}
    kept = [(s, k) for s, k in sorted(assignment.items()) if s in on_disk]
    sizes = dict(sorted(Counter(k for _, k in kept).items()))
    print(f"{dataset:12s} listed={len(assignment):4d} on_disk={len(on_disk):4d} "
          f"written={len(kept):4d} folds={sizes}")
    if not write:
        return
    dst = os.path.join(OUT_ROOT, dataset, 'label')
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(annot_dir):
        if f.endswith('.folds'):
            continue
        link = os.path.join(dst, f)
        if not os.path.islink(link):
            try:
                os.symlink(os.path.join(annot_dir, f), link)
            except FileExistsError:
                pass
    with open(os.path.join(dst, f'{dataset}_8-fold_cv_beat_transformer.folds'), 'w') as fp:
        for stem, fold in kept:
            fp.write(f"{dataset}_{stem}\t{fold}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--smc_annot_dir', default=None,
                        help="SMC annotation directory (the shipped one is often unreadable; "
                             "extract SMC_MIREX_Annotations_05_08_2014 somewhere writable)")
    args = parser.parse_args()

    for dataset, (list_stem, annot_dir, _glob) in DATASETS.items():
        if dataset == 'harmonix':
            order, ambiguous = recover_harmonix_order(annot_dir)
            print(f"  harmonix: recovered {len(order)} by annotation signature "
                  f"({ambiguous} ambiguous, broken arbitrarily)")
        elif dataset == 'rwc_popular':
            order, exact = recover_rwc_order(annot_dir)
            print(f"  rwc_popular: recovered {len(order)} by assignment "
                  f"({exact} exact time matches, rest by elimination)")
        elif dataset == 'smc':
            annot_dir = args.smc_annot_dir
            if annot_dir is None or not os.path.isdir(annot_dir):
                print("smc          skipped (pass --smc_annot_dir)")
                continue
            # BT lists audio stems (SMC_088); annotations carry extra suffixes
            order = published_order(list_stem)
            audio = {os.path.splitext(f)[0] for f in os.listdir(SMC_AUDIO) if f.endswith('.wav')}
            order = [s for s in order if s in audio]
        else:
            order = published_order(list_stem)

        assignment = assign_folds(order)
        if dataset == 'smc':
            # fold file keys on the audio stem, which is what the dataloader resolves
            dst = os.path.join(OUT_ROOT, 'smc', 'label')
            print(f"{'smc':12s} listed={len(assignment):4d} "
                  f"folds={dict(sorted(Counter(assignment.values()).items()))}")
            if args.write:
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(annot_dir):
                    link = os.path.join(dst, f)
                    if not os.path.islink(link):
                        try:
                            os.symlink(os.path.join(annot_dir, f), link)
                        except FileExistsError:
                            pass
                with open(os.path.join(dst, 'smc_8-fold_cv_beat_transformer.folds'), 'w') as fp:
                    for stem, fold in sorted(assignment.items()):
                        fp.write(f"smc_{stem}\t{fold}\n")
            continue
        mirror_and_write(dataset, assignment, annot_dir, args.write)


if __name__ == '__main__':
    main()
