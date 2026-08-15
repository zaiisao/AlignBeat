#!/usr/bin/env python3
"""Reproduce Beat Transformer's published 8-fold splits for our datasets.

[왜] ballroom/beatles/hainsworth의 기존 .folds 파일은 사실 Beat This!(Foscarin et
al. 2024)가 공개한 split과 byte-identical이었고, carnatic/harmonix에는 아예 fold
파일이 없어서 임의의 80/10/10 fallback을 쓰고 있었다. 두 출처의 fold 배정은 서로
완전히 무관하다(ballroom 기준 일치율 11.8%, 8-fold 무작위 기대치 12.5%). 즉 어느
쪽을 쓰는지가 결과에 영향을 주므로 하나로 통일해야 한다.

Beat Transformer(Zhao et al. 2022)를 택한 이유는 학습 데이터 구성이 우리와 거의
같기 때문이다 - 그들의 7개(ballroom, hainsworth, rwc_popular, harmonix, carnatic,
smc + gtzan은 test-only)에서 beatles만 빠진다. Beat This!는 18개 데이터셋으로
학습해서, fold를 맞춰봐야 published 숫자와 비교가 성립하지 않는다.

[방법] 그들의 분할 규칙은 완전히 재현 가능하다 (code/spectrogram_dataset.py
L328-355):
    FOLD_SIZE = len(data) // 8
    np.random.seed(0); np.random.shuffle(audio_root)
    fold i (i < 7) = audio_root[i*FOLD_SIZE : (i+1)*FOLD_SIZE]
    fold 7         = 나머지 전부
셔플 이전 순서는 저장소에 함께 공개된 data/audio_lists/{key}.txt의 줄 순서다.

[출처별 신뢰도]
  ballroom / hainsworth / carnatic / smc
      공개된 audio_lists를 그대로 사용. stem이 우리 파일명과 직접 일치.
  harmonix
      그들은 파일을 000~911로 재명명해서 공개 목록만으로는 어느 곡인지 알 수 없다.
      annotation 시그니처(비트 수, 첫/마지막 시각)로 역매핑함 - 912개 중 910개가
      유일하게 결정되고 나머지 2개(0250_sexyandiknowit / 0324_yeah3x)는 시그니처가
      완전히 같아 임의로 갈랐다. 최악의 경우 두 곡의 fold가 서로 바뀌는 정도.
  rwc_popular
      audio_lists/rwc.txt는 공개되지 않았다(README: RWC는 royalty 문제로 오디오
      배포 불가). npz의 annotation과 우리 RM-P 파일을 Hungarian 매칭으로 복원 -
      100곡 중 87곡은 시각이 정확히 일치(0.000s), 나머지 13곡은 bijection 제약으로
      소거법으로 결정됨. 이 13곡은 Beat This! README가 언급한 "Böck이 부분 수정한
      RWC pop annotation"과 겹치는 것으로 보인다. 검증이 아니라 추론이므로 논문에
      그대로 밝힐 것.

[주의] 우리 ballroom 사본은 두 출처가 공통으로 나열하는 685곡 중 13곡이 없다
(예: Albums-AnaBelen_Veneo-03, -11). fold 배정 자체는 그들 것이지만 fold 구성은
부분집합이며, 이는 어느 fold 출처를 쓰든 동일하게 적용된다.

사용법:
  python make_bt_folds.py            # 미리보기
  python make_bt_folds.py --write    # dataset_folds/<ds>/label/ 에 기록
                                     #   (annotation은 symlink로 미러링)
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
