"""Dataset over Beat This's released spectrograms (Zenodo 13922116).

Why this exists: their corpus is 16 datasets (5554 tracks) against our 5 (2112), and
crucially it covers tempi we have none of - 375 tracks below 64 BPM and 161 below 55,
where our pool has 15 and ZERO. Since SMC is 37% below 64 BPM, that gap is the leading
explanation for our SMC score. Using their spectrograms rather than recomputing our own
keeps preprocessing out of the comparison entirely.

Their format:
  spectrograms  {root}/{dataset}.npz, keys "{track}/track" plus "{track}/track_ps{N}"
                (pitch, -5..+6 semitones) and "{track}/track_ts{N}" (tempo, -20..+20 %).
                Each (T, 128) float16 = log1p(1000 * mel), 22050 Hz / n_fft 1024 /
                hop 441 -> 50 fps, 128 mels 30-11000 Hz, slaney, magnitude (power=1).
  annotations   {ann}/{dataset}/annotations/beats/{track}.beats - one column of times
                (beat-only) or "time<TAB>beat_number" where 1 marks a downbeat.
  folds         {ann}/{dataset}/8-folds.split - "{track}<TAB>{fold}".
  info.json     {"has_downbeats": bool}

Note the frame rate: 50 fps here vs 43.07 in our own pipeline. Pass
--audio_downsampling_factor 441 so target_sample_rate = 22050/441 = 50 and every
frame<->second conversion downstream stays consistent; mixing the two rates in one run
would teach the model a blurred notion of tempo.

A time-stretched variant of p percent has its annotations scaled by 1/(1 + p/100),
exactly as beat_this/dataset/augment.py:stretch_annotations does.
"""
import json
import os
import random

import numpy as np
import torch

from beatfcos.dataloader import BEAT_ONLY_DATASETS, CLASS_BEAT_ONLY

# Mask augmentation, ported from beat_this/dataset/augment.py (augment_mask_ /
# apply_mask_excerpt) with beat_this's own default params (launch_scripts/train.py:50-56):
# kind="permute", 1-6 masks per crop, each 0.1-2s long, split into 5-9 pieces and
# shuffled in place. Operates on the already-cropped torch tensor (our pipeline
# converts to a tensor earlier than beat_this's numpy-based loader does).
_MASK_DEFAULTS = dict(kind="permute", min_count=1, max_count=6,
                       min_len=0.1, max_len=2.0, min_parts=5, max_parts=9)


def _apply_mask_excerpt(excerpt, kind, min_parts, max_parts):
    if kind == "permute":
        num_parts = random.randint(min_parts, max_parts)
        num_parts = min(num_parts, len(excerpt) + 1)
        if num_parts <= 1:
            return
        positions = sorted(random.sample(range(len(excerpt)), num_parts - 1))
        parts, prev = [], 0
        for pos in positions:
            parts.append(excerpt[prev:pos])
            prev = pos
        parts.append(excerpt[prev:])
        order = list(range(len(parts)))
        random.shuffle(order)
        excerpt[:] = torch.cat([parts[i] for i in order])
    elif kind == "zero":
        excerpt[:] = 0
    else:
        raise ValueError(f"Unsupported mask operation: {kind}")


def _augment_mask(spect, fps, kind="permute", min_count=1, max_count=6,
                   min_len=0.1, max_len=2.0, min_parts=5, max_parts=9):
    count = random.randint(min_count, max_count)
    min_len_f, max_len_f = int(min_len * fps), int(max_len * fps)
    for _ in range(count):
        length = random.randint(min_len_f, max_len_f)
        if length <= 0 or length >= len(spect):
            continue
        start = random.randint(0, len(spect) - length)
        _apply_mask_excerpt(spect[start:start + length], kind, min_parts, max_parts)
    return spect


class BeatThisSpectDataset(torch.utils.data.Dataset):
    def __init__(self, spect_root, annot_root, datasets, subset="train",
                 validation_fold=0, num_folds=8, target_length=1280,
                 frames_per_second=50.0, tempo_aug=(), pitch_aug=(),
                 mask_aug=False, beat_only_datasets=None):
        self.spect_root, self.annot_root = spect_root, annot_root
        self.subset, self.target_length = subset, target_length
        self.fps = frames_per_second
        self.tempo_aug = tuple(tempo_aug)
        self.pitch_aug = tuple(pitch_aug)
        self.mask_aug = bool(mask_aug)
        # Which datasets lack downbeat labels is read from their own info.json rather
        # than hardcoded: our BEAT_ONLY_DATASETS lists only smc, but this corpus also
        # ships simac with has_downbeats=false (595 tracks). Treating those beats as
        # class 1 would assert "definitely not a downbeat" for every one of them - the
        # exact wrong-supervision failure the smc handling exists to avoid.
        if beat_only_datasets is None:
            beat_only_datasets = set(BEAT_ONLY_DATASETS)
            for name in datasets:
                info = os.path.join(annot_root, name, "info.json")
                if os.path.exists(info):
                    with open(info) as fh:
                        if not json.load(fh).get("has_downbeats", True):
                            beat_only_datasets.add(name)
        self.beat_only = set(beat_only_datasets)
        self._handles = {}

        test_fold = (validation_fold + 1) % num_folds
        self.items = []
        for name in datasets:
            split = os.path.join(annot_root, name, "8-folds.split")
            if not os.path.exists(split):
                # Test-only datasets ship no folds - gtzan is held out entirely in this
                # corpus, exactly as the paper's protocol requires. Never train on them;
                # expose every track to val/test so the held-out number is over the whole
                # set rather than an arbitrary eighth of it.
                if subset == "train":
                    continue
                beats_dir = os.path.join(annot_root, name, "annotations", "beats")
                for fn in sorted(os.listdir(beats_dir)):
                    if fn.endswith(".beats"):
                        self.items.append((name, fn[:-len(".beats")]))
                continue
            for line in open(split):
                line = line.strip()
                if not line:
                    continue
                track, fold = line.split()[0], int(line.split()[1])
                if subset == "train" and fold in (validation_fold, test_fold):
                    continue
                if subset == "val" and fold != validation_fold:
                    continue
                if subset == "test" and fold != test_fold:
                    continue
                self.items.append((name, track))
        if not self.items:
            raise RuntimeError(f"no items for subset={subset} over {datasets}")

    def __len__(self):
        return len(self.items)

    def _npz(self, name):
        # opened lazily and cached per worker: np.load on a multi-GB npz is cheap
        # (it memory-maps the zip index) but doing it per item is not.
        if name not in self._handles:
            self._handles[name] = np.load(
                os.path.join(self.spect_root, f"{name}.npz"), allow_pickle=True)
        return self._handles[name]

    def _beats(self, name, track):
        path = os.path.join(self.annot_root, name, "annotations", "beats", f"{track}.beats")
        times, downbeats = [], []
        for line in open(path):
            parts = line.split()
            if not parts:
                continue
            t = float(parts[0])
            times.append(t)
            if len(parts) > 1 and int(float(parts[1])) == 1:
                downbeats.append(t)
        return np.array(times), np.array(downbeats)

    def __getitem__(self, index):
        name, track = self.items[index]
        z = self._npz(name)

        key, scale = f"{track}/track", 1.0
        if self.subset == "train" and (self.tempo_aug or self.pitch_aug):
            # tempo and pitch are mutually exclusive per item, as in beat_this
            choices = [("ts", p) for p in self.tempo_aug] + [("ps", s) for s in self.pitch_aug]
            kind, amount = random.choice(choices + [(None, 0)])
            if kind is not None and amount != 0:
                candidate = f"{track}/track_{kind}{amount}"
                if candidate in z:
                    key = candidate
                    if kind == "ts":
                        scale = 1.0 / (1.0 + amount / 100.0)

        spect = torch.from_numpy(np.asarray(z[key], dtype=np.float32))   # (T, 128)
        beats, downbeats = self._beats(name, track)
        beats, downbeats = beats * scale, downbeats * scale

        total = spect.shape[0]
        if total > self.target_length:
            start = (random.randint(0, total - self.target_length)
                     if self.subset == "train" else 0)
            spect = spect[start:start + self.target_length]
        else:
            start = 0
            spect = torch.nn.functional.pad(spect, (0, 0, 0, self.target_length - total))
        offset = start / self.fps

        if self.subset == "train" and self.mask_aug:
            spect = _augment_mask(spect, self.fps, **_MASK_DEFAULTS)

        target = torch.zeros(2, self.target_length)
        for t in beats - offset:
            f = int(round(t * self.fps))
            if 0 <= f < self.target_length:
                target[0, f] = 1
        if name not in self.beat_only:
            for t in downbeats - offset:
                f = int(round(t * self.fps))
                if 0 <= f < self.target_length:
                    target[1, f] = 1
                    target[0, f] = 1   # a downbeat is also a beat, as our parser assumes

        annot = _make_intervals(target, beat_only=name in self.beat_only)
        if self.subset == "train":
            return spect, annot
        return spect, annot, {"Filename": f"{name}/{track}", "Genre": name,
                              "Time signature": None}


def _make_intervals(target, beat_only):
    """(2, T) frame target -> (M, 3) intervals, matching BeatDataset.make_intervals.

    Reimplemented rather than reused because that method reads self.dataset and, more
    importantly, returns EMPTY whenever the downbeat channel holds fewer than two marks.
    Their parser sets the downbeat channel even for beat-only datasets and discards it
    later; this loader leaves it genuinely empty, so calling the original would silently
    drop every SMC fragment.

    Semantics preserved: consecutive-event intervals, class 0 for downbeats and 1 for
    beats, or class 2 (CLASS_BEAT_ONLY) alone when the dataset has no downbeat labels -
    the marker the subset head uses to take the section 7.1 marginal/EM path.
    """
    def chain(locations, class_id):
        if locations.numel() < 2:
            return torch.zeros((0, 3))
        left = locations[:-1].float()
        right = locations[1:].float()
        out = torch.zeros((left.numel(), 3))
        out[:, 0], out[:, 1], out[:, 2] = left, right, class_id
        return out

    beats = torch.nonzero(target[0]).flatten()
    if beat_only:
        return chain(beats, CLASS_BEAT_ONLY)
    downbeats = torch.nonzero(target[1]).flatten()
    return torch.cat((chain(downbeats, 0), chain(beats, 1)), axis=0)
