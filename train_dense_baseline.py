"""Dense frame-classification baseline (Section 12 of the paper).

This is the "Beat This!" (Foscarin, Schlueter & Widmer, ISMIR 2024) style
reference system, implemented so that the ONLY difference from the subset-head
model of train.py is the head: the encoder, the dataset, the frame rate and the
mir_eval scoring path are all the repo's own.

Section 12 describes four devices, all implemented below:

  1. Architecture (class DenseFrameClassifier): the repo's own
     alignbeat.beat_this_encoder.BeatThisEncoder -- a conv + partial-transformer
     frontend feeding a full self-attention transformer -- with a single Linear
     head emitting two per-frame logits (beat, downbeat) at the spectrogram
     frame rate (50 fps).  Deliberately NOT a dilated / DSA "Beat Transformer":
     that is a different paper's architecture and using it would confound the
     comparison, which is meant to isolate the head.

  2. Shift-tolerant loss (shift_tolerant_bce): the PREDICTIONS are max-pooled
     over a small window before the BCE against frame-synchronous binary
     targets, so only the largest nearby prediction is penalized and small
     annotation-timing shifts go unpunished.  The targets are NOT widened with
     Boeck-style +-2-frame soft labels; that is the other architecture's scheme.

  3. Class imbalance (pos_weight_from_targets): the positive-frame term of the
     BCE is weighted by the ratio of negative to positive frame counts, counted
     from the actual targets and passed as BCEWithLogitsLoss's `pos_weight`.

  4. Beat/downbeat consistency: a "Sum Head" (DenseFrameClassifier.forward)
     adds the downbeat logit into the beat logit before thresholding, and a
     final postprocessing step (snap_downbeats_to_beats) snaps every predicted
     downbeat to its nearest predicted beat.

Decoding is sigmoid + local-maximum peak picking above a probability threshold,
then the Sum Head and the snapping above.  No DBN.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from alignbeat.beat_this_encoder import BeatThisEncoder
from alignbeat.bt_dataset import BeatThisSpectDataset
from alignbeat.dataloader import collater, BEAT_ONLY_DATASETS, CLASS_BEAT_ONLY


# --------------------------------------------------------------------------- #
# 1. Architecture
# --------------------------------------------------------------------------- #
class DenseFrameClassifier(nn.Module):
    """(B, T, n_mels) log-mel spectrogram -> (B, T, 2) per-frame logits.

    Channel 0 is beat, channel 1 is downbeat.  The encoder never touches the
    time axis, so the output is frame-synchronous with the input at 50 fps.
    """

    def __init__(self, spect_dim=128, transformer_dim=512, ff_mult=4, n_layers=6,
                 head_dim=32, stem_dim=32, frontend_dropout=0.1,
                 transformer_dropout=0.2, partial_transformers=True, sum_head=True):
        super().__init__()
        self.encoder = BeatThisEncoder(
            spect_dim=spect_dim, transformer_dim=transformer_dim, ff_mult=ff_mult,
            n_layers=n_layers, head_dim=head_dim, stem_dim=stem_dim,
            dropout={"frontend": frontend_dropout, "transformer": transformer_dropout},
            partial_transformers=partial_transformers)
        self.head = nn.Linear(transformer_dim, 2)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.sum_head = sum_head

    def forward(self, spect):
        logits = self.head(self.encoder(spect))          # (B, T, 2)
        beat, downbeat = logits[..., 0], logits[..., 1]
        if self.sum_head:
            # Section 12 device (4a), the "Sum Head": the downbeat logit is
            # added into the beat logit, so that a confident downbeat cannot be
            # thresholded on without also pushing the beat channel over its own
            # threshold.  This is architectural (it is trained through), exactly
            # as in Beat This!, not a decode-time hack.
            beat = beat + downbeat
        return torch.stack((beat, downbeat), dim=-1)


# --------------------------------------------------------------------------- #
# targets: (M, 3) interval annotations -> frame-synchronous binary labels
# --------------------------------------------------------------------------- #
def intervals_to_frame_targets(annots, num_frames):
    """(B, M, 3) collated intervals -> ((B, 2, T) targets, (B,) downbeat mask).

    Mirrors alignbeat.beat_eval's ground-truth extraction exactly: an interval
    contributes its left endpoint, and the largest right endpoint of a class
    contributes the final event of that class.  Class 0 = downbeat,
    1 = beat, CLASS_BEAT_ONLY (2) = "a beat, but this dataset has no downbeat
    labels", -1 = collater padding.  A downbeat is also a beat.

    The returned mask is False for items coming from a beat-only dataset (SMC);
    their downbeat channel carries no supervision and must be excluded from the
    loss rather than trained towards all-zeros.
    """
    batch = annots.shape[0]
    targets = torch.zeros(batch, 2, num_frames, device=annots.device)
    downbeat_mask = torch.ones(batch, dtype=torch.bool, device=annots.device)

    def put(b, channel, frame):
        f = int(round(float(frame)))
        if 0 <= f < num_frames:
            targets[b, channel, f] = 1.0

    for b in range(batch):
        last_beat = last_downbeat = None
        for interval in annots[b]:
            label = int(interval[2])
            if label < 0:
                continue
            left, right = float(interval[0]), float(interval[1])
            if label == 0:
                put(b, 1, left)
                put(b, 0, left)          # a downbeat is also a beat
                last_downbeat = right if last_downbeat is None else max(last_downbeat, right)
            elif label == 1:
                put(b, 0, left)
                last_beat = right if last_beat is None else max(last_beat, right)
            elif label == CLASS_BEAT_ONLY:
                put(b, 0, left)
                last_beat = right if last_beat is None else max(last_beat, right)
                downbeat_mask[b] = False
        if last_beat is not None:
            put(b, 0, last_beat)
        if last_downbeat is not None:
            put(b, 1, last_downbeat)
            put(b, 0, last_downbeat)
    return targets, downbeat_mask


# --------------------------------------------------------------------------- #
# 2. + 3. shift-tolerant, class-balanced BCE
# --------------------------------------------------------------------------- #
def pos_weight_from_targets(targets, valid_mask):
    """Section 12 device (3): negative-to-positive frame-count ratio, per channel.

    Counted over the frames that actually carry supervision (valid_mask), so the
    masked-out downbeat channel of a beat-only item never enters the count.
    """
    with torch.no_grad():
        valid = valid_mask.sum(dim=(0, 2)).clamp(min=1.0)            # (2,)
        pos = (targets * valid_mask).sum(dim=(0, 2))                 # (2,)
        neg = valid - pos
        return (neg / pos.clamp(min=1.0)).clamp(min=1.0)


def shift_tolerant_bce(logits, targets, valid_mask, window):
    """Section 12 device (2): max-pool the PREDICTIONS, then BCE against the
    frame-synchronous binary targets.

    `window` is the (odd) pooling width in frames.  With window=1 this reduces
    to plain per-frame BCE.  Two things happen for window > 1:

      * the prediction at frame t is replaced by the largest prediction within
        +-(window//2) frames, so a peak that is off by a frame or two still
        satisfies the positive target and is not penalized;
      * the negative term is switched off for the frames whose pooling window
        can reach a frame within +-radius of a positive target (i.e. within
        2*radius of it), since after pooling those frames necessarily see the
        (correct, merely shifted) peak and would otherwise be punished for it --
        which is precisely the punishment the device exists to remove.  Without
        this exemption a peak one frame off its annotation still incurs a large
        negative-term loss, and the pooling buys nothing.

    Note that the targets themselves are never widened: no Boeck-style +-2-frame
    soft labels.  The tolerance lives entirely in the pooling of the predictions.
    """
    if window % 2 != 1:
        raise ValueError(f"--max_pool_window must be odd, got {window}")
    radius = window // 2
    if radius > 0:
        pooled = F.max_pool1d(logits, kernel_size=window, stride=1, padding=radius)
        # frames within `radius` of a positive target: exempt from the negative term
        neighbourhood = F.max_pool1d(targets, kernel_size=4 * radius + 1,
                                     stride=1, padding=2 * radius)
        loss_mask = valid_mask * (1.0 - (neighbourhood * (1.0 - targets)))
    else:
        pooled = logits
        loss_mask = valid_mask

    pos_weight = pos_weight_from_targets(targets, valid_mask)
    per_frame = F.binary_cross_entropy_with_logits(
        pooled, targets, pos_weight=pos_weight.view(1, -1, 1), reduction="none")
    per_frame = per_frame * loss_mask
    return per_frame.sum() / loss_mask.sum().clamp(min=1.0)


def compute_loss(logits, annots, window):
    """logits (B, T, 2) + collated (B, M, 3) annotations -> scalar loss."""
    logits = logits.transpose(1, 2)                     # (B, 2, T)
    targets, downbeat_mask = intervals_to_frame_targets(annots, logits.shape[-1])
    valid = torch.ones_like(targets)
    valid[:, 1, :] = downbeat_mask.to(targets.dtype).unsqueeze(1)
    return shift_tolerant_bce(logits, targets, valid, window)


# --------------------------------------------------------------------------- #
# decoding: peak picking + 4b. snapping
# --------------------------------------------------------------------------- #
def pick_peaks(probs, threshold):
    """1-D probabilities -> indices of local maxima strictly above `threshold`."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return np.zeros(0, dtype=np.int64)
    left = np.empty_like(probs)
    right = np.empty_like(probs)
    left[0], left[1:] = -np.inf, probs[:-1]
    right[-1], right[:-1] = -np.inf, probs[1:]
    return np.flatnonzero((probs > threshold) & (probs >= left) & (probs > right))


def snap_downbeats_to_beats(downbeat_times, beat_times):
    """Section 12 device (4b): every predicted downbeat is moved onto its nearest
    predicted beat, guaranteeing a musically valid output in which no downbeat
    exists without a coincident beat.  Duplicates created by snapping two
    downbeats onto the same beat are collapsed."""
    downbeat_times = np.asarray(downbeat_times, dtype=np.float64)
    beat_times = np.asarray(beat_times, dtype=np.float64)
    if downbeat_times.size == 0 or beat_times.size == 0:
        return downbeat_times
    beat_times = np.sort(beat_times)
    idx = np.searchsorted(beat_times, downbeat_times)
    lo = np.clip(idx - 1, 0, beat_times.size - 1)
    hi = np.clip(idx, 0, beat_times.size - 1)
    take_hi = np.abs(beat_times[hi] - downbeat_times) < np.abs(beat_times[lo] - downbeat_times)
    snapped = np.where(take_hi, beat_times[hi], beat_times[lo])
    return np.unique(snapped)


@torch.no_grad()
def predict_logits(model, spect, window_frames):
    """Run the model over a whole piece by tiling it into fixed-length windows.

    Chunks are non-overlapping: the head is per-frame, so unlike the subset head
    there is no correspondence to stitch across a boundary -- only the encoder's
    context is truncated there, which costs at most a frame or two of accuracy at
    each seam.
    """
    total = spect.shape[0]
    outputs = []
    for start in range(0, total, window_frames):
        chunk = spect[start:start + window_frames]
        pad = window_frames - chunk.shape[0]
        if pad > 0:
            chunk = F.pad(chunk, (0, 0, 0, pad))
        out = model(chunk.unsqueeze(0))[0]              # (window_frames, 2)
        outputs.append(out[:window_frames - pad] if pad > 0 else out)
    return torch.cat(outputs, dim=0)                    # (total, 2)


# --------------------------------------------------------------------------- #
# evaluation (mir_eval, identical protocol to alignbeat.beat_eval)
# --------------------------------------------------------------------------- #
def ground_truth_times(annots_single, to_seconds):
    """Copy of alignbeat.beat_eval.evaluate_beat_f_measure_subset's GT extraction,
    so the F-measures here and the subset head's are computed from identical
    reference lists.  Note the beat reference INCLUDES downbeats."""
    beat, downbeat = [], []
    last_beat = last_downbeat = None
    for interval in annots_single:
        label = int(interval[2])
        if label < 0:
            continue
        left, right = int(interval[0]), int(interval[1])
        if label == 0:
            downbeat.append(left * to_seconds)
            if last_downbeat is None or right > last_downbeat:
                last_downbeat = right
        elif label in (1, CLASS_BEAT_ONLY):
            beat.append(left * to_seconds)
            if last_beat is None or right > last_beat:
                last_beat = right
    if last_beat is not None:
        beat.append(last_beat * to_seconds)
    if last_downbeat is not None:
        downbeat.append(last_downbeat * to_seconds)
    return np.sort(np.array(beat)), np.sort(np.array(downbeat))


@torch.no_grad()
def evaluate_dense(dataloader, model, to_seconds, window_frames,
                   threshold_beat, threshold_downbeat, snap=True, label=""):
    model.eval()
    inner = getattr(model, "module", model)
    beat_f, downbeat_f = [], []
    for index, (spect, annots, metadata) in enumerate(dataloader):
        mel = spect[0]
        if torch.cuda.is_available():
            mel = mel.cuda()
        logits = predict_logits(inner, mel, window_frames).float().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))           # Sum Head already applied in forward

        beat_frames = pick_peaks(probs[:, 0], threshold_beat)
        downbeat_frames = pick_peaks(probs[:, 1], threshold_downbeat)
        beat_pred = beat_frames * to_seconds
        downbeat_pred = downbeat_frames * to_seconds
        if snap:
            downbeat_pred = snap_downbeats_to_beats(downbeat_pred, beat_pred)

        beat_gt, downbeat_gt = ground_truth_times(annots[0], to_seconds)
        bs = mir_eval.beat.evaluate(mir_eval.beat.trim_beats(beat_gt),
                                    mir_eval.beat.trim_beats(beat_pred))
        ds = mir_eval.beat.evaluate(mir_eval.beat.trim_beats(downbeat_gt),
                                    mir_eval.beat.trim_beats(downbeat_pred))
        beat_f.append(bs["F-measure"])
        downbeat_f.append(ds["F-measure"])
        print(f"{index}/{len(dataloader)} {metadata[0]['Filename']} | "
              f"BEAT F {bs['F-measure']:0.3f} | DOWNBEAT F {ds['F-measure']:0.3f} | "
              f"pred B/DB: {len(beat_pred)}/{len(downbeat_pred)} | "
              f"gt B/DB: {len(beat_gt)}/{len(downbeat_gt)}")

    beat_mean = float(np.mean(beat_f)) if beat_f else 0.0
    downbeat_mean = float(np.mean(downbeat_f)) if downbeat_f else 0.0
    print(f"{label}Average beat F-measure: {beat_mean:0.3f} | "
          f"downbeat F-measure: {downbeat_mean:0.3f}")
    return beat_mean, downbeat_mean


def beat_only_names(annot_root, names):
    """Which datasets have no downbeat labels, read the same way bt_dataset does:
    our hardcoded BEAT_ONLY_DATASETS (smc) plus anything whose info.json says
    has_downbeats=false (simac in this corpus)."""
    out = set(BEAT_ONLY_DATASETS)
    for name in names:
        info = os.path.join(annot_root, name, "info.json")
        if os.path.exists(info):
            with open(info) as fh:
                if not json.load(fh).get("has_downbeats", True):
                    out.add(name)
    return out


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Dense frame-classification (Beat This!-style) baseline, "
                    "Section 12. Uses the repo's own BeatThisEncoder so the "
                    "comparison against the subset head isolates the head.")
    # data (mirrors train.py's --spect_* flags; this baseline is spectrogram-only)
    p.add_argument('--spect_root', type=str, required=True,
                   help='dir of {dataset}.npz from the Beat This corpus')
    p.add_argument('--spect_annot_root', type=str,
                   default='/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations')
    p.add_argument('--spect_datasets', type=str,
                   default='ballroom,hainsworth,rwc,harmonix,simac,smc,asap',
                   help='comma separated; gtzan is test-only and never enters train')
    p.add_argument('--spect_tempo_aug', type=str, default='')
    p.add_argument('--spect_pitch_aug', type=str, default='')
    p.add_argument('--spect_mask_aug', action='store_true')
    p.add_argument('--audio_sample_rate', type=int, default=22050)
    p.add_argument('--audio_downsampling_factor', type=int, default=441,
                   help='must be 441: the corpus is 50 fps')
    p.add_argument('--train_length', type=int, default=1500 * 441,
                   help='training window in samples (default 1500 frames = 30 s)')
    p.add_argument('--eval_length', type=int, default=2097152)
    p.add_argument('--validation_fold', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--shuffle', type=lambda x: str(x).lower() != 'false', default=True)
    # encoder (same defaults as train.py)
    p.add_argument('--transformer_dim', type=int, default=512)
    p.add_argument('--ff_mult', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--head_dim', type=int, default=32)
    p.add_argument('--stem_dim', type=int, default=32)
    p.add_argument('--beat_this_frontend_dropout', type=float, default=0.1)
    p.add_argument('--beat_this_transformer_dropout', type=float, default=0.2)
    p.add_argument('--partial_transformers', type=lambda x: str(x).lower() != 'false', default=True)
    # Section 12 devices
    p.add_argument('--max_pool_window', type=int, default=3,
                   help='odd width, in frames, of the max-pool applied to the '
                        'PREDICTIONS before the BCE (Section 12 shift tolerance); '
                        '1 disables it')
    p.add_argument('--sum_head', type=lambda x: str(x).lower() != 'false', default=True,
                   help='add the downbeat logit into the beat logit')
    p.add_argument('--snap_downbeats', type=lambda x: str(x).lower() != 'false', default=True,
                   help='snap each predicted downbeat to its nearest predicted beat')
    p.add_argument('--tau_beat', type=float, default=0.5)
    p.add_argument('--tau_downbeat', type=float, default=0.5)
    # optimization / bookkeeping
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--accumulate_grad_batches', type=int, default=1)
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_dense_baseline')
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    if args.audio_downsampling_factor != 441:
        raise SystemExit("--spect_root implies 50 fps: pass --audio_downsampling_factor 441")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    window_frames = args.train_length // args.audio_downsampling_factor
    eval_frames = args.eval_length // args.audio_downsampling_factor
    to_seconds = args.audio_downsampling_factor / args.audio_sample_rate

    names = [d.strip() for d in args.spect_datasets.split(',') if d.strip()]
    tempo = tuple(int(v) for v in args.spect_tempo_aug.split(',') if v.strip())
    pitch = tuple(int(v) for v in args.spect_pitch_aug.split(',') if v.strip())
    no_downbeats = beat_only_names(args.spect_annot_root, names)

    train_sets, val_loaders = [], []
    for name in names:
        train_sets.append(BeatThisSpectDataset(
            args.spect_root, args.spect_annot_root, [name], subset="train",
            validation_fold=args.validation_fold, target_length=window_frames,
            tempo_aug=tempo, pitch_aug=pitch, mask_aug=args.spect_mask_aug))
        val = BeatThisSpectDataset(
            args.spect_root, args.spect_annot_root, [name], subset="val",
            validation_fold=args.validation_fold, target_length=eval_frames)
        val_loaders.append((name, torch.utils.data.DataLoader(
            val, shuffle=False, batch_size=1, num_workers=args.num_workers,
            collate_fn=collater)))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_sets), shuffle=args.shuffle,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collater)
    print(f"[dense] {len(names)} datasets | train {len(train_loader.dataset)} | "
          f"window {window_frames} frames ({window_frames / 50.0:.1f}s at 50 fps) | "
          f"max_pool_window={args.max_pool_window} sum_head={args.sum_head} "
          f"snap={args.snap_downbeats} | beat-only: {sorted(no_downbeats & set(names))}")

    model = DenseFrameClassifier(
        transformer_dim=args.transformer_dim, ff_mult=args.ff_mult,
        n_layers=args.n_layers, head_dim=args.head_dim, stem_dim=args.stem_dim,
        frontend_dropout=args.beat_this_frontend_dropout,
        transformer_dropout=args.beat_this_transformer_dropout,
        partial_transformers=args.partial_transformers, sum_head=args.sum_head)
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # resume: pick the numerically highest epoch, as train.py does (glob order is
    # not sorted, and picking the wrong file silently resumes from an old model)
    ckpts = glob.glob(os.path.join(args.checkpoint_dir, 'dense_[0-9]*.pt'))
    start_epoch, best_joint = 0, 0.0
    if ckpts:
        ckpts.sort(key=lambda p: int(re.search(r"dense_(\d+)\.pt", p).group(1)))
        state = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        best_joint = state.get('best_joint', 0.0)
        print(f"resumed from {ckpts[-1]} (epoch {start_epoch}, best joint {best_joint:.3f})")

    accum = max(1, args.accumulate_grad_batches)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for it, (spect, annots) in enumerate(train_loader):
            spect, annots = spect.to(device), annots.to(device)
            if it % accum == 0:
                optimizer.zero_grad()
            loss = compute_loss(model(spect), annots, args.max_pool_window) / accum
            if not torch.isfinite(loss):
                print(f"epoch {epoch} iter {it}: non-finite loss, skipping")
                optimizer.zero_grad()
                continue
            loss.backward()
            if (it + 1) % accum == 0 or it == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            losses.append(loss.item() * accum)
            if it % 50 == 0:
                print(f"epoch {epoch} | iter {it}/{len(train_loader)} | "
                      f"loss {np.mean(losses[-50:]):.4f}", flush=True)

        # per-epoch macro average: mean over datasets of the per-dataset mean, so
        # a large dataset cannot dominate.  Beat-only datasets (SMC) contribute to
        # the beat average but are excluded from the downbeat one, where their
        # score is structurally 0 for lack of any reference.
        beats, downbeats = [], []
        for name, loader in val_loaders:
            b, d = evaluate_dense(loader, model, to_seconds, window_frames,
                                  args.tau_beat, args.tau_downbeat,
                                  snap=args.snap_downbeats, label=f"[{name}] ")
            beats.append(b)
            if name in no_downbeats:
                print(f"epoch {epoch} | [{name}] Beat {b:0.3f} | Downbeat n/a (beat-only)")
            else:
                downbeats.append(d)
                print(f"epoch {epoch} | [{name}] Beat {b:0.3f} | Downbeat {d:0.3f}")
        beat_macro = float(np.mean(beats)) if beats else 0.0
        downbeat_macro = float(np.mean(downbeats)) if downbeats else 0.0
        joint = (beat_macro + downbeat_macro) / 2
        print(f"epoch {epoch} | train loss {np.mean(losses):.4f} | Beat {beat_macro:0.3f} | "
              f"Downbeat {downbeat_macro:0.3f} | Joint {joint:0.3f}", flush=True)

        scheduler.step(joint)
        if joint > best_joint:
            best_joint = joint
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch,
                        'best_joint': best_joint, 'args': vars(args)},
                       os.path.join(args.checkpoint_dir, f'dense_{epoch}.pt'))
            print(f"saved new best (Joint {joint:0.3f})")


if __name__ == '__main__':
    main()
