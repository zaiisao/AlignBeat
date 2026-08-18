#!/usr/bin/env python3
"""Vanilla Beat Transformer (non-demix) baseline on OUR data/folds.

Why: every number in Beat Transformer's Table 1 has a madmom DBN behind it, and our
subset head produces discrete candidates rather than frame activations, so a DBN cannot
be bolted onto it fairly. That leaves us with no like-for-like reference. This trains
the reference model's own recipe - DSA encoder + out_linear beat/downbeat head, BCE on
widened frame targets - on our five datasets and fold 0, decoded by peak-picking. It is
the honest "same data, same folds, no post-processing" comparison for the subset head.

Deviations from the paper, all deliberate and recorded:
  - non-demix (single mel), because we have no demixed npz and no spleeter. This is the
    ablation their own repo ships as code/ablation_models/non_demix_model.py.
  - no tempo branch. Measured today: their tempo head is 65% Accuracy1 on unseen GTZAN
    and 53% on slow music, and the paper reports no ablation for it.
  - our 29.7s windows and batch 16 rather than whole-sequence batch 1, so the comparison
    against the subset arms holds everything except the head constant.
"""
import argparse, glob, os, re, sys
import numpy as np, torch, torch.nn as nn, mir_eval
from beatfcos.beat_transformer_encoder import BeatTransformerEncoder
from beatfcos.dataloader import BeatDataset, collater
from beatfcos.optim import Lookahead
from beatfcos.gt_utils import intervals_to_times

p = argparse.ArgumentParser()
p.add_argument('--dmodel', type=int, default=256)
p.add_argument('--d_hid', type=int, default=1024)
p.add_argument('--nhead', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--num_workers', type=int, default=6)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--validation_fold', type=int, default=0)
p.add_argument('--threshold', type=float, default=0.5)
p.add_argument('--checkpoint_dir', type=str, default='./checkpoints_bt_baseline')
p.add_argument('--log_file', type=str, default='./bt_baseline.log')
# Beat This spectrogram corpus. Lets this same BCE baseline be trained on THEIR corpus
# as well as ours, which is what makes Beat Transformer and Beat This comparable at all:
# today they differ in architecture AND in training data, so neither comparison isolates
# either factor. 50 fps (hop 441) instead of our 43.07, so the frame<->second constants
# below switch with it.
# --with_smc adds SMC to the AUDIO pipeline's dataset list. Point of the flag: our
# subset head loses 0.085 macro joint when SMC is added (0.775 -> 0.690) while the two
# FCOS heads lose only ~0.033 (0.620 -> 0.586, 0.633 -> 0.601). The composition penalty
# (SMC entering the average at ~0.45, and contributing nothing to the downbeat mean) is
# identical for every head, so the ~0.05 excess looks specific to the subset head - most
# likely its beat-only path (CLASS_BEAT_ONLY -> section 7.1 marginal + EM pseudo-label),
# which the FCOS heads do not have. This BCE head IS Beat Transformer's non-demix head,
# so running it with and without SMC says whether the excess is ours or the data's.
p.add_argument('--with_smc', action='store_true', default=False)
p.add_argument('--spect_root', type=str, default=None)
p.add_argument('--spect_annot_root', type=str,
               default='/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations')
p.add_argument('--spect_datasets', type=str,
               default='ballroom,hainsworth,rwc,harmonix,simac,smc,asap,beatles,'
                       'candombe,filosax,groove_midi,guitarset,hjdb,jaah,tapcorrect,gtzan')
p.add_argument('--spect_tempo_aug', type=str, default='')
p.add_argument('--spect_pitch_aug', type=str, default='')
args = p.parse_args()

SR = 22050
DSF = 441 if args.spect_root else 512      # 50 fps for their corpus, 43.07 for ours
TO_SEC = DSF / SR
WINDOW = 160 * 8 * DSF          # 29.7 s: the window the subset arms TRAIN on
# Evaluation must use the same excerpt length the subset head is evaluated with
# (evaluate_all_datasets.py passes 2097152 = 4096 frames = 95.1 s). dataloader.py's
# spectral branch truncates val items to target_len with no guard, so passing WINDOW
# here silently scored this baseline on 29.7 s of each song while our head saw 95 s -
# which is where the entire reported advantage of the subset head came from.
EVAL_LENGTH = 2097152
FRAMES = WINDOW // DSF

class Tee:
    def __init__(self, path):
        self.f = open(path, 'a'); self.out = sys.stdout
    def write(self, s): self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.out.flush(); self.f.flush()
sys.stdout = Tee(args.log_file)
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.cuda.set_device(args.gpu)

DATA = {
    "ballroom":    ("/disk1/taegum/mnt/labeled_data/ballroom/data", "./dataset_folds/ballroom/label"),
    "hainsworth":  ("/disk1/taegum/mnt/labeled_data/hains/data", "./dataset_folds/hainsworth/label"),
    "rwc_popular": ("/disk1/taegum/mnt/labeled_data/rwc_popular/data", "./dataset_folds/rwc_popular/label"),
    "carnatic":    ("/disk4/taegum/carnatic/data", "./dataset_folds/carnatic/label"),
    "harmonix":    ("/disk4/taegum/harmonix_griffinlim/audio", "./dataset_folds/harmonix/label"),
}
if args.with_smc:
    DATA["smc"] = ("/disk1/taegum/mnt/SMC_MIREX/SMC_MIREX/SMC_MIREX_Audio",
                   "./dataset_folds/smc/label")

DATASET_NAMES = list(DATA.keys())   # replaced by build() when --spect_root is used


def build(subset):
    global DATASET_NAMES
    if args.spect_root:
        from beatfcos.bt_dataset import BeatThisSpectDataset
        frames = (WINDOW if subset in ("train", "full-train") else EVAL_LENGTH) // DSF
        names = [d.strip() for d in args.spect_datasets.split(',') if d.strip()]
        tempo = tuple(int(v) for v in args.spect_tempo_aug.split(',') if v.strip())
        pitch = tuple(int(v) for v in args.spect_pitch_aug.split(',') if v.strip())
        out, kept = [], []
        for n in names:
            try:
                out.append(BeatThisSpectDataset(
                    args.spect_root, args.spect_annot_root, [n],
                    subset=("train" if subset in ("train", "full-train") else "val"),
                    validation_fold=args.validation_fold, target_length=frames,
                    tempo_aug=(tempo if subset.startswith("train") else ()),
                    pitch_aug=(pitch if subset.startswith("train") else ())))
                kept.append(n)
            except RuntimeError:
                continue          # e.g. gtzan has no train split: test-only by design
        # The eval loop zips these against dataset names; with the corpus loader the set
        # differs from DATA.keys() (more datasets, and gtzan drops out of train), so the
        # per-dataset rows would otherwise carry the wrong labels.
        DATASET_NAMES = kept
        return out
    sets = []
    for name, (adir, ldir) in DATA.items():
        sets.append(BeatDataset(adir, ldir, dataset=name, audio_sample_rate=SR,
            audio_downsampling_factor=DSF, subset=subset, augment=False, half=True,
            preload=True, length=(WINDOW if subset in ("train", "full-train") else EVAL_LENGTH),
            dry_run=False, spectral=True,
            validation_fold=args.validation_fold))
    return sets

def intervals_to_frames(annot, T):
    """(M,3) intervals -> (2,T) soft targets, widened to +-2 frames with 0.5/0.25
    weights exactly as Bock/Beat Transformer do."""
    y = torch.zeros(2, T)
    # Beat-only datasets (SMC) carry CLASS_BEAT_ONLY and have NO downbeat annotation.
    # Leaving the downbeat channel at 0 trains "not a downbeat" on every SMC beat -
    # about one in four of which really is one - i.e. thousands of false negatives.
    # Beat Transformer does not do this: spectrogram_dataset.py sets the downbeat target
    # to MASK_VALUE = -1 for smc/musicnet and train.py:170-172 builds a weight that
    # zeroes the loss wherever gt == -1. Mirror that, or this baseline measures a bug of
    # ours rather than their method.
    labs = [int(iv[2]) for iv in annot if int(iv[2]) >= 0]
    if labs and all(l == 2 for l in labs):
        y[1, :] = -1.0
    for iv in annot:
        lab = int(iv[2])
        if lab < 0: continue
        f = int(iv[0])
        if not (0 <= f < T): continue
        y[0, f] = 1.0                      # channel 0: beat (downbeats included)
        if lab == 0: y[1, f] = 1.0         # channel 1: downbeat
    for ch in range(2):
        if float(y[ch].min()) < 0:      # masked channel: nothing to widen
            continue
        base = y[ch].clone()
        for d, w in ((1, 0.5), (2, 0.25)):
            y[ch, d:] = torch.maximum(y[ch, d:], base[:-d] * w)
            y[ch, :-d] = torch.maximum(y[ch, :-d], base[d:] * w)
    return y

class BTBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BeatTransformerEncoder(dmodel=args.dmodel, nhead=args.nhead,
                                              d_hid=args.d_hid, nlayers=9, attn_len=5, dropout=0.1)
        self.out_linear = nn.Linear(args.dmodel, 2)   # their beat/downbeat head
    def forward(self, mel):
        _c1, _c2, c3 = self.encoder(mel)              # deepest tap = their final layer
        return self.out_linear(c3)                    # (B, T, 2) logits

def peak_pick(a, thr):
    idx = np.where(a > thr)[0]
    keep = [i for i in idx if (i == 0 or a[i] >= a[i-1]) and (i == len(a)-1 or a[i] >= a[i+1])]
    return np.array(keep) * TO_SEC

def gt_times(annot):
    # Delegates to the shared reconstruction. The inline version here appended every
    # label to the beat list, which double-counted downbeats (make_intervals emits a
    # downbeat chain AND a beat chain and downbeats appear in both): 20.9% duplicate
    # references on ballroom val, unmatchable by mir_eval, so this baseline was scored
    # against a corrupted reference while beat_eval.py scored the subset head against a
    # correct one - and the two were then compared.
    return intervals_to_times(annot, TO_SEC)

def fscore(ref, est):
    if len(ref) < 3 or len(est) < 3: return None
    return mir_eval.beat.evaluate(mir_eval.beat.trim_beats(ref), mir_eval.beat.trim_beats(est))['F-measure']

model = BTBaseline().cuda()
print(f"[bt-baseline] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
      f"(dmodel={args.dmodel} d_hid={args.d_hid} nhead={args.nhead})")
opt = Lookahead(torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4), k=5, alpha=0.5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5, factor=0.5, min_lr=1e-6)
bce = nn.BCEWithLogitsLoss()
bce_none = nn.BCEWithLogitsLoss(reduction='none')

train_sets, val_sets = build("train"), build("val")
train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_sets),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collater)

best, start_epoch = 0.0, 0
_cks = glob.glob(os.path.join(args.checkpoint_dir, "btbase_*.pt"))
if _cks:
    _cks.sort(key=lambda q: int(re.search(r"btbase_(\d+).pt", q).group(1)))
    model.load_state_dict(torch.load(_cks[-1], map_location='cuda'))
    start_epoch = int(re.search(r"btbase_(\d+).pt", _cks[-1]).group(1)) + 1
    print(f"[bt-baseline] resumed from {_cks[-1]} at epoch {start_epoch}", flush=True)

for epoch in range(start_epoch, args.epochs):
    model.train(); losses = []
    for audio, annots in train_loader:
        mel = audio.squeeze(1).cuda() if audio.dim() == 4 else audio.cuda()
        T = mel.shape[1]
        y = torch.stack([intervals_to_frames(a, T) for a in annots]).cuda()   # (B,2,T)
        logits = model(mel)                                                   # (B,T,2)
        # Masked BCE on the downbeat channel: -1 marks "no annotation" and must not
        # contribute, exactly as Beat Transformer's weighted loss does.
        db_mask = (y[:, 1] >= 0).float()
        db_terms = bce_none(logits[..., 1], torch.clamp(y[:, 1], min=0.0)) * db_mask
        loss = bce(logits[..., 0], y[:, 0]) + db_terms.sum() / db_mask.sum().clamp(min=1.0)
        # train.py가 subset head에 대해 갖고 있는 것과 같은 가드. 이게 없어서 첫 시도가
        # epoch 31에서 NaN에 빠진 뒤 6 에폭 동안 loss=nan / score=0으로 조용히 돌았음.
        if not torch.isfinite(loss):
            print(f"[bt-baseline] WARNING: non-finite loss at epoch {epoch}; skipping step", flush=True)
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            print(f"[bt-baseline] WARNING: non-finite grad norm at epoch {epoch}; skipping step", flush=True)
            opt.zero_grad(); continue
        opt.step(); losses.append(float(loss))
    model.eval(); per = {}
    with torch.no_grad():
        for name, ds in zip(DATASET_NAMES, val_sets):
            bs, ds_ = [], []
            for i in range(len(ds)):
                audio, annot, _meta = ds[i]
                mel = audio.unsqueeze(0).cuda()
                act = torch.sigmoid(model(mel))[0].cpu().numpy()
                gb, gd = gt_times(annot)
                fb = fscore(gb, peak_pick(act[:, 0], args.threshold))
                fd = fscore(gd, peak_pick(act[:, 1], args.threshold))
                if fb is not None: bs.append(fb)
                if fd is not None: ds_.append(fd)
            per[name] = (float(np.mean(bs)) if bs else 0.0, float(np.mean(ds_)) if ds_ else 0.0)
            print(f"Epoch = {epoch} | [{name}] Beat: {per[name][0]:.3f} | Downbeat: {per[name][1]:.3f}")
    beat = float(np.mean([v[0] for v in per.values()]))
    down = float(np.mean([v[1] for v in per.values()]))
    joint = (beat + down) / 2
    print(f"Epoch = {epoch} | Beat score: {beat:.3f} | Downbeat score: {down:.3f} | "
          f"Joint score: {joint:.3f} | train loss {np.mean(losses):.5f}")
    sched.step(joint)
    if joint > best:
        print(f"Joint score of {joint:.3f} exceeded previous best at {best:.3f}")
        best = joint
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"btbase_{epoch}.pt"))
