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
args = p.parse_args()

SR, DSF = 22050, 512
TO_SEC = DSF / SR
WINDOW = 160 * 8 * DSF          # same 29.7 s window the subset arms train on
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

def build(subset):
    sets = []
    for name, (adir, ldir) in DATA.items():
        sets.append(BeatDataset(adir, ldir, dataset=name, audio_sample_rate=SR,
            audio_downsampling_factor=DSF, subset=subset, augment=False, half=True,
            preload=True, length=WINDOW, dry_run=False, spectral=True,
            validation_fold=args.validation_fold))
    return sets

def intervals_to_frames(annot, T):
    """(M,3) intervals -> (2,T) soft targets, widened to +-2 frames with 0.5/0.25
    weights exactly as Bock/Beat Transformer do."""
    y = torch.zeros(2, T)
    for iv in annot:
        lab = int(iv[2])
        if lab < 0: continue
        f = int(iv[0])
        if not (0 <= f < T): continue
        y[0, f] = 1.0                      # channel 0: beat (downbeats included)
        if lab == 0: y[1, f] = 1.0         # channel 1: downbeat
    for ch in range(2):
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
    b, d = [], []
    for iv in annot:
        lab = int(iv[2])
        if lab < 0: continue
        b.append(int(iv[0]) * TO_SEC)
        if lab == 0: d.append(int(iv[0]) * TO_SEC)
    return np.sort(np.array(b)), np.sort(np.array(d))

def fscore(ref, est):
    if len(ref) < 3 or len(est) < 3: return None
    return mir_eval.beat.evaluate(mir_eval.beat.trim_beats(ref), mir_eval.beat.trim_beats(est))['F-measure']

model = BTBaseline().cuda()
print(f"[bt-baseline] params {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
      f"(dmodel={args.dmodel} d_hid={args.d_hid} nhead={args.nhead})")
opt = Lookahead(torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4), k=5, alpha=0.5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5, factor=0.5, min_lr=1e-6)
bce = nn.BCEWithLogitsLoss()

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
        loss = bce(logits[..., 0], y[:, 0]) + bce(logits[..., 1], y[:, 1])
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
        for name, ds in zip(DATA.keys(), val_sets):
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
