"""Where is the error? Decompose validation performance into its component failures.

A joint F-measure tells you the model is bad, not which part is bad. This splits it
into the four things that have to go right for one ground-truth event to be scored
correct, so a flat F can be attributed rather than guessed at:

  A. structural ceiling  - is there ANY candidate within the +-70 ms tolerance?
                           If this is low, N or the window is wrong.
  B. localisation        - does the candidate the order-preserving match actually
                           assigns land within 70 ms? A large A-to-B gap means the
                           candidate times cannot be put in one-to-one correspondence
                           with the events even though individually they are close.
  C. firing              - does that matched candidate predict anything but background?
  D. class               - given it fired, is it the right one of B/DB?

and two checks on the time branch specifically:

  t_hat spread across songs  - how much t_hat moves when the INPUT changes. Near zero
                               means the regression head is input-independent: it has
                               learned a fixed grid, not event positions.
  |t_hat - uniform|          - how far t_hat has travelled from its initialisation.

Run over several checkpoints to see trends; a component that is flat across epochs is
not "still warming up", it is stuck.

One warning, learned the hard way. The ground truth MUST come from
subset_head.intervals_to_events, as it does below. The raw (M, 3) annotation rows are
two concatenated interval chains -- every downbeat interval, then every beat interval --
so reading column 0 directly gives a list that ascends, restarts, and ascends again.
Feeding that to an order-preserving DP pins B at a fixed value (72% here) no matter what
the model does, which is indistinguishable from a genuinely frozen component and led to
exactly that wrong diagnosis before this was caught.

    python diagnose_bottleneck.py --checkpoints ckpt/*.pt --spect_root ... --spect_annot_root ...
"""
import argparse
import glob
import os

import numpy as np
import torch

from alignbeat import model_module
from alignbeat.bt_dataset import BeatThisSpectDataset
from alignbeat.dataloader import collater
from alignbeat.subset_head import (BACKGROUND, BEAT, DOWNBEAT,
                                   intervals_to_events, subset_select_dp)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', nargs='+', required=True)
parser.add_argument('--spect_root', required=True)
parser.add_argument('--spect_annot_root', required=True)
parser.add_argument('--datasets', default='ballroom,hainsworth,rwc')
parser.add_argument('--songs_per_dataset', type=int, default=20)
parser.add_argument('--window_frames', type=int, default=1500)
parser.add_argument('--n_min', type=int, default=172)
parser.add_argument('--fps', type=float, default=50.0)
parser.add_argument('--tolerance_ms', type=float, default=70.0)
parser.add_argument('--validation_fold', type=int, default=0)
parser.add_argument('--class_attention_layers', type=int, default=1)
parser.add_argument('--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

WIN, FPS = args.window_frames, args.fps
TOL = args.tolerance_ms / 1000.0
to_ms = (WIN / FPS) * 1000.0

batches = []
for name in [d.strip() for d in args.datasets.split(',') if d.strip()]:
    ds = BeatThisSpectDataset(args.spect_root, args.spect_annot_root, [name],
                              subset='val', validation_fold=args.validation_fold,
                              target_length=WIN)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collater)
    for i, b in enumerate(loader):
        if i >= args.songs_per_dataset:
            break
        batches.append((b[0], b[1]))
print(f"[diagnose] {len(batches)} val songs from {args.datasets}")

model = model_module.create_alignbeat_model(
    args=None, encoder_input_frames=WIN, n_min=args.n_min,
    audio_downsampling_factor=int(round(22050 / FPS)),
    dropout={"frontend": 0.1, "transformer": 0.2},
    class_attention_layers=args.class_attention_layers).cuda().eval()

paths = sorted(p for pattern in args.checkpoints for p in glob.glob(pattern))
print(f"\n{'ckpt':>14} | {'A any':>6} {'B loc':>6} {'C fire':>6} {'D cls':>6} | "
      f"{'DB ok':>6} {'DB->bg':>6} | {'spread':>7} {'|-unif|':>7}")
print("-" * 88)
for path in paths:
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in state.items()},
                          strict=False)
    n = reach = loc = fired = right = db_n = db_ok = db_bg = 0
    times = []
    for spect, target in batches:
        with torch.no_grad():
            logits, t_hat = model(spect.cuda())
        logits = logits[0].float().cpu()
        t_hat = t_hat[0].float().cpu().numpy()
        times.append(t_hat)

        # Use the SAME conversion the training loss uses. The raw (M, 3) rows are two
        # concatenated interval chains -- every downbeat interval, then every beat
        # interval -- so reading column 0 directly yields a NON-MONOTONE list that
        # restarts partway through. Feeding that to an order-preserving DP guarantees a
        # large fixed failure rate regardless of the model, which reads exactly like a
        # frozen metric. intervals_to_events merges the chains and applies the
        # downbeat-wins exclusivity rule.
        events = intervals_to_events(target, WIN)[0]
        gt_t = events['times'].numpy()
        gt_c = np.where(events['classes'].numpy() == DOWNBEAT, 0, 1)
        M = len(gt_t)
        if M < 2 or M > len(t_hat):
            continue

        distance = np.abs(gt_t[:, None] - t_hat[None, :])
        n += M
        reach += int((distance.min(axis=1) <= TOL).sum())
        # match on TIME ALONE: isolates localisation from the class term
        sigma = subset_select_dp(distance)
        loc += int((np.abs(gt_t - t_hat[sigma]) <= TOL).sum())

        predicted = logits.argmax(-1).numpy()[sigma]
        wanted = np.where(gt_c == 0, DOWNBEAT, BEAT)
        active = predicted != BACKGROUND
        fired += int(active.sum())
        right += int((predicted[active] == wanted[active]).sum())
        downbeat = gt_c == 0
        db_n += int(downbeat.sum())
        db_ok += int((predicted[downbeat] == DOWNBEAT).sum())
        db_bg += int((predicted[downbeat] == BACKGROUND).sum())

    stacked = np.stack(times)
    uniform = np.arange(1, stacked.shape[1] + 1) / stacked.shape[1]
    spread = stacked.std(axis=0).mean() * to_ms
    drift = np.abs(stacked - uniform).mean() * to_ms
    pct = lambda a, b: 100 * a / max(b, 1)
    print(f"{os.path.basename(path):>14} | {pct(reach,n):5.1f}% {pct(loc,n):5.1f}% "
          f"{pct(fired,n):5.1f}% {pct(right,fired):5.1f}% | "
          f"{pct(db_ok,db_n):5.1f}% {pct(db_bg,db_n):5.1f}% | "
          f"{spread:6.1f}ms {drift:6.1f}ms")

print("\nA = some candidate within tolerance   B = the MATCHED one is within tolerance")
print("C = matched candidate is not background   D = its class is right, given it fired")
print("spread = how much t_hat moves when the input changes (near 0 = input-independent)")
print("|-unif| = how far t_hat has moved from its uniform initialisation")
