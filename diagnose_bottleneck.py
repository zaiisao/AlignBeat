"""Where is the error? Decompose validation performance into its component failures."""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from alignbeat.classes import BACKGROUND, DOWNBEAT
from alignbeat.dp import subset_select_dp
from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', nargs='+', required=True)
parser.add_argument('--songs', type=int, default=60,
                    help='number of validation excerpts to diagnose over')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--tolerance_ms', type=float, default=70.0)
parser.add_argument('--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

paths = sorted(p for pattern in args.checkpoints for p in glob.glob(pattern))
if not paths:
    raise SystemExit('no checkpoints matched')

# Model and data both come from the FIRST checkpoint's own saved hyperparameters, the
# same way launch_scripts/compute_paper_metrics.py sets them up. That is what keeps the
# diagnostic on the fold, window and augmentation settings the run actually trained
# with, rather than on defaults restated here that could silently drift from them.
first = torch.load(paths[0], map_location='cpu', weights_only=False)
model = PLBeatThis(**first['hyper_parameters'])
if model.subset_criterion is None:
    raise SystemExit('checkpoint is not a subset-head run (head_type != "subset")')

dm_hparams = dict(first['datamodule_hyper_parameters'])
dm_hparams['num_workers'] = args.num_workers
dm_hparams['data_dir'] = Path(__file__).parent / 'data'
datamodule = BeatDataModule(**dm_hparams)
datamodule.setup(stage='fit')

batches = []
for batch in datamodule.val_dataloader():
    batches.append(batch)
    if sum(len(b['spect']) for b in batches) >= args.songs:
        break
n_songs = sum(len(b['spect']) for b in batches)
print(f"[diagnose] {n_songs} validation excerpts, fold {dm_hparams.get('fold')}")

model = model.cuda().eval()
WIN = dm_hparams.get('train_length', 1500)
FPS = first['hyper_parameters'].get('fps', 50)
window_seconds = WIN / FPS
# t_hat and the targets both live on eq. (1)'s normalised (0, 1] axis, so the tolerance
# has to be normalised by the excerpt duration too rather than compared in seconds.
TOL = (args.tolerance_ms / 1000.0) / window_seconds
to_ms = window_seconds * 1000.0

def label(path):
    """Show the END of the name, not the start."""
    return os.path.basename(path).replace('.ckpt', '')[-22:]


print(f"\n{'ckpt':>22} | {'A any':>6} {'B loc':>6} {'C fire':>6} {'D cls':>6} | "
      f"{'DB rec':>6} {'DB pre':>6} {'DB F1':>6} {'DB->bg':>6} | {'spread':>7} {'|-unif|':>7}")
print("-" * 112)
for path in paths:
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=False)
    n = reach = loc = fired = right = db_n = db_ok = db_bg = 0
    db_claimed = db_true = 0
    times = []
    for batch in batches:
        # The datamodule yields float16 spectrograms because training runs under
        # the trainer's "16-mixed" autocast; a standalone script gets no autocast,
        # so cast explicitly rather than hitting a Half/Float conv mismatch.
        spect = batch['spect'].cuda().float()
        with torch.no_grad():
            prediction = model.model(spect)
        # The SAME conversion the training loss uses -- see the warning in the module
        # docstring. _subset_targets returns normalised times and B/DB/UNKNOWN classes.
        targets = model._subset_targets({**batch, 'spect': spect})
        logits = prediction['class_logits'].float().cpu()
        t_hat_all = prediction['t_hat'].float().cpu().numpy()
        for index, target in enumerate(targets):
            t_hat = t_hat_all[index]
            times.append(t_hat)
            gt_t = target['times'].cpu().numpy()
            gt_c = target['classes'].cpu().numpy()
            M = len(gt_t)
            if M < 2 or M > len(t_hat):
                continue

            distance = np.abs(gt_t[:, None] - t_hat[None, :])
            n += M
            reach += int((distance.min(axis=1) <= TOL).sum())
            # match on TIME ALONE: isolates localisation from the class term
            sigma = subset_select_dp(distance)
            loc += int((np.abs(gt_t - t_hat[sigma]) <= TOL).sum())

            predicted = logits[index].argmax(-1).numpy()[sigma]
            active = predicted != BACKGROUND
            fired += int(active.sum())
            right += int((predicted[active] == gt_c[active]).sum())
            downbeat = gt_c == DOWNBEAT
            db_n += int(downbeat.sum())
            db_ok += int((predicted[downbeat] == DOWNBEAT).sum())
            db_bg += int((predicted[downbeat] == BACKGROUND).sum())
            # ...and how many of the DOWNBEATS IT CLAIMS are real: raising omega_DB buys
            # recall by calling more things downbeats, so recall alone cannot say whether
            # it helped.
            claimed = predicted == DOWNBEAT
            db_claimed += int(claimed.sum())
            db_true += int((gt_c[claimed] == DOWNBEAT).sum())

    stacked = np.stack(times)
    uniform = np.arange(1, stacked.shape[1] + 1) / stacked.shape[1]
    spread = stacked.std(axis=0).mean() * to_ms
    drift = np.abs(stacked - uniform).mean() * to_ms
    pct = lambda a, b: 100 * a / max(b, 1)
    recall, precision = pct(db_ok, db_n), pct(db_true, db_claimed)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    print(f"{label(path):>22} | {pct(reach,n):5.1f}% {pct(loc,n):5.1f}% "
          f"{pct(fired,n):5.1f}% {pct(right,fired):5.1f}% | "
          f"{recall:5.1f}% {precision:5.1f}% {f1:5.1f}% {pct(db_bg,db_n):5.1f}% | "
          f"{spread:6.1f}ms {drift:6.1f}ms")

print("\nDB rec = of true downbeats, called downbeat   DB pre = of claimed downbeats, real")
print("A = some candidate within tolerance   B = the MATCHED one is within tolerance")
print("C = matched candidate is not background   D = its class is right, given it fired")
print("spread = how much t_hat moves when the input changes (near 0 = input-independent)")
print("|-unif| = how far t_hat has moved from its uniform initialisation")
