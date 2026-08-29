import math
import os
import glob
import torch
import torchsummary
import re
import random
import numpy as np
import collections
from itertools import product
from argparse import ArgumentParser
import traceback
import sys
from os.path import join as ospj
from kmeans_pytorch import kmeans, kmeans_predict

from alignbeat import model_module
from alignbeat.dataloader import collater, BEAT_ONLY_DATASETS
from alignbeat.beat_eval import evaluate_beat_f_measure_subset

class Logger(object):
    """Log stdout messages."""
    def __init__(self, outfile, mode="w"):
        self.terminal = sys.stdout
        self.log = open(outfile, mode)
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

def configure_log(log_file_name, mode="w"):
    Logger(log_file_name, mode)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing over `max_iters` steps with `warmup` linear warmup steps.

    Copied from CPJKU/beat_this (beat_this/model/pl_module.py) so the encoder
    can be trained with its own original schedule (--lr_schedule cosine_warmup),
    keeping a comparison against beat_this's peak-picking head down to the head
    itself rather than confounding it with a different schedule on the shared
    encoder.
    Steps once per training iteration (not per epoch) -- see the
    `--optimizer adamw --lr_schedule cosine_warmup` call site below.
    """

    def __init__(self, optimizer, warmup, max_iters, raise_last=0, raise_to=0.5):
        self.warmup = warmup
        self.max_num_iters = int((1 - raise_last) * max_iters)
        self.raise_to = raise_to
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.max_num_iters:
            progress = step / self.max_num_iters
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
            if step <= self.warmup:
                lr_factor *= step / self.warmup
        else:
            progress = (step - self.max_num_iters) / self.warmup
            lr_factor = self.raise_to * min(progress, 1)
        return lr_factor

# The log.log / GPU / checkpoints paths used to be hardcoded, so running several runs
# at once (8-fold CV, say) had them all overwrite the same log.log and checkpoints/
# directory. --log_file, --gpu and --checkpoint_dir separate them per fold, which is
# why this setup moved below argparse (see just after the args parse).

torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--dataset', type=str, default='ballroom')
parser.add_argument('--audio_sample_rate', type=int, default=22050)
parser.add_argument('--audio_downsampling_factor', type=int, default=512)  # 128 -> 512 (hop_length)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
# default=None: resolved below to T=1500 frames (50 fps, 30s), exactly the paper's
# worked example. An explicit value always wins.
parser.add_argument('--train_length', type=int, default=None)
parser.add_argument('--eval_length', type=int, default=2097152)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
# default=None: resolved to 3e-4 below (an explicit value wins). lambda_L1 = 1/b is
# 200, so the time term's gradient is large (measured grad norm: median 50-160, peak
# 7.7e4) and at 1e-3 even overfitting a single fragment diverged (loss 2.77 -> 2093).
# At 3e-4 / 1e-4 it converges normally, hitting 63/63 events within +-70ms.
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--patience', type=int, default=3)
# Beat Transformer's training recipe (their Section 4.2): RAdam + Lookahead, lr 1e-3,
# decay by 1/5 when validation stalls for 2 epochs, floor 1e-7. What we had been running
# (plain Adam, fixed 3e-4, patience=10 so the scheduler never once fired) was never
# tuned for this encoder or this head. Defaults keep the old behaviour.
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'radam_lookahead', 'adamw'])
parser.add_argument('--lookahead_k', type=int, default=5)
parser.add_argument('--lookahead_alpha', type=float, default=0.5)
parser.add_argument('--lr_factor', type=float, default=0.1)   # torch default
parser.add_argument('--lr_patience', type=int, default=None)  # None falls back to --patience
parser.add_argument('--min_lr', type=float, default=0.0)
# --optimizer adamw: reproduces the optimizer from the beat_this repository
# (beat_this/model/pl_module.py) exactly. weight_decay applies only to tensors of rank
# >= 2 (matrices); bias/norm parameters (rank <= 1) get 0 -- their own comment credits
# this convention to Karpathy's nanoGPT.
parser.add_argument('--adamw_weight_decay', type=float, default=0.01)
# --lr_schedule cosine_warmup: reproduces beat_this's CosineWarmupScheduler (see the
# class above). The default 'plateau' is ReduceLROnPlateau. Turn this on to train the
# beat_this encoder under its original recipe, so that comparing
# "beat_this+peak_picking" against "beat_this+subset_selection" reflects only the head
# (--optimizer adamw --lr_schedule cosine_warmup --lr 0.0008 is their exact recipe).
parser.add_argument('--lr_schedule', type=str, default='plateau', choices=['plateau', 'cosine_warmup'])
parser.add_argument('--warmup_steps', type=int, default=1000)
parser.add_argument('--validation_fold', type=int, default=None)
"""
Registers the values needed to build BeatThisEncoder (beat_this_encoder.py)
and ProgressiveDownsample (progressive_downsample.py) as CLI options.

train.py turns all argparse arguments into a dict via dict_args=vars(args),
then passes the whole thing through **dict_args into the model-creation
function -- so as long as the names match exactly, model_module.py's
kwargs.get('transformer_dim',...) picks up the values automatically, with no
extra wiring needed.
"""
parser.add_argument('--transformer_dim', type=int, default=512)
parser.add_argument('--ff_mult', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--head_dim', type=int, default=32)
parser.add_argument('--stem_dim', type=int, default=32)
parser.add_argument('--beat_this_frontend_dropout', type=float, default=0.1)
parser.add_argument('--beat_this_transformer_dropout', type=float, default=0.2)
parser.add_argument('--partial_transformers', type=lambda x: x.lower() != 'false', default=True)

# b_scale: the scale of the Laplace observation model in eq. (2). lambda_L1 = 1/b, and
# this is not a free weight but the precision of the timing observation noise. Times are
# normalized to (0,1) over the whole window, so the +-70ms tolerance is only
# 70ms/29.72s ~ 0.0024 in normalized units -- set b anywhere near that and lambda_L1
# reaches the hundreds, which turns the DP selection into plain nearest-in-time.
# Tune it from the [subset] diagnostic line in the training log (one-slot time cost vs
# class spread).
parser.add_argument('--b_scale', type=float, default=0.005)
# learn_b: update b by EMA from the eq. (5) closed-form MLE (mean absolute residual over
# matched pairs). Off by default -- confirm training is stable with a fixed b first.
# [caution, measured] Under multi-GPU DataParallel the replicas' buffer updates are
# discarded, so only GPU 0's half-batch residuals reach the EMA (reproduced). On a single
# GPU (one --gpu N) it behaves correctly. Run single-GPU if you enable learn_b.
parser.add_argument('--learn_b', action='store_true', default=False)
# gamma: the weight on the background (unmatched candidate) term of eq. (8). Of the
# N=160 candidates only 60-70 are typically real events, so the remaining ~100 all
# contribute background signal.
parser.add_argument('--gamma', type=float, default=0.5)
# --no_event_norm: compute eq. (8) as the paper writes it -- a plain sum over the
# fragment with no normalization. The current default divides by M and averages over
# fragments, a documented deliberate deviation with one side effect: for a track decoded
# at exactly double tempo the background cost is (M-1)/M * gamma*log2, which saturates
# at ~0.347 independently of M. Under the paper's plain sum it is (M-1)*gamma*log2 and
# grows with M. So if the doubling penalty looks too weak, that is our normalization,
# not the paper's design.
parser.add_argument('--no_event_norm', action='store_true', default=False)
# omega_db: the per-class weight in eq. (8). Downbeats are rarer than beats by roughly a
# factor 1/L, so classification errors on matched downbeats are weighted up.
# omega_beat is fixed at 1.0.
parser.add_argument('--omega_db', type=float, default=2.0)
# cont_weight: the log-variance of the expected event count per sub-window. It is
# invariant to a global doubling (log 2n = log n + const), so it targets tempo
# instability rather than octave errors. Unvalidated.
parser.add_argument('--cont_weight', type=float, default=0.0)
parser.add_argument('--cont_windows', type=int, default=8)
# lambda_r: the periodicity regularizer of Section 10.4, equations (36)-(37). It
# penalizes how far consecutive predicted downbeat intervals stray from L times the
# model's own mean matched interval. 0 disables it. With meter_L=0, L (beats per bar) is
# read off the ground truth per fragment -- measured as ballroom 3.98 / harmonix 4.03 /
# carnatic 6.01, so no dataset metadata is needed.
# (Bar-level diagnostic: on carnatic the model predicts 4.82 beats per bar, i.e. it is
# forcing 4-beat bars onto a 6-beat tala. This term targets exactly that.)
parser.add_argument('--lambda_r', type=float, default=0.0)
parser.add_argument('--meter_L', type=int, default=0)
# mu_meter: Section 4.2, eq. (6). Adds a known-meter spacing constraint to the selection
# step -- the only mechanism in the paper that can change WHICH candidates are selected.
# subset_select_dp_meter is O(N^2 M) as implemented (the paper derives O(N^2 M / L); see
# its docstring for why that saving is not realized here), so ~94ms per fragment against
# the plain DP's 0.34ms.
# [scale caution] t_hat is normalized to (0,1], so one candidate slot costs
# lambda/N = 1.25 in time, while mu=1000 gives a penalty of 1000*(1/160)^2 = 0.039 --
# 3% of a slot. Changing the selection at all needs mu ~ 1e5 or more (measured: 52/60
# assignments changed at 1e3, 58/60 at 1e5).
parser.add_argument('--mu_meter', type=float, default=0.0)
# --- Beat This spectrogram corpus (Zenodo 13922116) ------------------------------
# 5554 songs across 16 datasets. Against our 5 datasets / 2112 songs, the decisive
# difference is slow-tempo coverage: 375 songs below 64 BPM (we have 15) and 161 below
# 55 BPM (we have 0). SMC is 37% below 64 BPM, so this gap is the leading explanation for
# our SMC scores.
# The corpus is 50 fps (hop 441), so it MUST be used with --audio_downsampling_factor 441.
# (Our original pipeline runs at 43.07 fps; mixing two frame rates in one training run
#  destroys any consistent notion of tempo.) Their +-20% time-stretch and -5..+6 semitone
# copies ship with it.
parser.add_argument('--spect_root', type=str, required=True,
                    help='dir of {dataset}.npz from the Beat This corpus')
parser.add_argument('--spect_annot_root', type=str,
                    default='/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations')
parser.add_argument('--spect_datasets', type=str,
                    default='ballroom,hainsworth,rwc,harmonix,simac,smc,asap',
                    help='comma separated; gtzan is test-only and never enters train')
parser.add_argument('--spect_tempo_aug', type=str, default='',
                    help='e.g. -20,-16,-12,-8,8,12,16,20 (percent); empty disables')
parser.add_argument('--spect_pitch_aug', type=str, default='',
                    help='e.g. -5,-4,-3,-2,-1,1,2,3,4,5,6 (semitones); empty disables')
parser.add_argument('--spect_mask_aug', action='store_true',
                    help='beat_this-style online mask augmentation (permute, 1-6 masks/crop, 0.1-2s each)')
parser.add_argument('--eval_gtzan_ckpt', type=str, default=None,
                    help='if set: skip training entirely, load this checkpoint, evaluate on the GTZAN '
                         'test split (never seen in training - beat_this\'s own held-out test set), '
                         'print Beat/Downbeat/Joint F-measure, and exit.')
# --marginal: Section 8.4. Instead of picking a single sigma with a hard DP, train on
# -log Z (eq. 14), marginalized over every order-preserving injection. No stop-gradient
# needed. Evidence (measured, temp_wide): the sigma posterior is effectively a point mass
# on ballroom (~1.2 effective alternatives) but spreads over ~18 on carnatic, with not a
# single decisive track. So hard EM is training confidently on one arbitrary assignment
# exactly where the model is weakest.
parser.add_argument('--marginal', action='store_true', default=False)
# Equation (23) is -log Z alone. Marginalizing the FULL loss (8) adds gamma*sum_j g_j
# back, and that term is not optional: build_cost SUBTRACTS gamma*g_j, so with the term
# omitted nothing compensates, dL/dg_j <= 0 for every j, and the objective is unbounded
# below. Default to the exact marginalization; --literal_eq23 gives the equation as
# written, which is the variant the paper states but not the one that trains.
# (The failure is not subtle when it bites: background pressure lands on all N
# candidates while the -log Z' counter-pressure spreads across many assignments, and
# everything collapses to background -- 4 epochs of 0.000 with the loss still falling.)
parser.add_argument('--literal_eq23', action='store_true', default=False,
                    help='use equation (23) literally (-log Z alone); unbounded below')
# --joint_phase: sections 8.3 (eq. 19) and 8.5 (eq. 30). Infer (sigma, phi_0) together
# instead of matching first and phase second. The sequential decomposition has two
# failure modes, both because L_match is evaluated before any phase hypothesis exists:
# it cannot prefer the candidate that sits at a metrically sensible phase, and once
# sigma is fixed no later phase evidence can revise it. Costs a factor L: Algorithm 1
# is rerun once per hypothesis (or, under --marginal, the log-sum-exp recursion is).
# Needs a meter, from --meter_L or from the annotation's own downbeat spacing.
parser.add_argument('--joint_phase', action='store_true', default=False)
# --marginal_meters: section 8.6. Treat L itself as latent and marginalize over a
# finite candidate set, on the same footing as sigma and phi_0. Implies --marginal.
# Costs sum_L L recursions rather than L_max -- 77 for the default 2..12.
parser.add_argument('--marginal_meters', type=str, default='',
                    help='e.g. 2-12 or 3,4,6; empty keeps L fixed external metadata')
# --- per-candidate precision (section 4.1.2 / 4.1.3) ----------------------------
# Lets the model express reduced confidence in acoustically ambiguous passages rather
# than assuming one dataset-wide noise level. This reopens two failure modes the shared
# global b closes -- inflating b_j is a cheaper way to reduce the loss than localizing
# correctly, and shrinking it is unbounded below on already-easy candidates -- so it
# ships with all three mitigations of 4.1.3 plus the warm-up, and the DP's own cost
# keeps using the shared b regardless.
parser.add_argument('--predict_precision', action='store_true', default=False)
parser.add_argument('--precision_warmup', type=int, default=2000,
                    help='steps on a fixed global b before the precision head is used')
parser.add_argument('--precision_prior_alpha', type=float, default=2.0,
                    help='Gamma(alpha, beta) prior on the precision 1/b_j; beta tracks b')
# tau: the confidence threshold of Algorithm 5. It is swept on val after training, and
# is kept separate per class because beat and downbeat confidence distributions differ.
parser.add_argument('--tau_beat', type=float, default=0.2)
parser.add_argument('--tau_downbeat', type=float, default=0.2)
# stitch_beta_frames: the border beta of Section 9.3's stitching, in frames. The paper
# fixes no value and suggests starting near the candidate spacing D/N -- here
# D/N = 8 frames (0.186s).
parser.add_argument('--stitch_beta_frames', type=int, default=8)
# Section 9.2: run one pass of self-attention across the candidate features and feed it
# to the classification branch only (t_hat is still computed from z_j alone, so eq. 1's
# monotonicity guarantee is untouched).
# [evidence] Measured on a trained checkpoint: candidates matched to a downbeat have
# p(DB)=0.854 and background candidates 0.029, both healthy, but candidates matched to a
# genuine beat carry p(DB)=0.141. Beats outnumber downbeats ~3:1, so that tail is the
# source of every false downbeat and the cause of the 0.430 downbeat precision. Timing
# (median 14ms), thresholds (Joint +0.001 over a 9x9 sweep), DP assignment (6.7%) and
# class weighting (gradient ratio 0.81) were all ruled out by measurement. What remains
# is the Section 9.1 diagnosis: "which beat of the bar is this" is not expressible in a
# single candidate's local features. 0 disables it (previous behaviour).
parser.add_argument('--class_attention_layers', type=int, default=0)
parser.add_argument('--class_attention_heads', type=int, default=4)

# Options to separate log.log and checkpoints/ per fold (per run), so that running
# several runs in parallel or in sequence (8-fold CV) does not have them overwrite each
# other. Defaults reproduce the previous behaviour.
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--log_file', type=str, default='./log.log')

# AlignBeat Section 3: "unlike WaveBeat, we did not clip our gradients."
# None (the default) clips at 1.0. Any value <= 0 disables clipping entirely, which is
# what the paper does.
parser.add_argument('--grad_clip', type=float, default=None)
# --accumulate_grad_batches: beat_this states it trains with "gradient accumulation over
# 8 batches of size 8" (effective batch 64). The default stays 1 (previous behaviour:
# optimizer.step() every iteration); match their recipe explicitly with
# --batch_size 8 --accumulate_grad_batches 8. optimizer.step()/scheduler.step() are then
# called once every N iterations, and each micro-batch's loss is divided by N before
# backward() so the accumulated gradient is the mean (see the training loop below).
parser.add_argument('--accumulate_grad_batches', type=int, default=1)

# Paper Section 3: "we also made each dataset represent 1000 music excerpts per epoch...
# in order to prevent one dataset from dominating another in representation". 0 (the
# default) keeps the previous behaviour of concatenating all datasets whole, which is
# imbalanced by dataset size; a positive value draws exactly that many random excerpts
# from each dataset per epoch instead.
parser.add_argument('--samples_per_dataset', type=int, default=0)
# --amp: run the encoder in bfloat16 autocast. Measured 687 -> 366 ms per batch of 8 at
# T=1500 on an A6000, i.e. roughly 1.9x, and training here is compute-bound on the
# encoder's own 1500-frame self-attention (the DP is not the bottleneck: 40 vs 100
# events per fragment differ by under 5%). bfloat16 rather than float16 so no loss
# scaling is needed -- its exponent range matches float32, which matters because the
# loss contains log-sum-exp and -log p terms that float16 would underflow.
# The DP cost matrix and the log-sum-exp recursions stay in float32 regardless: they are
# built inside SubsetCriterion, which autocast does not reach through, and their
# numerics are load-bearing.
# --val_every: run the macro-average validation only every N epochs. Validation is the
# larger half of an epoch here (571 songs, each tiled, decoded and stitched), so N=2
# nearly halves wall-clock at the cost of resolution on the curve. The last epoch is
# always validated regardless, so the final number is never missed. N=1 is the old
# behaviour.
parser.add_argument('--val_every', type=int, default=1)
parser.add_argument('--amp', action='store_true', default=False,
                    help='bfloat16 autocast for the encoder (~1.9x faster)')

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# parse them args
args = parser.parse_args()

# Window length and candidate count.
#
# T (the encoder's frame count) is fixed first; N is then derived from it, never
# chosen directly (Section 3). The physical lower bound on the candidate count is
# N_min := BPM_max * D_min -- the most beats that can physically occur in a
# fragment of duration D -- and ProgressiveDownsample halves T repeatedly for as
# long as it can stay at or above N_min, taking whatever length remains as N.
num_frames_per_30s = 1500

max_bpm = 340

# JA: We need to define N = 30 seconds (=0.5 minutes) * 340 beats/minute max candidate number of beats
n_min = 0.5 * max_bpm
N = 2 * (math.floor(n_min / 2) + 1)   # 172

# N_min is what the model actually needs: the downsampling schedule is not free
# to shrink below it. Passing it through dict_args is what makes the derivation
# above reach ProgressiveDownsample instead of the placeholder default.
args.n_min = N

if args.lr is None:
    args.lr = 3e-4
if args.train_length is None:
    args.train_length = num_frames_per_30s * args.audio_downsampling_factor

WINDOW_FRAMES = args.train_length // args.audio_downsampling_factor
# NOTE: the raw-audio path runs at audio_sample_rate/audio_downsampling_factor
# = 43.07 fps, so 1500 frames is 34.8s there, not the 30s the N_min above
# assumes. The spectrogram path (--spect_root) is the 50 fps / 30s one the
# figure is derived for.
print(f"[alignbeat] window = {WINDOW_FRAMES} mel frames "
      f"({WINDOW_FRAMES / 50.0:.1f}s at 50 fps), "
      f"N_min = {N} (BPM_max = {max_bpm})")

# Decide up front whether to append ("a") or truncate ("w") the log, based on whether
# this is a resume. It used to be "w" unconditionally, so every restart after a crash
# wiped the entire log so far (epochs 0-56, for instance) and it had to be backed up by
# hand each time.
_is_resuming = len(glob.glob(os.path.join(args.checkpoint_dir, 'alignbeat_[0-9]*.pt'))) > 0
configure_log(args.log_file, mode="a" if _is_resuming else "w")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# set the seed
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
args.default_root_dir = os.path.join("lightning_logs", "full")
print(args.default_root_dir)

state_dicts = glob.glob(os.path.join(args.checkpoint_dir, 'alignbeat_[0-9]*.pt'))
start_epoch = 0
checkpoint_path = None

if len(state_dicts) > 0:
    # glob.glob's ordering depends on the filesystem/OS and is not sorted, so
    # state_dicts[-1] was never guaranteed to be the highest-numbered epoch. This bit us:
    # resuming checkpoints_fold0_expanded_datasets, which had over 20 checkpoints, loaded
    # epoch 58 instead of the latest epoch 96. Parse the epoch number explicitly and take
    # the numeric maximum.
    state_dicts.sort(key=lambda path: int(re.search("alignbeat_(.*).pt", path).group(1)))
    checkpoint_path = state_dicts[-1]
    start_epoch = int(re.search("alignbeat_(.*).pt", checkpoint_path).group(1)) + 1
    print("loaded:" + checkpoint_path)
else:
    print("no checkpoint found")

# setup the dataloaders
train_datasets = []
val_datasets = []
val_dataset_names = []  # dataset names in val_datasets order, for the macro-average eval below

# The corpus spectrograms are 50 fps (hop 441), so audio_downsampling_factor must be
# 441 for every frame<->second conversion downstream (target_sample_rate, train_length,
# stitching) to stay consistent. It is the only supported rate now that the raw-audio
# loader is gone; the check stays because the flag is still read in those conversions.
from alignbeat.bt_dataset import BeatThisSpectDataset

if args.audio_downsampling_factor != 441:
    raise SystemExit(
        f"the corpus is 50 fps: pass --audio_downsampling_factor 441 "
        f"(got {args.audio_downsampling_factor})")

_names = [d.strip() for d in args.spect_datasets.split(',') if d.strip()]
_tempo = tuple(int(v) for v in args.spect_tempo_aug.split(',') if v.strip())
_pitch = tuple(int(v) for v in args.spect_pitch_aug.split(',') if v.strip())
_train_frames = args.train_length // args.audio_downsampling_factor
_eval_frames = args.eval_length // args.audio_downsampling_factor
for _name in _names:
    train_datasets.append(BeatThisSpectDataset(
        args.spect_root, args.spect_annot_root, [_name], subset="train",
        validation_fold=args.validation_fold, target_length=_train_frames,
        tempo_aug=_tempo, pitch_aug=_pitch, mask_aug=args.spect_mask_aug))
    val_datasets.append(BeatThisSpectDataset(
        args.spect_root, args.spect_annot_root, [_name], subset="val",
        validation_fold=args.validation_fold, target_length=_eval_frames))
val_dataset_names = list(_names)
print(f"[spect] {len(_names)} datasets | train {sum(len(d) for d in train_datasets)} "
      f"val {sum(len(d) for d in val_datasets)} | {_train_frames} frames "
      f"({_train_frames/50.0:.1f}s at 50 fps) | tempo_aug={_tempo} pitch_aug={_pitch}")

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)

class PerDatasetBalancedSampler(torch.utils.data.Sampler):
    """Paper Section 3: "we also made each dataset represent 1000 music excerpts
    per epoch... in order to prevent one dataset from dominating another in
    representation". Concatenating and shuffling a ConcatDataset lets a large
    dataset (ballroom) appear far more often within an epoch than a small one
    (hainsworth), giving it more representation. This sampler instead draws
    exactly samples_per_dataset items at random from each dataset (with
    replacement if the dataset is smaller), shuffles them together, and yields
    num_datasets x samples_per_dataset items per epoch.
    """
    def __init__(self, datasets, samples_per_dataset):
        self.samples_per_dataset = samples_per_dataset
        # the [start, end) offset each sub-dataset occupies inside the ConcatDataset
        self.offsets = []
        start = 0
        for d in datasets:
            self.offsets.append((start, start + len(d)))
            start += len(d)

    def __iter__(self):
        indices = []
        for start, end in self.offsets:
            n = end - start
            replacement = n < self.samples_per_dataset
            local_indices = torch.randint(0, n, (self.samples_per_dataset,)) if replacement \
                else torch.randperm(n)[:self.samples_per_dataset]
            indices.append(local_indices + start)
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())

    def __len__(self):
        return self.samples_per_dataset * len(self.offsets)

if args.samples_per_dataset > 0:
    train_sampler = PerDatasetBalancedSampler(train_datasets, args.samples_per_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                                    sampler=train_sampler,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    collate_fn=collater)
else:
    train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                                    shuffle=args.shuffle,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    collate_fn=collater)
# Per-dataset val dataloaders, so each dataset is scored separately. The best epoch used
# to be chosen from a single pooled val_dataloader over the ConcatDataset, but with large
# size differences (harmonix 912 songs vs rwc_popular 13) the big datasets dominate the
# pooled mean, so the pooled Joint score kept rising while small datasets were actually
# getting worse -- observed directly: downbeat F on the original four datasets dropped
# sharply while the pooled score still climbed. The best epoch is now chosen on the
# macro-average (the mean of the per-dataset means).
per_dataset_val_dataloaders = [
    (name, torch.utils.data.DataLoader(ds, shuffle=False, batch_size=1,
                                        num_workers=args.num_workers,
                                        pin_memory=False, collate_fn=collater))
    for name, ds in zip(val_dataset_names, val_datasets)
]

# Which datasets have no downbeat annotation at all. Ask the DATASET rather than a
# hardcoded name list: BeatThisSpectDataset already derives this from each corpus
# directory's own info.json, and that is the same flag it uses to route those events
# down the beat-only (CLASS_UNKNOWN) path in the loss. Keeping a second, hand-maintained
# list in sync with it is exactly the kind of thing that silently drifts -- and did:
# the corpus marks both smc AND simac has_downbeats=false, but the hardcoded set held
# only smc, so simac's structurally-zero downbeat F was being averaged into the macro
# (measured: 0.000 on every epoch). That macro gates checkpoint saving and feeds
# ReduceLROnPlateau, so an unmeasurable constant was shaping the training schedule --
# the very failure the smc exclusion was added to prevent, reintroduced one dataset over.
beat_only_val_names = set()
for _name, _ds in zip(val_dataset_names, val_datasets):
    if _name in getattr(_ds, 'beat_only', BEAT_ONLY_DATASETS):
        beat_only_val_names.add(_name)
if beat_only_val_names:
    print(f"[alignbeat] beat-only datasets (excluded from the downbeat macro): "
          f"{sorted(beat_only_val_names)}")

def evaluate_macro_joint_f_measure(model, label):
    """Run per_dataset_val_dataloaders dataset by dataset and compute the
    macro-average Beat/Downbeat/Joint F-measure. Shared by the per-epoch
    evaluation and by the re-scoring of a checkpoint just after a resume."""
    per_dataset_beat_f, per_dataset_downbeat_f = [], []
    for name, loader in per_dataset_val_dataloaders:
        # The model only accepts fixed-length windows, so a piece is tiled into
        # fragments, decoded independently, and reassembled by Section 9.3's
        # stitching. window_frames must be the same T the model was built for:
        # a different T runs the downsampling schedule to a different N, and the
        # head then rejects the input outright.
        ds_beat_f, ds_downbeat_f, _ = evaluate_beat_f_measure_subset(
            loader, model, args.audio_downsampling_factor, args.audio_sample_rate,
            window_frames=WINDOW_FRAMES,
            border_frames=args.stitch_beta_frames,
            threshold_beat=args.tau_beat, threshold_downbeat=args.tau_downbeat,
            use_amp=args.amp)
        per_dataset_beat_f.append(ds_beat_f)
        # [beat-only datasets are excluded from the downbeat macro] smc and simac have
        # no downbeat ground truth, so their downbeat F is structurally 0. Including it
        # lets an unmeasurable value drag the macro down by a constant 1/N, and that
        # macro is exactly what gates checkpoint saving and feeds ReduceLROnPlateau -- so
        # the training schedule would be driven by a constant penalty unrelated to model
        # quality (measured at epoch 2: downbeat macro 0.417 -> 0.348). Beat F is
        # meaningful on both, so it stays in.
        if name in beat_only_val_names:
            print(f"{label} | [{name}] Beat: {ds_beat_f:0.3f} | Downbeat: n/a (beat-only)")
        else:
            per_dataset_downbeat_f.append(ds_downbeat_f)
            print(f"{label} | [{name}] Beat: {ds_beat_f:0.3f} | Downbeat: {ds_downbeat_f:0.3f}")

    beat_mean_f_measure = float(np.mean(per_dataset_beat_f))
    downbeat_mean_f_measure = float(np.mean(per_dataset_downbeat_f)) if per_dataset_downbeat_f else 0.0
    joint_f_measure = (beat_mean_f_measure + downbeat_mean_f_measure) / 2
    print(f"{label} | Beat score: {beat_mean_f_measure:0.3f} | Downbeat score: {downbeat_mean_f_measure:0.3f} | Joint score: {joint_f_measure:0.3f}")
    return beat_mean_f_measure, downbeat_mean_f_measure, joint_f_measure

dict_args = vars(args)

if __name__ == '__main__':
    dict_args['meter_length'] = args.meter_L
    dict_args['marginal_background'] = not args.literal_eq23
    if args.marginal_meters:
        text = args.marginal_meters
        if '-' in text:
            low, high = (int(v) for v in text.split('-'))
            meters = tuple(range(low, high + 1))
        else:
            meters = tuple(int(v) for v in text.split(','))
        dict_args['marginal_meters'] = meters
        dict_args['marginal'] = True
        print(f"[alignbeat] marginalizing the meter over {meters} "
              f"({sum(meters)} log-sum-exp recursions per fragment)")
    # BeatThisEncoder wants dropout as a {"frontend":.., "transformer":..} dict,
    # and encoder_input_frames is derived rather than passed on the command line.
    dict_args['dropout'] = {
        "frontend": args.beat_this_frontend_dropout,
        "transformer": args.beat_this_transformer_dropout,
    }
    dict_args['encoder_input_frames'] = WINDOW_FRAMES

    model = model_module.create_alignbeat_model(args=args, **dict_args)
    print(f"[alignbeat] N = {model.num_candidates} candidates, spacing "
          f"{args.train_length / args.audio_sample_rate / model.num_candidates * 1000:.0f}ms")

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    device = next(model.module.parameters()).device

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, device))

    if args.eval_gtzan_ckpt:
        # Standalone eval mode: score one checkpoint on GTZAN (beat_this's own held-out
        # test set, never in --spect_datasets for any of our runs) using OUR OWN
        # postprocessing (evaluate_beat_f_measure_subset) - the subset head has no
        # framewise output for beat_this's minimal/DBN postprocessing to apply to, so
        # this is the correct comparison, not a compromise. Only the test-set identity
        # (GTZAN) is being matched to beat_this's protocol here, not the postprocessing.
        model.load_state_dict(torch.load(args.eval_gtzan_ckpt, device))
        model.eval()
        gtzan_test_dataset = BeatThisSpectDataset(
            args.spect_root, args.spect_annot_root, ["gtzan"], subset="test",
            validation_fold=args.validation_fold, target_length=_eval_frames)
        gtzan_loader = torch.utils.data.DataLoader(
            gtzan_test_dataset, shuffle=False, batch_size=1,
            num_workers=args.num_workers, pin_memory=False, collate_fn=collater)
        gtzan_beat_f, gtzan_downbeat_f, _ = evaluate_beat_f_measure_subset(
            gtzan_loader, model, args.audio_downsampling_factor, args.audio_sample_rate,
            window_frames=WINDOW_FRAMES,
            border_frames=args.stitch_beta_frames,
            threshold_beat=args.tau_beat, threshold_downbeat=args.tau_downbeat,
            use_amp=args.amp, label="[GTZAN test] ")
        gtzan_joint = (gtzan_beat_f + gtzan_downbeat_f) / 2
        print(f"=== GTZAN TEST (checkpoint={args.eval_gtzan_ckpt}) ===")
        print(f"Beat: {gtzan_beat_f:.3f} | Downbeat: {gtzan_downbeat_f:.3f} | Joint: {gtzan_joint:.3f}")
        sys.exit(0)

    model.training = True
    print(f'[MEM] after model init: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')

    accum = max(1, args.accumulate_grad_batches)

    if args.optimizer == 'radam_lookahead':
        from alignbeat.optim import Lookahead
        base_optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizer = Lookahead(base_optimizer, k=args.lookahead_k, alpha=args.lookahead_alpha)
        print(f"[optim] RAdam + Lookahead(k={args.lookahead_k}, alpha={args.lookahead_alpha}) lr={args.lr}")
    elif args.optimizer == 'adamw':
        # As in the beat_this repository: apply weight_decay only to parameters of rank
        # >= 2 (matrices), and 0 to bias/norm parameters (rank <= 1).
        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
        param_groups = [
            {"params": decay_params, "weight_decay": args.adamw_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        print(f"[optim] AdamW lr={args.lr} weight_decay={args.adamw_weight_decay} "
              f"(decay params={len(decay_params)}, no-decay params={len(no_decay_params)})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # Default weight decay is 0

    if args.lr_schedule == 'cosine_warmup':
        # Reproduces beat_this's CosineWarmupScheduler (interval="step"). step() is called
        # at the end of each accumulation window, i.e. once per real optimizer.step() (see
        # the call site right after optimizer.step() below) -- so total_steps must be
        # counted in real optimizer steps (raw iterations divided by accum, the same
        # quantity as Lightning's estimated_stepping_batches), not raw iterations, for
        # warmup_steps=1000 to end at the 1000th optimizer step as in their recipe.
        steps_per_epoch = math.ceil(len(train_dataloader) / accum)
        total_steps = steps_per_epoch * args.epochs
        scheduler = CosineWarmupScheduler(optimizer, warmup=args.warmup_steps, max_iters=total_steps)
        print(f"[optim] scheduler: cosine_warmup warmup_steps={args.warmup_steps} "
              f"total_steps={total_steps} ({steps_per_epoch} optimizer steps/epoch "
              f"[{len(train_dataloader)} iters / accum={accum}] x {args.epochs} epochs)")
    else:
        lr_patience = args.patience if args.lr_patience is None else args.lr_patience
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=lr_patience, factor=args.lr_factor,
            min_lr=args.min_lr, verbose=True)
        print(f"[optim] scheduler: patience={lr_patience} factor={args.lr_factor} min_lr={args.min_lr}")

    # If the optim_{epoch}.pt matching checkpoint_path exists, restore the optimizer and
    # scheduler state too. If not (older checkpoints), just start with a fresh optimizer
    # -- the previous behaviour, kept for backwards compatibility.
    if checkpoint_path:
        optim_path = checkpoint_path.replace('alignbeat_', 'optim_')
        if os.path.exists(optim_path):
            optim_state = torch.load(optim_path, map_location=device)
            optimizer.load_state_dict(optim_state['optimizer'])
            scheduler.load_state_dict(optim_state['scheduler'])
            print(f"restored optimizer/scheduler state as well: {optim_path}")
        else:
            print(f"no optimizer/scheduler state file ({optim_path}) - starting fresh")

    loss_hist = collections.deque(maxlen=500)

    model.train()

    print('Num training images: {}'.format(len(train_dataset_list)))

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    highest_joint_f_measure = 0
    if checkpoint_path:
        # If highest_joint_f_measure reset to 0 on every restart, scores far below the
        # loaded checkpoint's real score (e.g. Joint 0.864 at epoch 56) would be mistaken
        # for new records and saved. Re-evaluate the loaded checkpoint once to establish
        # the true baseline.
        # Lookahead's model is the slow sequence; training runs on fast weights and an
        # epoch boundary almost never coincides with a sync (313 iters, k=5), so evaluate
        # and checkpoint the slow weights and restore the fast ones afterwards.
        model.eval()
        _, _, highest_joint_f_measure = evaluate_macro_joint_f_measure(model, label="Resume check")
        print(f"resume baseline (highest_joint_f_measure) = {highest_joint_f_measure:0.3f}")

    for epoch_num in range(start_epoch, args.epochs):
        model.train()

        epoch_loss = []
        cls_losses = []
        time_losses = []
        bg_losses = []
        cont_losses = []
        per_losses = []

        print(f'[MEM] epoch {epoch_num} start: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')

        for iter_num, data in enumerate(train_dataloader): #target[:,:,0:2]=interval, target[:,:,2]=class
            audio, target = data  #MJ: audio:shape =(16,1,3000,81); target:shape=(16,128,3)
            if torch.cuda.is_available():
                audio = audio.cuda()
                target = target.cuda()

            # --accumulate_grad_batches > 1: call optimizer.step()/scheduler.step() exactly
            # once every N iterations (micro-batches); beat_this's recipe is batch_size=8,
            # accumulate_grad_batches=8 -> effective batch 64. is_accum_end is true on the
            # last of those N iterations, and also on the epoch's last iteration, so that
            # a partially filled window at the end of an epoch still steps rather than
            # discarding the gradient accumulated so far.
            is_accum_start = (iter_num % accum == 0)
            is_accum_end = ((iter_num + 1) % accum == 0) or (iter_num == len(train_dataloader) - 1)

            try:
                if is_accum_start:
                    optimizer.zero_grad()

                with torch.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=args.amp and torch.cuda.is_available()):
                    class_loss, time_loss, background_loss, \
                        continuity_loss, periodicity_loss = \
                        model((audio, target))

                class_loss = class_loss.mean()
                time_loss = time_loss.mean()
                background_loss = background_loss.mean()
                continuity_loss = continuity_loss.mean()
                periodicity_loss = periodicity_loss.mean()

                cls_losses.append(class_loss.item())
                time_losses.append(time_loss.item())
                bg_losses.append(background_loss.item())
                cont_losses.append(continuity_loss.item())
                per_losses.append(periodicity_loss.item())

                loss = class_loss + time_loss + background_loss + continuity_loss + periodicity_loss

                if bool(loss == 0):
                    continue

                # Belt and braces for every head type. A single non-finite loss
                # backpropagates NaN into every weight and the run is dead, but the
                # epoch loop keeps spinning and validation keeps reporting a frozen
                # model -- so it costs hours before anyone notices. Observed live on
                # subset_nosmc (epoch 0 iteration 194: "BG: nan", then seven epochs of
                # 0.00000 losses and 0.000 scores). The head-level guard in
                # SubsetCriterion catches the known source; this catches the rest.
                if not torch.isfinite(loss):
                    print(f"[train] WARNING: non-finite loss at epoch {epoch_num} "
                          f"iteration {iter_num}; skipping this micro-batch", flush=True)
                    # Deliberately no zero_grad(): with accum > 1 that would erase the
                    # gradient earlier micro-batches already accumulated. Skip only this
                    # micro-batch (no backward at all) and keep accumulating.
                    continue

                # With accum > 1, divide each micro-batch's loss by accum before backward:
                # summing N micro-batch gradients gives a sum rather than a mean, which is
                # a different scale from training on the effective batch in one go.
                # Dividing makes the total accumulated gradient exactly the gradient of the
                # mean loss.
                (loss / accum).backward()

                if not is_accum_end:
                    # Still inside this accumulation window -> on to the next micro-batch
                    # without stepping. loss_hist/epoch_loss are recorded once per
                    # iteration by the shared code after this if/else (appending here too
                    # would double-count).
                    pass
                else:
                    # 1.0 unless --grad_clip is given; a value <= 0 disables clipping
                    # entirely, as the paper does (see the argparse comment above).
                    clip_norm = 1.0 if args.grad_clip is None else args.grad_clip

                    # Check the GRADIENTS, not just the loss. A finite loss is not enough:
                    # log_softmax is computed over the whole batch at once, so a fragment
                    # whose logits are NaN stays in the graph even when its loss terms are
                    # dropped, and grad_weight = grad_out^T @ activation turns 0 x NaN into
                    # NaN. clip_grad_norm_ then spreads that single NaN across EVERY
                    # parameter (total_norm becomes NaN, so every grad is scaled by NaN).
                    # clip_grad_norm_ conveniently returns the pre-clip total norm, so this
                    # costs one extra scalar read on the path that already computes it.
                    if clip_norm > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                        grads_finite = bool(torch.isfinite(total_norm))
                    else:
                        grads_finite = all(
                            bool(torch.isfinite(p.grad).all())
                            for p in model.parameters() if p.grad is not None)

                    if not grads_finite:
                        print(f"[train] WARNING: non-finite gradients at epoch {epoch_num} "
                              f"iteration {iter_num}; skipping optimizer step", flush=True)
                        optimizer.zero_grad()
                        loss_hist.append(float(loss))
                        epoch_loss.append(float(loss))
                        continue

                    optimizer.step()
                    if args.lr_schedule == 'cosine_warmup':
                        # In plateau mode scheduler.step(joint_f_measure) is called once per
                        # epoch on the validation metric (at the bottom), but cosine_warmup
                        # counts steps at the end of each accumulation window (i.e. per real
                        # optimizer step), so it is stepped here.
                        scheduler.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                # Loss (8): CLS = class term, TIME = L1 term, BG = background
                # term. CONT/PER are the optional continuity and periodicity
                # (Section 10.4) regularizers, 0 unless their weights are set.
                print(
                    'Epoch: {} | Iteration: {} | CLS: {:1.5f} | TIME: {:1.5f} | BG: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num,
                        float(class_loss), float(time_loss),
                        float(background_loss), np.mean(loss_hist))
                )

                if iter_num % 10 == 0:
                    print(f'[MEM] iter {iter_num}: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB, audio={audio.shape}')

                del class_loss
                del time_loss
                del background_loss
                del continuity_loss
                del periodicity_loss
                del loss
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(e)
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

        # End of: for iter_num, data in enumerate(train_dataloader)

        # Evaluate the evaluation dataset in each epoch
        # [guard against a silent stall] If every training iteration is skipped by an
        # exception the optimizer never runs, yet the epoch loop and validation keep going
        # and print the same frozen model's scores all the way to epoch 100 (observed: a
        # subset run stalled silently at epoch 74 and then failed 9,749 iterations in a
        # row, while the log looked like a normal finish). Abort immediately when a whole
        # epoch fails -- better than burning hours.
        if len(epoch_loss) == 0:
            raise RuntimeError(
                f"epoch {epoch_num}: every training iteration was skipped (0 optimizer "
                f"steps). Training has stopped; aborting instead of validating a frozen "
                f"model for the remaining epochs.")
        if len(epoch_loss) < 0.5 * len(train_dataloader):
            print(f"[warn] epoch {epoch_num}: only {len(epoch_loss)}/{len(train_dataloader)} "
                  f"iterations succeeded - training is degrading", flush=True)

        print(f'[MEM] before eval: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')
        print('Evaluating dataset')
        # NOTE [Lookahead]: this evaluates the FAST weights. Lookahead's reported model is
        # the slow sequence, and with 313 iterations per epoch and k=5 an epoch boundary
        # essentially never lands on a sync, so the reported score is not the Lookahead
        # average. Swapping the slow weights in for eval+save was tried and REVERTED: it
        # made every resumed run score 0.000 at the next epoch while the training loss
        # stayed healthy, and the cause was not identified (it is not the model/optimizer
        # weight mismatch - saving fast weights for resume did not fix it). Reverting is
        # the conservative choice: fast-weight evaluation is at least self-consistent and
        # matches every number produced so far. optim.py still carries sync_to_slow() /
        # restore_fast() for offline use on a checkpoint.
        is_val_epoch = ((epoch_num + 1) % args.val_every == 0
                        or epoch_num == args.epochs - 1)
        if is_val_epoch:
            beat_mean_f_measure, downbeat_mean_f_measure, joint_f_measure = (
                evaluate_macro_joint_f_measure(model, label=f"Epoch = {epoch_num}"))

        print(f"Epoch = {epoch_num} | CLS: {np.mean(cls_losses):0.3f} "
              f"| TIME: {np.mean(time_losses):0.3f} | BG: {np.mean(bg_losses):0.3f} "
              f"| CONT: {np.mean(cont_losses):0.3f} | PER: {np.mean(per_losses):0.3f}")
        if not is_val_epoch:
            # No fresh score this epoch. The scheduler step and the best-checkpoint test
            # both consume joint_f_measure, and feeding either a stale value would be
            # wrong -- ReduceLROnPlateau would read a flat metric and decay early, and the
            # best-checkpoint test would compare a score against itself. Skip both, but
            # still write the rolling "last" checkpoint so a resume never loses more than
            # one epoch.
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, 'alignbeat_last.pt'))
            torch.save({'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_num, 'joint_f_measure': highest_joint_f_measure},
                       os.path.join(args.checkpoint_dir, 'optim_last.pt'))
            continue

        if args.lr_schedule != 'cosine_warmup':
            # cosine_warmup already advanced per step inside the iteration loop above;
            # only ReduceLROnPlateau is stepped once per epoch on joint_f_measure.
            scheduler.step(joint_f_measure)

        should_save_checkpoint = False
        if joint_f_measure > highest_joint_f_measure:
            should_save_checkpoint = True
            print(f"Joint score of {joint_f_measure:0.3f} exceeded previous best at {highest_joint_f_measure:0.3f}")
            highest_joint_f_measure = joint_f_measure

        #should_save_checkpoint = True # FOR DEBUGGING
        if should_save_checkpoint:
            new_checkpoint_path = os.path.join(args.checkpoint_dir, 'alignbeat_{}.pt'.format(epoch_num))
            print(f"Saving checkpoint at {new_checkpoint_path}")
            torch.save(model.state_dict(), new_checkpoint_path)
            # The weights file format is left alone because other scripts
            # (evaluate_all_datasets.py and friends) expect a raw state_dict; the
            # optimizer/scheduler state goes into a separate file alongside it, so a
            # resume can pick it up and not restart LR/momentum from scratch. This was
            # real: when an nhead=8 run crashed on a NaN during eval, the absence of this
            # file left the restarted run stuck below its previous best for over 20 epochs.
            new_optim_path = os.path.join(args.checkpoint_dir, 'optim_{}.pt'.format(epoch_num))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, new_optim_path)

        # beat_this does not select a best epoch at all: it simply overwrites with the last
        # epoch's weights (their Section 9 -- deliberately training on past the point where
        # validation loss rises, to get overconfident predictions). Whether that is better
        # for our subset head is untested, so the best-checkpoint logic above stays, and in
        # addition the most recent epoch's weights are written to a fixed filename
        # (overwritten each epoch) so best and last can be compared directly no matter when
        # training is stopped.
        last_checkpoint_path = os.path.join(args.checkpoint_dir, 'alignbeat_last.pt')
        torch.save(model.state_dict(), last_checkpoint_path)
        last_optim_path = os.path.join(args.checkpoint_dir, 'optim_last.pt')
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'epoch': epoch_num, 'joint_f_measure': joint_f_measure}, last_optim_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()

    torch.save(model, os.path.join(args.checkpoint_dir, 'model_final.pt'))
