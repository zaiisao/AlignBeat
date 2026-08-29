#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on each dataset's val subset separately.
train.py pools four datasets (ballroom/beatles/hainsworth/rwc_popular) into a
single val_dataloader, so per-dataset performance was invisible - this script
builds one dataset/DataLoader per corpus dataset and scores them individually.

Neither gtzan nor smc has a CLI argument in train.py, so neither was used for
training - they are completely unseen for this checkpoint.
smc has no downbeat distinction, but dataloader.py's parser hardcodes every event
as beat=1 (line 461) and that value is reused verbatim by the downbeat decision
logic (lines 483-485), so SMC's ground-truth downbeats end up identical to its
ground-truth beats. The evaluation is therefore not "zero because there are no
downbeats" but "every beat is also labelled as a downbeat" - a nonzero Downbeat
F-measure here is not surprising.

Fold consistency: ballroom/beatles/hainsworth/rwc_popular were trained with 8-fold
CV, so --validation_fold must name the actual held-out fold. Leaving subset="val"
with validation_fold=None reads the val portion of the old 80/10/10 split, which
is unrelated to the 8-fold partition, and training songs leak into the evaluation
set. gtzan/smc have no fold file at all and were never trained on, so they use
subset="full-val" over the whole dataset.
"""
import argparse
import os
import sys
# Always import the alignbeat package from the repo this script lives in.
# The absolute path '/disk1/taegum/mnt/AlignBeat' that used to be hardcoded here
# pointed at a stale tree, so old code missing the newer modules shadowed the local
# ones. Deriving the path from the script location makes any checkout use its own
# package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from alignbeat import model_module
from alignbeat.bt_dataset import BeatThisSpectDataset
from alignbeat.dataloader import collater
from alignbeat.beat_eval import evaluate_beat_f_measure_subset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--validation_fold', type=int, default=0,
                     help="8-fold CV held-out fold index for ballroom/beatles/hainsworth/rwc_popular (must match the validation_fold the checkpoint was trained with)")
# Must match the value the checkpoint was trained with (N is derived from it).
parser.add_argument('--spect_root', required=True,
                    help='dir of {dataset}.npz from the Beat This corpus')
parser.add_argument('--spect_annot_root',
                    default='/disk4/shared/beat_this/data/annotations')
parser.add_argument('--datasets', type=str,
                    default='asap,ballroom,beatles,candombe,filosax,groove_midi,guitarset,'
                            'hainsworth,harmonix,hjdb,jaah,rwc,simac,smc,tapcorrect,gtzan')
parser.add_argument('--eval_length', type=int, default=2097152)
parser.add_argument('--window_frames', type=int, default=1500)
parser.add_argument('--n_min', type=int, default=172)
parser.add_argument('--transformer_dim', type=int, default=512)
parser.add_argument('--class_attention_layers', type=int, default=0)
# Algorithm 3's threshold. Split in two so beat and downbeat, whose score
# distributions differ, can be swept independently.
parser.add_argument('--tau_beat', type=float, default=0.2)
parser.add_argument('--tau_downbeat', type=float, default=0.2)
# Border beta for the Section 9.3 stitching, in frames. The default is the candidate
# spacing D/N = 8 frames.
parser.add_argument('--stitch_beta_frames', type=int, default=8)
args = parser.parse_args()

# Every dataset in the corpus. gtzan has no fold split at all -- it ships whole, as
# the held-out test set -- so it is evaluated as "test" while everything else is scored
# on its own validation fold.
DATASETS = [d.strip() for d in args.datasets.split(',') if d.strip()]

AUDIO_SAMPLE_RATE = 22050
AUDIO_DOWNSAMPLING_FACTOR = 441   # the corpus is 50 fps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# WINDOW_FRAMES must match what the checkpoint was trained with: the
# downsampling schedule derives N from it, and the head rejects any other N.
WINDOW_FRAMES = args.window_frames
_eval_frames = args.eval_length // AUDIO_DOWNSAMPLING_FACTOR
model = model_module.create_alignbeat_model(
    args=None,
    audio_downsampling_factor=AUDIO_DOWNSAMPLING_FACTOR,
    audio_sample_rate=AUDIO_SAMPLE_RATE,
    encoder_input_frames=WINDOW_FRAMES,
    n_min=args.n_min,
    transformer_dim=args.transformer_dim,
    dropout={"frontend": 0.1, "transformer": 0.2},
    class_attention_layers=args.class_attention_layers,
)

state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if unexpected:
    raise RuntimeError(f"checkpoint has keys the model does not: {unexpected}")
# Any missing key means evaluating with a randomly initialised submodule, which
# yields plausible-looking garbage numbers, so always treat it as an error.
if missing:
    raise RuntimeError(
        f"missing keys in checkpoint: {missing} - it was trained with a different "
        f"configuration, or the file is corrupt")
model = model.to(device)

print(f"checkpoint: {args.checkpoint}")
print(f"tau_beat={args.tau_beat} | tau_downbeat={args.tau_downbeat} | "
      f"window_frames={WINDOW_FRAMES} | n_min={args.n_min}\n")

results = {}
beat_only_names = set()
for name in DATASETS:
    subset = "test" if name == "gtzan" else "val"
    val_dataset = BeatThisSpectDataset(
        args.spect_root, args.spect_annot_root, [name], subset=subset,
        validation_fold=args.validation_fold, target_length=_eval_frames)
    if name in val_dataset.beat_only:
        beat_only_names.add(name)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collater)

    beat_f, downbeat_f, song_results = evaluate_beat_f_measure_subset(
        val_dataloader, model, AUDIO_DOWNSAMPLING_FACTOR, AUDIO_SAMPLE_RATE,
        window_frames=WINDOW_FRAMES,
        border_frames=args.stitch_beta_frames,
        threshold_beat=args.tau_beat, threshold_downbeat=args.tau_downbeat,
        full_metrics=True, verbose=True,
    )
    # Pull CMLt (Correct Metric Level Total) and AMLt (Any Metric Level Total) out of
    # the per-song dicts mir_eval.beat.evaluate returns and average over songs - the
    # evaluation helper itself only aggregates and returns F-measure.
    beat_cmlt = np.mean([r['beat_scores']['Correct Metric Level Total'] for r in song_results])
    beat_amlt = np.mean([r['beat_scores']['Any Metric Level Total'] for r in song_results])
    downbeat_cmlt = np.mean([r['downbeat_scores']['Correct Metric Level Total'] for r in song_results])
    downbeat_amlt = np.mean([r['downbeat_scores']['Any Metric Level Total'] for r in song_results])

    results[name] = (beat_f, downbeat_f, beat_cmlt, beat_amlt, downbeat_cmlt, downbeat_amlt)
    print(f"[{name}] Beat F: {beat_f:.3f} CMLt: {beat_cmlt:.3f} AMLt: {beat_amlt:.3f} | "
          f"Downbeat F: {downbeat_f:.3f} CMLt: {downbeat_cmlt:.3f} AMLt: {downbeat_amlt:.3f} | "
          f"Joint F: {(beat_f+downbeat_f)/2:.3f}")

print("\n=== Summary ===")
for name, (beat_f, downbeat_f, beat_cmlt, beat_amlt, downbeat_cmlt, downbeat_amlt) in results.items():
    downbeat = ("      n/a (beat-only)" if name in beat_only_names else
                f"F:{downbeat_f:.3f} CMLt:{downbeat_cmlt:.3f} AMLt:{downbeat_amlt:.3f}")
    print(f"{name:<12} Beat  F:{beat_f:.3f} CMLt:{beat_cmlt:.3f} AMLt:{beat_amlt:.3f}  |  "
          f"Downbeat  {downbeat}  |  Joint F:{(beat_f+downbeat_f)/2:.3f}")

# Macro averages, on the same rules train.py uses: gtzan is the held-out test set and is
# reported separately rather than folded into the number that describes validation, and
# datasets with no downbeat annotation are excluded from the downbeat mean instead of
# contributing a structural zero.
val = {k: v for k, v in results.items() if k != "gtzan"}
if val:
    beat_macro = np.mean([v[0] for v in val.values()])
    down_macro = np.mean([v[1] for k, v in val.items() if k not in beat_only_names])
    print(f"\nmacro over {len(val)} val datasets | Beat: {beat_macro:.3f} | "
          f"Downbeat: {down_macro:.3f} ({len(val)-len(beat_only_names & val.keys())} datasets) | "
          f"Joint: {(beat_macro+down_macro)/2:.3f}")
if "gtzan" in results:
    g = results["gtzan"]
    print(f"GTZAN (held-out test)          | Beat: {g[0]:.3f} | Downbeat: {g[1]:.3f} | "
          f"Joint: {(g[0]+g[1])/2:.3f}")
