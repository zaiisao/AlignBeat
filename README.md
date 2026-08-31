# AlignBeat

Beat and downbeat tracking as **latent order-preserving alignment**.

A fixed number of candidate predictors independently regress candidate event
times and classify them, deliberately overgenerating more candidates than there
are true events. Training must then decide which candidates align to a genuine
event and which align to nothing — the same role a blank symbol plays in CTC's
own alignment. Because the ground truth is sorted by construction and the
candidate times are made strictly increasing by an architectural
reparameterization, this reduces to choosing *which subset of candidates is
active*, solved exactly by an O(NM) dynamic program.

Equation and algorithm numbers throughout the code refer to the paper draft,
which is kept in `private/` (local-only, not tracked).

## Layout

This repository is a fork of [Beat This!](https://github.com/CPJKU/beat_this)
(Foscarin et al., 2024). Upstream's tree sits at the root unmodified except for
three files; ours is the `alignbeat/` package it imports.

| path | what |
| --- | --- |
| `alignbeat/subset_head.py` | candidate heads, eq. (1), the DP (Alg. 1), and the losses |
| `alignbeat/progressive_downsample.py` | T → N patch-merging downsample; N derived from N_min |
| `alignbeat/stitching.py` | reference implementation of §9.3 fragment reassembly |
| `beat_this/model/beat_tracker.py` | upstream, plus a `SubsetHead` branch |
| `beat_this/model/pl_module.py` | upstream, plus the subset loss, targets and decode |
| `launch_scripts/train.py` | upstream, plus `--head_type` and the flags below |
| `diagnose_bottleneck.py` | attribute a flat F-measure to its component failure |

Everything else — dataset, augmentation, losses, postprocessor, metrics,
`compute_paper_metrics.py` — is upstream code, untouched. That is deliberate:
`--head_type dense` reproduces Beat This! exactly, so the A/B against
`--head_type subset` differs in the head and its loss and in nothing else.

To pull upstream changes: `git merge beat_this/main`.

## What is implemented

Everything is off by default: the defaults are the hard-EM pipeline of Sections 5-7.

| construction | flag |
| --- | --- |
| §4.2 known-meter spacing in the selection, eq. (6) | `--mu_meter` |
| §4.1.2/4.1.3 per-candidate precision, with all three mitigations | `--predict_precision` |
| §8 EM over the shared phase latent, eqs. (12)-(15) | `--meter_L` (beat-only data) |
| §8.1 mixed meter within a fragment, eq. (17) | per-fragment `segments` metadata |
| §8.3 joint MAP of (σ, φ₀), eq. (19) | `--joint_phase` |
| §8.4 marginal likelihood over σ, eq. (23) | `--marginal` |
| §8.5 joint marginalization of (σ, φ₀), eq. (30) | `--marginal --joint_phase` |
| §8.6 marginalizing the meter too, eq. (32) | `--marginal_meters 2-12` |
| §10.2 candidate self-attention for the class branch, eq. (35) | `--class_attention_layers` |
| §10.3 class-specific reweighting | `--omega_db` |
| §10.4 periodicity regularizer, eqs. (36)-(37) | `--lambda_r` |

`tests/test_phase.py` checks eqs. (19), (20), (25)-(29) and (32)-(34) against
brute-force enumeration, which is the paper's own stated verification. Two things are
deliberately not done: the O(N²M/L) saving §4.2 derives for the augmented recursion
(the implementation is exact but O(N²M) — see the docstring), and Remark 3's global
meter/phase head, which the paper does not formalize.

## Running

```sh
export PYTHONPATH=$PWD

# the alignment head, and the dense baseline it is measured against
python launch_scripts/train.py --name myrun --head_type subset --fold 0
python launch_scripts/train.py --name mydense --head_type dense --fold 0

# scoring, and attributing a flat result to its component failure
python launch_scripts/compute_paper_metrics.py --models "checkpoints/myrun*.ckpt"
python diagnose_bottleneck.py --checkpoints "checkpoints/myrun*.ckpt"

# tests are plain scripts, not a pytest suite
for t in tests/test_*.py; do python "$t" || break; done
```

Data lives where upstream expects it, under `data/`. Folds are Beat This!'s own
`8-folds.split`; `--fold N` names the fold held out.

`N` is not a free hyperparameter: the window is fixed at T = 1500 frames (30s at
50fps) and the candidate count derives from the physical tempo bound
`N_min = BPM_max * D`, which the downsampling schedule is not free to shrink
below.

See `docs/ABLATIONS.md` for the measured arms.

## History

This began as BeatFCOS, a 1-D adaptation of FCOS with anchors, interval
regression and Soft-NMS ([arXiv:2510.14391](https://arxiv.org/abs/2510.14391)),
then moved from a WaveBeat backbone to Beat Transformer to Beat This!, and
finally dropped anchors entirely for the subset-selection formulation above.
None of the anchor machinery survives here; the original is preserved in its
own repository.

## License

MIT, matching upstream Beat This!. See `LICENSE`.
