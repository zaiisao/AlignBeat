import argparse
import os
from pathlib import Path

import torch

# Our own checkpoints carry numpy scalars and dtypes in hyper_parameters, and torch
# >= 2.6 loads with weights_only=True by default, which refuses them. Allowlisting the
# individual globals is a moving target (scalar, then dtype[float64], ...), so restore
# the pre-2.6 default for the resume path. Safe here: these are checkpoints this repo
# wrote. Do NOT copy this into anything that loads third-party checkpoints.
_torch_load = torch.load
def _load_trusted(*a, **k):
    # Lightning passes weights_only=True EXPLICITLY, so setdefault is not enough.
    k["weights_only"] = False
    return _torch_load(*a, **k)
torch.load = _load_trusted
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset import BeatDataModule
from alignbeat.downsample import (BPM_MAX, choose_num_candidates, halved_candidates,
                                  n_candidates_from_tempo)
from beat_this.model.pl_module import PLBeatThis


def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)

    print("Starting a new run with the following parameters:")
    print(args)

    params_str = f"{'noval ' if not args.val else ''}{'hung ' if args.hung_data else ''}{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}{args.loss}-h{args.transformer_dim}-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}{' nosumH ' if not args.sum_head else ''}{' nopartialT ' if not args.partial_transformers else ''}"
    if args.logger == "wandb":
        if args.resume_checkpoint and args.resume_id:
            wandb_args = dict(id=args.resume_id, resume="must")
        else:
            wandb_args = {}
        logger = WandbLogger(
            project="beat_this", name=f"{args.name} {params_str}".strip(), **wandb_args
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    checkpoint_dir = (
        Path(__file__).parent.parent.relative_to(Path.cwd()) / "checkpoints"
    )
    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {
            "kind": "permute",
            "min_count": 1,
            "max_count": 6,
            "min_len": 0.1,
            "max_len": 2,
            "min_parts": 5,
            "max_parts": 9,
        }

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
    )
    # The classifier's initial bias is measured from this fold's training items only --
    # never validation or test -- so it leaks nothing about what it is scored on. It
    # needs N, which is settled just below, so the datamodule is set up first.
    args.class_prior = (tuple(float(v) for v in args.class_prior.split(","))
                        if args.class_prior else None)

    if args.num_candidates is None:
        args.num_candidates = choose_num_candidates(
            args.train_length, args.fps, args.bpm_max)
    floor = n_candidates_from_tempo(args.train_length, args.fps, args.bpm_max)
    if args.num_candidates < floor:
        # Below the floor an order-preserving injection may not exist, and
        # SubsetCriterion silently drops those fragments from the loss rather than
        # training in the wrong direction. Refuse instead of discovering it in a log.
        parser.error(
            f"--num_candidates {args.num_candidates} is below the floor {floor} implied "
            f"by --bpm_max {args.bpm_max:g} over a {args.train_length}-frame window; "
            f"fragments denser than N would be dropped from the loss.")
    print(f"[subset] N = {args.num_candidates} candidates "
          f"(floor {floor} from bpm_max={args.bpm_max:g}, "
          f"window={args.train_length}f @ {args.fps}fps)")
    if args.dbn and args.head_type == "subset":
        # The DBN postprocessor consumes frame-wise activations; the alignment head
        # emits per-candidate events and never builds them, so use_dbn would be silently
        # ignored. Refuse rather than report DBN numbers that were not produced by one.
        parser.error("--dbn is not supported with --head_type subset: the alignment "
                     "head emits events, not the frame-wise activations a DBN needs.")

    datamodule.setup(stage="fit")

    # compute positive weights
    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    print("Using positive weights: ", pos_weights)
    dropout = {
        "frontend": args.frontend_dropout,
        "transformer": args.transformer_dropout,
    }
    if args.class_prior is None:
        datamodule.setup("fit")
        window_seconds = args.train_length / float(args.fps)
        n_effective = args.num_candidates
        if args.downsample_stages is not None:
            n_effective = halved_candidates(args.train_length, args.downsample_stages)
        args.class_prior = datamodule.subset_class_prior(
            n_effective, window_seconds,
            omega_downbeat=args.omega_db, gamma=args.gamma)

    pl_model = PLBeatThis(
        spect_dim=128,
        fps=50,
        transformer_dim=args.transformer_dim,
        ff_mult=4,
        n_layers=args.n_layers,
        stem_dim=32,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weights=pos_weights,
        head_dim=32,
        loss_type=args.loss,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head,
        partial_transformers=args.partial_transformers,
        head_type=args.head_type,
        head_lr=args.head_lr,
        quantize_targets=args.quantize_targets,
        num_candidates=args.num_candidates,
        stitch_border=args.stitch_border,
        downsample_mode=args.downsample_mode,
        train_length=args.train_length,
        downsample_stages=args.downsample_stages,
        class_prior=args.class_prior,
        class_attention_layers=args.class_attention_layers,
        class_attention_heads=args.class_attention_heads,
        tau_beat=args.tau_beat,
        tau_downbeat=args.tau_downbeat,
        db_margin=args.db_margin,
        # Every SubsetCriterion knob is passed explicitly. A construction that is
        # implemented but has no path from the CLI is worse than one that is absent:
        # an ablation of it shows no difference and reads as "the idea does not help",
        # when in fact it never ran.
        subset_kwargs={
            "gamma": args.gamma,
            "omega_downbeat": args.omega_db,
            "joint_phase": args.joint_phase,
            "meter_length": args.meter_L,
            "meter_candidates": (tuple(int(v) for v in args.meter_candidates.split(","))
                                 if args.meter_candidates else ()),
            "meter_prior": args.meter_prior or None,
            "mu_meter": args.mu_meter,
        },
    )
    # --- frozen-encoder head swap -------------------------------------------------
    # Isolates "is the encoder good enough" from "is the head lossy". Both arms share
    # the identical frontend + transformer_blocks and differ only in task_heads, so
    # loading a converged encoder and training ONLY the head answers whether the
    # candidate representation can express what the dense per-frame one can.
    if args.init_encoder_from:
        blob = torch.load(args.init_encoder_from, map_location="cpu", weights_only=False)
        src = blob["state_dict"]
        loaded, skipped = 0, 0
        own = pl_model.state_dict()
        transfer = {}
        for k, v in src.items():
            if not (k.startswith("model.frontend") or k.startswith("model.transformer_blocks")):
                continue
            if k in own and own[k].shape == v.shape:
                transfer[k] = v; loaded += 1
            else:
                skipped += 1
        assert loaded > 0, f"no encoder weights matched from {args.init_encoder_from}"
        assert skipped == 0, f"{skipped} encoder tensors did not match shape; wrong config?"
        missing, unexpected = pl_model.load_state_dict(transfer, strict=False)
        print(f"[encoder-init] loaded {loaded} tensors from "
              f"{os.path.basename(args.init_encoder_from)} (epoch {blob.get('epoch')})",
              flush=True)
    if args.init_all_from:
        # Full-model init (encoder AND head), no optimizer/epoch state: start a FRESH
        # schedule from an already-converged system. Used for the thaw experiment --
        # take the frozen-encoder run (vanilla encoder + head trained on top of it, a
        # converged pair) and release the encoder underneath a head that already works.
        # This separates a TRANSIENT cause of the end-to-end inversion (a random head
        # damages a good encoder early) from a STEADY-STATE one (the loss degrades
        # encoders no matter what).
        blob = torch.load(args.init_all_from, map_location="cpu", weights_only=False)
        missing, unexpected = pl_model.load_state_dict(blob["state_dict"], strict=False)
        crit = [k for k in missing if "frontend" in k or "transformer_blocks" in k or "task_heads" in k]
        assert not crit, f"init_all_from missing core weights: {crit[:4]}"
        print(f"[init-all] loaded full model from {os.path.basename(args.init_all_from)} "
              f"(epoch {blob.get('epoch')}); {len(missing)} missing, {len(unexpected)} unexpected",
              flush=True)

    if args.freeze_encoder:
        n = 0
        for mod in (pl_model.model.frontend, pl_model.model.transformer_blocks):
            for prm in mod.parameters():
                prm.requires_grad_(False); n += 1
        pl_model.model.frontend.eval()
        pl_model.model.transformer_blocks.eval()
        trainable = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pl_model.parameters())
        print(f"[freeze] froze {n} encoder tensors; trainable {trainable/1e6:.2f}M "
              f"of {total/1e6:.2f}M params", flush=True)

    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            raise ValueError("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.snapshot_every:
        # Keep a checkpoint per N epochs instead of overwriting one file. Validation
        # reads WHOLE pieces (571 songs of ~95 s) rather than 30 s training crops, so on
        # a shared machine it is the dominant disk cost -- measured at 11-12% iowait with
        # several jobs blocked in D state, dropping training from 6.1 to 0.86 it/s.
        # Snapshotting lets training run with validation disabled entirely
        # (--val-frequency larger than --max-epochs) and the curve reconstructed
        # afterwards by scoring the snapshots, which is what score_fold0_snapshots.py
        # already does for the vanilla arms.
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=args.snapshot_every,
                save_top_k=-1,
                save_on_train_epoch_end=True,
                dirpath=str(checkpoint_dir),
                filename=f"{args.name} S{args.seed} {params_str}".strip() + "-ep{epoch:03d}",
            )
        )
    else:
        # save only the last model
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=1,
                dirpath=str(checkpoint_dir),
                filename=f"{args.name} S{args.seed} {params_str}".strip(),
            )
        )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=[args.gpu],
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
    )

    trainer.fit(pl_model, datamodule, ckpt_path=args.resume_checkpoint)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--force-flash-attention", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--compile",
        action="store",
        nargs="*",
        type=str,
        default=["frontend", "transformer_blocks", "task_heads"],
        help="Which model parts to compile, among frontend, transformer_encoder, task_heads",
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument(
        "--frontend-dropout",
        type=float,
        default=0.1,
        help="dropout rate to apply in the frontend",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply in the main transformer blocks",
    )
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50, help="The spectrograms fps.")
    parser.add_argument(
        "--loss",
        type=str,
        default="shift_tolerant_weighted_bce",
        choices=[
            "shift_tolerant_weighted_bce",
            "fast_shift_tolerant_weighted_bce",
            "weighted_bce",
            "bce",
        ],
        help="The loss to use",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="warmup steps for optimizer"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="max epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for training"
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    # --- order-preserving alignment head -------------------------------------------
    # Everything before the head is shared with the dense baseline, so
    # "--head_type subset" against the default is a controlled A/B: same encoder, data,
    # augmentation, trainer, precision and metrics, differing only in the head and the
    # loss it brings with it.
    parser.add_argument("--snapshot_every", type=int, default=0,
                        help="keep a checkpoint every N epochs (0 = overwrite one file); "
                             "pair with a large --val-frequency to train without eval")
    parser.add_argument("--head_type", type=str, default="dense",
                        choices=["dense", "subset"])
    parser.add_argument("--train_length", type=int, default=1500,
                        help="T, the excerpt length in frames; N is derived from it")
    parser.add_argument("--stitch_border", type=int, default=None,
                        help="frames discarded either side of a chunk seam at whole-piece "
                             "inference; defaults to the dense arm's 2*tolerance so both "
                             "A/B arms decode under the same edge convention")
    parser.add_argument("--class_prior", type=str, default="",
                        help="classifier bias init, e.g. 0.1,0.3,0.6. Empty measures it "
                             "from this fold's TRAINING split, weighted by --omega_db "
                             "and --gamma so the init sits at the loss's own optimum.")
    parser.add_argument("--meter_candidates", type=str, default="",
                        help="latent meter: e.g. 2,3,4,6. Empty keeps L fixed at --meter_L")
    parser.add_argument("--meter_prior", type=str, default="",
                        help="'corpus' uses the measured meter distribution from "
                             "docs/METER_DISTRIBUTION.md; empty is uniform over (L, phi_0)")
    parser.add_argument("--downsample_stages", type=int, default=None,
                        help="Strict halving: N is derived as ceil(T / 2**stages) "
                             "(1500 -> 750 -> 375 -> 188 for 3) instead of factorising "
                             "T/num_candidates. Overrides --num_candidates.")
    parser.add_argument("--downsample_mode", type=str, default="learned",
                        choices=["learned", "avg", "max"],
                        help="how T frames become N candidates: a strided conv "
                             "(learned, best so far), or parameter-free avg/max pooling")
    parser.add_argument("--bpm_max", type=float, default=BPM_MAX,
                        help=f"fastest tempo the corpus contains (default {BPM_MAX:g}); "
                             "N is derived from it and the window length")
    parser.add_argument("--num_candidates", type=int, default=None,
                        help="N, the number of candidates the head emits. Derived from "
                             "--bpm_max and --train_length when not given; pass it to "
                             "overgenerate beyond that floor.")
    parser.add_argument("--class_attention_layers", type=int, default=0)
    parser.add_argument("--class_attention_heads", type=int, default=4)
    parser.add_argument("--tau_beat", type=float, default=0.2)
    parser.add_argument("--tau_downbeat", type=float, default=0.2)
    parser.add_argument("--init_encoder_from", type=str, default="",
                        help="checkpoint to copy frontend+transformer_blocks from; "
                             "the head is always freshly initialised")
    parser.add_argument("--head_lr", type=float, default=0.0,
                        help="separate lr for the head (task_heads + criterion); 0 = use --lr "
                             "for everything. Lets the encoder train at the rate that suits "
                             "it while the head keeps a rate it is stable at.")
    parser.add_argument("--init_all_from", type=str, default="",
                        help="load the FULL model (encoder+head) from a checkpoint, with a "
                             "fresh optimizer and schedule; for thawing a converged pair")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="train only task_heads, with the encoder held fixed")
    parser.add_argument("--db_margin", type=float, default=0.0,
                        help="B-vs-DB decode margin: call DOWNBEAT only if "
                             "log p(DB) - log p(B) exceeds this. log(omega_db) "
                             "undoes the class-weighted training bias; 0 keeps "
                             "Algorithm 10's plain argmax. See decode_events.")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--omega_db", type=float, default=2.0)
    parser.add_argument("--joint_phase", action="store_true", default=False)
    parser.add_argument("--meter_L", type=int, default=0)
    parser.add_argument("--mu_meter", type=float, default=0.0,
                        help="Section 4.2 eq. (6): known-meter spacing inside the selection")
    parser.add_argument("--quantize_targets", action="store_true", default=False,
                        help="round ground-truth event times to the frame grid, matching "
                             "what the dense head is necessarily trained on")
    # NOTE: --train_length (underscore) is defined above and owns dest=train_length.
    # A second "--train-length" action used to be declared here with its own default;
    # both wrote the same dest, so whichever was declared later silently won. Kept as a
    # pure alias so existing command lines keep working, with no competing default.
    parser.add_argument("--train-length", dest="train_length", type=int,
                        help="alias of --train_length")
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use madmom postprocessing DBN",
    )
    parser.add_argument(
        "--eval-trim-beats",
        metavar="SECONDS",
        type=float,
        default=5,
        help="Skip the first given seconds per piece in evaluating (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frequency",
        metavar="N",
        type=int,
        default=5,
        help="validate every N epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed tempo aumentation",
    )
    parser.add_argument(
        "--pitch-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed pitch aumentation",
    )
    parser.add_argument(
        "--mask-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use online mask aumentation",
    )
    parser.add_argument(
        "--sum-head",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use SumHead instead of two separate Linear heads",
    )
    parser.add_argument(
        "--partial-transformers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use Partial transformers in the frontend",
    )
    parser.add_argument(
        "--length-based-oversampling-factor",
        type=float,
        default=0.65,
        help="The factor to oversample the long pieces in the dataset. Set to 0 to only take one excerpt for each piece.",
    )
    parser.add_argument(
        "--val",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train on all data, including validation data, escluding test data. The validation metrics will still be computed, but they won't carry any meaning.",
    )
    parser.add_argument(
        "--hung-data",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Limit the training to Hung et al. data. The validation will still be computed on all datasets.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If given, the CV fold number to *not* train on (0-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generators.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume training from a local checkpoint.",
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        default=None,
        help="When resuming with --resume-checkpoint, optionally provide the wandb id to continue logging to.",
    )

    args = parser.parse_args()

    main(args)
