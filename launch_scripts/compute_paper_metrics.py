#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

# Running this file directly puts launch_scripts/, not the repository root, at the
# front of sys.path. Prefer this fork's Beat This package over any installed upstream
# copy that may also be present in the environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pytorch_lightning import Trainer, seed_everything

# Repo checkpoints contain numpy values and PosixPaths rejected by torch's
# weights_only=True loader. Use weights_only=False only for checkpoints written here.
import torch
_torch_load = torch.load
def _load_trusted(*a, **k):
    k["weights_only"] = False
    return _torch_load(*a, **k)
torch.load = _load_trusted

from beat_this.dataset import BeatDataModule
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis
from beat_this.utils import infer_beat_numbers

seed_everything(0, workers=True)


def downbeat_annotated(predict_dataset):
    """Map each dataset label the loader reports to whether it annotates downbeats.

    Read off predict_dataset.items, which is where the loader itself gets it: the flag
    is data/annotations/<dataset>/info.json's has_downbeats, and the item carries it
    already paired with the label the metrics are grouped by -- including rwc's
    per-section labels (rwc_classical and friends), which are renamed after the flag is
    resolved. So this needs no knowledge of which corpora lack downbeats.
    """
    return {item["dataset"]: bool(item["downbeat_mask"])
            for item in predict_dataset.items}


def is_downbeat_metric(key):
    return "downbeat" in key


def average_metrics(metrics, dataset, has_downbeats):
    """Mean over pieces, excluding pieces with no downbeat reference from downbeat means.

    mir_eval returns 0.0 when the reference is empty, so a beat-only corpus scores 0 on
    every downbeat metric. Averaging those in measures how much of the split is
    beat-only rather than how the model does -- on the 8-fold val split simac and smc are
    304 of 1715 pieces, which drags the downbeat means by roughly 18%. Beat metrics still
    use every piece; only the downbeat means are restricted.
    """
    scored = np.asarray([has_downbeats.get(d, True) for d in dataset])
    out = {}
    for k, v in metrics.items():
        keep = scored if is_downbeat_metric(k) else np.ones(len(v), dtype=bool)
        out[k] = float(np.mean(v[keep])) if keep.any() else float("nan")
    return out


def average_metrics_by_dataset(metrics, dataset, has_downbeats):
    """Per-dataset means, with a downbeat metric on a beat-only corpus left as nan.

    Reporting 0.0 there invites it into a table as if it were a score; nan does not.
    """
    out = {}
    for k, v in metrics.items():
        per = {}
        for d in np.unique(dataset):
            if is_downbeat_metric(k) and not has_downbeats.get(d, True):
                per[d] = float("nan")
            else:
                per[d] = float(np.mean(v[dataset == d]))
        out[k] = per
    return out


def format_metric(value):
    return "n/a (no downbeat annotation)" if np.isnan(value) else round(value, 3)


def main(args):
    if len(args.models) == 1:
        print("Single model prediction for", args.models[0])
        checkpoint_path = args.models[0]
        checkpoint = load_checkpoint(checkpoint_path)

        datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
        model, trainer = plmodel_setup(
            checkpoint, args.eval_trim_beats, args.dbn, args.gpu,
            args.read_out, args.stitch_border
        )
        metrics, dataset, preds, piece = compute_predictions(
            model,
            trainer,
            datamodule.predict_dataloader(),
            return_preds=args.dump_predictions,
        )

        has_downbeats = downbeat_annotated(datamodule.predict_dataset)
        averaged_metrics = average_metrics(metrics, dataset, has_downbeats)
        dataset_metrics = average_metrics_by_dataset(metrics, dataset, has_downbeats)
        print("Metrics (downbeat metrics over annotated pieces only)")
        for k, v in averaged_metrics.items():
            print(f"{k}: {v}")
        print("Dataset metrics")
        for k, v in dataset_metrics.items():
            print(k)
            for d, value in v.items():
                print(f"{d}: {format_metric(value)}")
            print("------")
        if args.dump_piece_metrics:
            write_piece_metrics(args.dump_piece_metrics, piece, dataset, metrics)
        if args.dump_predictions:
            write_predictions(args.dump_predictions, preds, piece)
    else:  # multiple models
        if args.aggregation_type == "mean-std":
            if args.dump_predictions:
                print(
                    "cannot dump predictions when doing inference for multiple models"
                )
                return
            # All models use the same dataset, so reuse one datamodule.
            checkpoint = load_checkpoint(args.models[0])
            datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
            all_metrics = []
            for checkpoint_path in args.models:
                checkpoint = load_checkpoint(checkpoint_path)
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.gpu,
                    args.read_out, args.stitch_border
                )

                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                has_downbeats = downbeat_annotated(datamodule.predict_dataset)
                averaged_metrics = average_metrics(metrics, dataset, has_downbeats)
                all_metrics.append(averaged_metrics)
            all_metrics_mean = {
                k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_std = {
                k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_stats = {
                k: (all_metrics_mean[k], all_metrics_std[k])
                for k, v in all_metrics[0].items()
            }
            print("Metrics (downbeat metrics over annotated pieces only)")
            for k, v in all_metrics_stats.items():
                print(f"{k}: {round(v[0],3)} +- {round(v[1],3)}")
        elif args.aggregation_type == "k-fold":
            all_piece_metrics = []
            all_piece_dataset = []
            all_piece_preds = []
            all_piece = []
            # has_downbeats is a per-corpus flag, so the folds agree wherever they
            # overlap; collect it across them because no single fold sees every corpus.
            has_downbeats = {}
            for i_model, checkpoint_path in enumerate(args.models):
                print(f"Model {i_model+1}/{len(args.models)}")
                checkpoint = load_checkpoint(checkpoint_path)
                datamodule = datamodule_setup(
                    checkpoint, args.num_workers, args.datasplit
                )
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.gpu,
                    args.read_out, args.stitch_border
                )
                metrics, dataset, preds, piece = compute_predictions(
                    model,
                    trainer,
                    datamodule.predict_dataloader(),
                    return_preds=args.dump_predictions,
                )
                has_downbeats.update(downbeat_annotated(datamodule.predict_dataset))
                all_piece_metrics.append(metrics)
                all_piece_dataset.append(dataset)
                if preds is not None:
                    all_piece_preds.extend(preds)
                all_piece.append(piece)
            all_piece_metrics = {
                k: np.concatenate([m[k] for m in all_piece_metrics])
                for k in all_piece_metrics[0]
            }
            all_piece_dataset = np.concatenate(all_piece_dataset)
            all_piece = np.concatenate(all_piece)
            # Each piece should occur in exactly one fold.
            assert len(all_piece) == len(
                np.unique(all_piece)
            ), "There are repeated pieces in the folds"
            if args.dump_piece_metrics:
                write_piece_metrics(args.dump_piece_metrics, all_piece,
                                    all_piece_dataset, all_piece_metrics)
            dataset_metrics = average_metrics_by_dataset(
                all_piece_metrics, all_piece_dataset, has_downbeats
            )
            # Pool pieces before averaging so each held-out piece has equal weight.
            averaged_metrics = average_metrics(
                all_piece_metrics, all_piece_dataset, has_downbeats
            )
            n_scored = sum(
                1 for d in all_piece_dataset if has_downbeats.get(d, True)
            )
            print(
                f"Metrics (k-fold over {len(all_piece)} pieces; downbeat metrics over "
                f"the {n_scored} with downbeat annotation)"
            )
            for k, v in averaged_metrics.items():
                print(f"{k}: {round(v,4)}")
            print("Dataset metrics")
            for k, v in dataset_metrics.items():
                print(k)
                for d, value in v.items():
                    print(f"{d}: {format_metric(value)}")
                print("------")
            if args.dump_predictions:
                write_predictions(args.dump_predictions, all_piece_preds, all_piece)
        else:
            raise ValueError(f"Unknown aggregation type {args.aggregation_type}")


def datamodule_setup(checkpoint, num_workers, datasplit):
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    datamodule_hparams = dict(checkpoint["datamodule_hyper_parameters"])
    # Older checkpoints recorded this training-only option; evaluation never applies it.
    datamodule_hparams.pop("downbeat_dropout", None)
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = datasplit
    datamodule_hparams["data_dir"] = data_dir
    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="predict")
    return datamodule


def plmodel_setup(checkpoint, eval_trim_beats, dbn, gpu, read_out=None,
                  stitch_border=None):
    """
    Set up the pytorch lightning model and trainer for evaluation.

    Args:
        checkpoint_path (dict): The dict containing the checkpoint to load.
        eval_trim_beats (int or None): The number of beats to trim during evaluation. If None, the setting is taken from the pretrained model.
        dbn (bool or None): Whether to use the Dynamic Bayesian Network (DBN) module during evaluation. If None, the default behavior from the pretrained model is used.
        gpu (int): The index of the GPU device to use for training.

    Returns:
        tuple: A tuple containing the initialized pytorch lightning model and trainer.

    """
    if eval_trim_beats is not None:
        checkpoint["hyper_parameters"]["eval_trim_beats"] = eval_trim_beats
    if dbn is not None:
        checkpoint["hyper_parameters"]["use_dbn"] = dbn

    hparams = dict(checkpoint["hyper_parameters"])
    # Map legacy head names so old AlignBeat checkpoints do not load as dense models.
    legacy_alignbeat = hparams.pop("subset_head", None)
    head_type = hparams.pop("head_type", None)
    if "alignbeat" not in hparams:
        hparams["alignbeat"] = (legacy_alignbeat if legacy_alignbeat is not None
                                else head_type == "subset")
    # Drop the training-only learning-rate option saved in older checkpoints.
    hparams.pop("head_lr", None)
    # Remove obsolete options no longer accepted by PLBeatThis.
    RETIRED = {"time_param", "tau", "stitch_border", "decode", "loss_event_term",
               "refine_meter", "duration_conditioned"}
    if hparams.get("subset_kwargs"):
        hparams["subset_kwargs"] = {k: v for k, v in hparams["subset_kwargs"].items()
                                    if k not in RETIRED}
        # These decode options are supplied at evaluation time; old checkpoints do not
        # contain the current settings.
        if read_out is not None:
            hparams["subset_kwargs"]["read_out"] = read_out
        if stitch_border is not None:
            hparams["subset_kwargs"]["stitch_border"] = stitch_border
    model = PLBeatThis(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    if gpu >= 0:
        devices = [gpu]
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=None,
        deterministic=True,
        precision="16-mixed",
    )
    return model, trainer


def compute_predictions(model, trainer, predict_dataloader, return_preds=False):
    print("Computing predictions ...")
    out = trainer.predict(model, predict_dataloader)
    metrics = [o[0] for o in out]
    if return_preds:
        # The alignment head returns decoded (beats, downbeats) in seconds directly;
        # only the dense head needs its activations run through the postprocessor.
        preds = [o[1] if isinstance(o[1], tuple)
                 else model.postprocessor(o[1]["beat"][0], o[1]["downbeat"][0])
                 for o in out]
    else:
        preds = None
    dataset = np.asarray([o[2][0] for o in out])
    piece = np.asarray([o[3][0] for o in out])
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}
    return metrics, dataset, preds, piece


def write_piece_metrics(fn, piece, dataset, metrics):
    """Per-piece metrics to CSV. Used by both the single-model and k-fold paths, which
    previously meant the flag was silently ignored for a single model."""
    import csv
    keys = sorted(metrics)
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["piece", "dataset"] + keys)
        for i in range(len(piece)):
            w.writerow([piece[i], dataset[i]] + [metrics[k][i] for k in keys])
    print(f"wrote per-piece metrics to {fn}")


def write_predictions(fn, preds, piece):
    np.savez(
        fn,
        **{
            name: np.vstack([beats, infer_beat_numbers(beats, downbeats)]).T
            for name, (beats, downbeats) in zip(piece, preds)
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and dataset, "
        "prints metrics, and optionally dumps predictions to a given file."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Local checkpoint files to use",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        choices=("train", "val", "test"),
        default="val",
        help="data split to use: train, val or test " "(default: %(default)s)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of data loading workers "
    )
    parser.add_argument(
        "--eval_trim_beats",
        metavar="SECONDS",
        type=float,
        default=None,
        help="Override whether to skip the first given seconds "
        "per piece in evaluating (default: as stored in model)",
    )
    parser.add_argument(
        "--dbn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="override the option to use madmom postprocessing dbn",
    )
    parser.add_argument(
        "--aggregation-type",
        type=str,
        choices=("mean-std", "k-fold"),
        default="mean-std",
        help="Type of aggregation to use for multiple models; ignored if only one model is given",
    )
    parser.add_argument(
        "--read-out",
        type=str,
        choices=("per_event", "map", "duration"),
        default=None,
        help="Which read-out to decode with, overriding the checkpoint: per_event is "
        "each candidate's own argmax and what the paper's tables report, map is "
        "Algorithm 5's joint mode over (omega, L), duration advances bar position by "
        "the elapsed time (see alignbeat.inference.decode.decode)",
    )
    parser.add_argument(
        "--stitch-border",
        metavar="FRAMES",
        type=int,
        default=None,
        help="Frames trimmed at each fragment seam, overriding the checkpoint. Default "
        "is section 9.3's D/N, one candidate cell (8 at N=188, D=1500). 0 abuts "
        "fragments with no overlap, so every seam keeps its own worst predictions.",
    )
    parser.add_argument(
        "--dump-piece-metrics",
        metavar="FILENAME",
        type=str,
        default=None,
        help="Write the per-piece metrics already computed for the k-fold "
        "aggregation to a CSV (reporting only; changes no metric)",
    )
    parser.add_argument(
        "--dump-predictions",
        metavar="FILENAME",
        type=str,
        default=None,
        help="File to write predictions to, in .npz format (optional)",
    )

    args = parser.parse_args()

    main(args)
