"""
Pytorch Lightning module, wraps a BeatThis model along with losses, metrics and
optimizers for training.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mir_eval
import numpy as np
import torch
from pytorch_lightning import LightningModule

import beat_this.model.loss
from beat_this.inference import split_predict_aggregate
import numpy as np

from beat_this.model.beat_tracker import BeatThis
from alignbeat.classes import BEAT, CLASS_UNKNOWN, DOWNBEAT
from alignbeat.criterion import SubsetCriterion
from alignbeat.decode import decode_events
from beat_this.model.postprocessor import Postprocessor
from beat_this.utils import replace_state_dict_key


class PLBeatThis(LightningModule):
    def __init__(
        self,
        spect_dim=128,
        fps=50,
        transformer_dim=512,
        ff_mult=4,
        n_layers=6,
        stem_dim=32,
        dropout={"frontend": 0.1, "transformer": 0.2},
        lr=0.0008,
        weight_decay=0.01,
        pos_weights={"beat": 1, "downbeat": 1},
        head_dim=32,
        loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000,
        max_epochs=100,
        use_dbn=False,
        eval_trim_beats=5,
        sum_head=True,
        partial_transformers=True,
        head_type: str = "dense",
        head_lr: float = 0.0,
        quantize_targets: bool = False,
        stitch_border: int = None,
        # Only used by head_type="subset": N, the number of candidates the head
        # emits. Derived once in launch_scripts/train.py; no default here, so the
        # value can never disagree with the one a run was configured with.
        num_candidates: int = None,
        downsample_mode: str = "learned",
        train_length: int = 1500,
        downsample_stages: int = None,
        class_attention_layers: int = 0,
        class_attention_heads: int = 4,
        subset_kwargs: dict = None,
        tau_beat: float = 0.2,
        tau_downbeat: float = 0.2,
        db_margin: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.fps = fps
        self.quantize_targets = quantize_targets
        self.stitch_border = stitch_border
        self.head_lr = head_lr
        self.tau_beat = tau_beat
        self.tau_downbeat = tau_downbeat
        # Decode-time Bayes correction for omega_DB-weighted training; see
        # decode_events. Enters through __init__ with a default so old checkpoints
        # still load, and can be overridden at load_from_checkpoint time.
        self.db_margin = db_margin
        # create model
        self.model = BeatThis(
            head_type=head_type,
            spect_dim=spect_dim,
            transformer_dim=transformer_dim,
            ff_mult=ff_mult,
            stem_dim=stem_dim,
            n_layers=n_layers,
            head_dim=head_dim,
            dropout=dropout,
            sum_head=sum_head,
            partial_transformers=partial_transformers,
            num_candidates=num_candidates,
            downsample_mode=downsample_mode,
            train_length=train_length,
            fps=fps,
            downsample_stages=downsample_stages,
            class_attention_layers=class_attention_layers,
            class_attention_heads=class_attention_heads,
        )
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        # set up the losses
        self.pos_weights = pos_weights
        # The order-preserving alignment head brings its own loss: the DP selects which
        # candidates are responsible for which events, and the loss is evaluated at that
        # selection. Nothing frame-wise applies, so the BCE variants below are skipped.
        self.subset_criterion = None
        if head_type == "subset":
            self.subset_criterion = SubsetCriterion(**(subset_kwargs or {}))
        elif loss_type == "shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.ShiftTolerantBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.ShiftTolerantBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        elif loss_type == "weighted_bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        elif loss_type == "bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss()
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss()
        elif loss_type == "splitted_shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.SplittedShiftTolerantBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.SplittedShiftTolerantBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        else:
            raise ValueError(
                "loss_type must be one of 'shift_tolerant_weighted_bce', 'weighted_bce', 'bce'"
            )

        self.postprocessor = Postprocessor(
            type="dbn" if use_dbn else "minimal", fps=fps
        )
        self.eval_trim_beats = eval_trim_beats
        self.metrics = Metrics(eval_trim_beats=eval_trim_beats)

    def _subset_decode(self, batch, model_prediction):
        """Algorithm 10 per excerpt, returned as predicted TIMES in seconds."""
        num_frames = batch["truth_beat"].shape[-1]
        window_seconds = num_frames / self.fps
        padding_mask = batch.get("padding_mask")
        beats, downbeats = [], []
        for index in range(len(batch["spect"])):
            classes, times, _scores = decode_events(
                model_prediction["class_logits"][index].float(),
                model_prediction["t_hat"][index].float(),
                self.tau_beat, self.tau_downbeat,
                db_margin=self.db_margin)
            seconds = (times * window_seconds).detach().cpu().numpy()
            classes = classes.detach().cpu().numpy()
            if padding_mask is not None:
                # The dense arm passes padding_mask to its postprocessor; without the
                # same restriction here, candidates landing in an excerpt's zero-padded
                # tail are emitted as detections that no ground-truth event can match
                # (truth_orig_* stops at the real end), so they are pure false positives
                # charged to one arm of the A/B only.
                valid_seconds = float(padding_mask[index].sum()) / self.fps
                keep = seconds < valid_seconds
                seconds, classes = seconds[keep], classes[keep]
            beats.append(np.sort(seconds))
            downbeats.append(np.sort(seconds[classes == DOWNBEAT]))
        return tuple(beats), tuple(downbeats)

    def _subset_targets(self, batch):
        """Ground-truth events for the alignment head, from this batch's own annotations."""
        num_frames = batch["truth_beat"].shape[-1]
        window_seconds = num_frames / self.fps
        device = batch["spect"].device
        targets = []
        for index in range(len(batch["spect"])):
            beats = np.frombuffer(batch["truth_orig_beat"][index])
            downbeats = np.frombuffer(batch["truth_orig_downbeat"][index])
            has_downbeats = bool(batch["downbeat_mask"][index])

            # eq. (1) maps onto the half-open axis (0, 1], so a target at exactly 0 is
            # unreachable by construction and would be an unmatchable event.
            keep = (beats > 0) & (beats <= window_seconds)
            beats = np.unique(beats[keep])   # unique, not just sorted: Definition 1
            if self.quantize_targets:
                # Round to the frame grid, which is what the DENSE head is necessarily
                # trained on (its output is per-frame, so it cannot represent sub-frame
                # targets). Off by default: eq. (1) produces a continuous time and
                # Definition 1 is stated over continuous ground truth, so quantizing
                # would degrade this head to match a limitation of the other one.
                # Exposed as a flag because it is a real asymmetry in the A/B -- the
                # subset arm otherwise sees ground truth up to 1/(2*fps) = 10 ms more
                # precise than the dense arm does -- and its size should be measured
                # rather than argued about.
                beats = np.round(beats * self.fps) / self.fps
            if has_downbeats:
                classes = np.where(np.isin(beats, downbeats), DOWNBEAT, BEAT)
            else:
                classes = np.full(len(beats), CLASS_UNKNOWN)
            targets.append({
                "times": torch.as_tensor(beats / window_seconds,
                                         dtype=torch.float32, device=device),
                "classes": torch.as_tensor(classes, dtype=torch.long, device=device),
            })
        return targets

    def _compute_loss(self, batch, model_prediction):
        if self.subset_criterion is not None:
            losses, _stats = self.subset_criterion(
                model_prediction["class_logits"].float(),
                model_prediction["t_hat"].float(),
                model_prediction["b_hat"].float(),
                self._subset_targets(batch))

            # Keys kept as "beat"/"downbeat" so log_losses and every downstream reader
            # are unchanged; they carry the class and timing terms of loss (8).
            return {"beat": losses["class"], "downbeat": losses["time"],
                    "total": losses["total"]}

        beat_mask = batch["padding_mask"]
        beat_loss = self.beat_loss(
            model_prediction["beat"], batch["truth_beat"].float(), beat_mask
        )
        # downbeat mask considers padding and also pieces which don't have downbeat annotations
        downbeat_mask = beat_mask * batch["downbeat_mask"][:, None]
        downbeat_loss = self.downbeat_loss(
            model_prediction["downbeat"], batch["truth_downbeat"].float(), downbeat_mask
        )
        # sum the losses and return them in a dictionary for logging
        return {
            "beat": beat_loss,
            "downbeat": downbeat_loss,
            "total": beat_loss + downbeat_loss,
        }

    def _compute_metrics(self, batch, postp_beat, postp_downbeat, step="val"):
        """ """
        # compute for beat
        metrics_beat = self._compute_metrics_target(
            batch, postp_beat, target="beat", step=step
        )
        # compute for downbeat
        metrics_downbeat = self._compute_metrics_target(
            batch, postp_downbeat, target="downbeat", step=step
        )

        # concatenate dictionaries
        metrics = {**metrics_beat, **metrics_downbeat}

        return metrics

    def _compute_metrics_target(self, batch, postp_target, target, step):

        def compute_item(pospt_pred, truth_orig_target):
            # take the ground truth from the original version, so there are no quantization errors
            piece_truth_time = np.frombuffer(truth_orig_target)
            # run evaluation
            metrics = self.metrics(piece_truth_time, pospt_pred, step=step)

            return metrics

        # if the input was not batched, postp_target is an array instead of a tuple of arrays
        # make it a tuple for consistency
        if not isinstance(postp_target, tuple):
            postp_target = (postp_target,)

        with ThreadPoolExecutor() as executor:
            piecewise_metrics = list(
                executor.map(
                    compute_item,
                    postp_target,
                    batch[f"truth_orig_{target}"],
                )
            )

        # average the beat metrics across the dictionary
        batch_metric = {
            key + f"_{target}": np.mean([x[key] for x in piecewise_metrics])
            for key in piecewise_metrics[0].keys()
        }

        return batch_metric

    def log_losses(self, losses, batch_size, step="train"):
        # log for separate targets
        for target in "beat", "downbeat":
            self.log(
                f"{step}_loss_{target}",
                losses[target].item(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        # log total loss
        self.log(
            f"{step}_loss",
            losses["total"].item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def log_metrics(self, metrics, batch_size, step="val"):
        for key, value in metrics.items():
            self.log(
                f"{step}_{key}",
                value,
                prog_bar=key.startswith("F-measure"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        if self.subset_criterion is not None:
            # Frozen for the first 30% of the RUN, then thawed: batch_idx counts
            # batches within an epoch, so comparing it against an epoch count froze the
            # head for the first 30 batches of every epoch and never for a 1-epoch run.
            # The bias stays frozen for the whole run either way (see SubsetHead), so
            # only the weights move -- the head can redistribute precision across
            # candidates but not shift its overall scale.
            train_precision = self.current_epoch >= self.max_epochs * 0.3
            self.model.task_heads.head.precision_head.weight.requires_grad_(train_precision)

        # run the model
        model_prediction = self.model(batch["spect"])

        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        self.log_losses(losses, len(batch["spect"]), "train")
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        # run the model
        model_prediction = self.model(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        # postprocess the predictions
        if self.subset_criterion is not None:
            postp_beat, postp_downbeat = self._subset_decode(batch, model_prediction)
        else:
            postp_beat, postp_downbeat = self.postprocessor(
                model_prediction["beat"],
                model_prediction["downbeat"],
                batch["padding_mask"],
            )
        # compute the metrics
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="val")
        # log
        self.log_losses(losses, len(batch["spect"]), "val")
        self.log_metrics(metrics, batch["spect"].shape[0], "val")

    def test_step(self, batch, batch_idx):
        metrics, model_prediction, _, _ = self.predict_step(batch, batch_idx)
        # The alignment head returns no piece-level framewise prediction (its loss is
        # defined over a fixed-length excerpt against a matched set of events, which a
        # whole piece is not), so there is no test loss to log -- only metrics, which
        # are the comparable quantity anyway.
        if model_prediction is not None:
            losses = self._compute_loss(batch, model_prediction)
            self.log_losses(losses, len(batch["spect"]), "test")
        self.log_metrics(metrics, batch["spect"].shape[0], "test")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        chunk_size: int = 1500,
        overlap_mode: str = "keep_first",
    ) -> Any:
        """
        Compute predictions and metrics for a batch (a dictionary with an "spect" key).
        It splits up the audio into multiple chunks of chunk size,
         which should correspond to the length of the sequence the model was trained with.
        Potential overlaps between chunks can be handled in two ways:
        by keeping the predictions of the excerpt coming first (overlap_mode='keep_first'), or
        by keeping the predictions of the excerpt coming last (overlap_mode='keep_last').
        Note that overlaps appear as the last excerpt is moved backwards
        when it would extend over the end of the piece.
        """
        if batch["spect"].shape[0] != 1:
            raise ValueError(
                "When predicting full pieces, only `batch_size=1` is supported"
            )
        if torch.any(~batch["padding_mask"]):
            raise ValueError(
                "When predicting full pieces, the Dataset must not pad inputs"
            )
        if self.subset_criterion is not None:
            return self._subset_predict_piece(batch, chunk_size, overlap_mode)

        # compute border size according to the loss type
        if hasattr(
            self.beat_loss, "tolerance"
        ):  # discard the edges that are affected by the max-pooling in the loss
            border_size = 2 * self.beat_loss.tolerance
        else:
            border_size = 0
        model_prediction = split_predict_aggregate(
            batch["spect"][0], chunk_size, border_size, overlap_mode, self.model
        )
        # add the batch dimension back in the prediction for consistency
        model_prediction = {
            key: value.unsqueeze(0) for key, value in model_prediction.items()
        }
        # postprocess the predictions
        postp_beat, postp_downbeat = self.postprocessor(
            model_prediction["beat"], model_prediction["downbeat"], None
        )
        # compute the metrics
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="test")
        return metrics, model_prediction, batch["dataset"], batch["spect_path"]

    def _subset_predict_piece(self, batch, chunk_size, overlap_mode):
        """Whole-piece decoding for the alignment head (Section 9.3)."""
        from beat_this.inference import split_piece

        spect = batch["spect"][0]
        # Border at chunk seams. The dense path always discards 2*tolerance frames
        # either side of an internal boundary (aggregate_prediction), and Section 9.3
        # likewise requires beta in (0, D/2) so that a candidate near a fragment edge is
        # never the ONLY candidate covering that moment. Using 0 here left the two arms
        # decoding under different edge conventions, which is not a head difference.
        # Default matches whatever the dense arm uses, so the A/B stays controlled.
        border = self.stitch_border
        if border is None:
            border = 2 * getattr(self.beat_loss, "tolerance", 3) if hasattr(
                self, "beat_loss") else 6
        chunks, starts = split_piece(spect, chunk_size, border_size=border,
                                     avoid_short_end=True)
        # Section 9.3 / Algorithm 11: fragment k owns [o_k + beta, o_k + D - beta], with
        # the edge exceptions that the FIRST fragment owns from 0 and the LAST owns to
        # the end of the piece. Previously each chunk owned its full span, so its
        # trailing border -- the lowest-context frames it has, and exactly the ones the
        # dense arm discards in aggregate_prediction -- won every seam over the next
        # chunk's interior. That is an edge-convention difference between the two A/B
        # arms, not a head difference.
        piece_end = spect.shape[0] / self.fps
        covered_to = 0.0          # high-water mark: end of what earlier fragments own
        beats, downbeats = [], []
        n_chunks = len(chunks)
        for index, (chunk, start) in enumerate(zip(chunks, starts)):
            prediction = self.model(chunk.unsqueeze(0))
            classes, times, _scores = decode_events(
                prediction["class_logits"][0].float(),
                prediction["t_hat"][0].float(),
                self.tau_beat, self.tau_downbeat,
                db_margin=self.db_margin)
            seconds = (start + times * chunk.shape[0]).detach().cpu().numpy() / self.fps
            classes = classes.detach().cpu().numpy()

            # keep region for this fragment, in seconds. split_piece(avoid_short_end)
            # shifts the LAST chunk's start left so it ends exactly at the piece end,
            # which can overlap its predecessor by far more than `border` -- so the
            # nominal starts are NOT uniformly strided and a naive start+border can land
            # before the previous fragment's own_end, re-emitting every event in between.
            # Clamping to the running high-water mark keeps the keep regions a true
            # partition, which is what Section 9.3 requires.
            own_start = (start + border) / self.fps if index > 0 else 0.0
            own_start = max(own_start, covered_to)
            own_end = ((start + chunk.shape[0] - border) / self.fps
                       if index < n_chunks - 1 else piece_end)
            own_end = max(own_end, own_start)
            keep = (seconds >= own_start) & (seconds < own_end)
            # The final chunk is right-zero-padded and t_hat is stretched over that
            # padding, so clamp to the real end of the piece rather than emitting
            # events into silence. The first chunk's left pad is handled by own_start.
            keep &= (seconds >= 0.0) & (seconds <= piece_end)
            seconds, classes = seconds[keep], classes[keep]
            covered_to = max(covered_to, own_end)

            beats.append(seconds)
            downbeats.append(seconds[classes == DOWNBEAT])

        beats = (np.sort(np.concatenate(beats)),) if beats else (np.zeros(0),)
        downbeats = (np.sort(np.concatenate(downbeats)),) if downbeats else (np.zeros(0),)
        metrics = self._compute_metrics(batch, beats, downbeats, step="test")
        # model_prediction is returned for the caller's loss computation; the alignment
        # head has no piece-level framewise prediction to hand back, so None is honest.
        return metrics, None, batch["dataset"], batch["spect_path"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW
        # only decay 2+-dimensional tensors, to exclude biases and norms
        # (filtering on dimensionality idea taken from Kaparthy's nano-GPT)
        # Discriminative learning rates. The encoder and the head have different
        # stability requirements and there is no reason they must share an lr: the dense
        # baseline reaches its best at 8e-4, but the subset head collapses there (0.617
        # at ep4 -> 0.545 at ep9, measured), which forced every subset arm to 3e-4. That
        # costs the ENCODER real quality -- the dense control loses 0.025 joint at 3e-4
        # versus 8e-4 (0.894 vs 0.919). head_lr lets the encoder train at the rate that
        # suits it while the head keeps the rate that keeps it stable.
        def _is_head(name):
            return name.startswith("model.task_heads") or name.startswith("subset_criterion")
        head_lr = self.head_lr if self.head_lr > 0 else self.lr
        groups, seen = [], set()
        for tag, pred, lr in (("encoder", lambda n: not _is_head(n), self.lr),
                              ("head",    _is_head,                  head_lr)):
            for decay, keep in (("decay", lambda p: p.ndim >= 2), ("nodecay", lambda p: p.ndim <= 1)):
                ps = [p for n, p in self.named_parameters()
                      if p.requires_grad and pred(n) and keep(p) and id(p) not in seen]
                for p in ps: seen.add(id(p))
                if ps:
                    groups.append({"params": ps, "lr": lr,
                                   "weight_decay": self.weight_decay if decay == "decay" else 0.0,
                                   "name": f"{tag}.{decay}"})
        if head_lr != self.lr:
            print(f"[optim] discriminative lr: encoder {self.lr:g}, head {head_lr:g} "
                  f"({sum(len(g['params']) for g in groups if g['name'].startswith('head'))} head tensors)",
                  flush=True)
        params = groups

        optimizer = optimizer(params, lr=self.lr)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
        )

        result = dict(optimizer=optimizer)
        result["lr_scheduler"] = {"scheduler": self.lr_scheduler, "interval": "step"}
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class Metrics:
    def __init__(self, eval_trim_beats: int) -> None:
        self.min_beat_time = eval_trim_beats

    def __call__(self, truth, preds, step) -> Any:
        truth = mir_eval.beat.trim_beats(truth, min_beat_time=self.min_beat_time)
        preds = mir_eval.beat.trim_beats(preds, min_beat_time=self.min_beat_time)
        if (
            step == "val"
        ):  # limit the metrics that are computed during validation to speed up training
            fmeasure = mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {"F-measure": fmeasure, "Cemgil": cemgil}
        elif step == "test":  # compute all metrics during testing
            CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(truth, preds)
            fmeasure = mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {"F-measure": fmeasure, "Cemgil": cemgil, "CMLt": CMLt, "AMLt": AMLt}
        else:
            raise ValueError("step must be either val or test")


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing over `max_iters` steps with `warmup` linear warmup steps.
    Optionally re-raises the learning rate for the final `raise_last` fraction
    of total training time to `raise_to` of the full learning rate, again with
    a linear warmup (useful for stochastic weight averaging).
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
