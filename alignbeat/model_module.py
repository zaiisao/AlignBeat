"""AlignBeat: beat tracking as latent order-preserving alignment.

One path, no branches: a Beat This! transformer encoder produces frame features
h_1..h_T (Section 3), progressive patch-merging downsamples them to N candidate
features z_0..z_{N-1} (eq. 1-3 of the downsampling schedule), and
SubsetSelectionHead emits a 3-way class distribution and a raw scalar per
candidate, the scalars turned into strictly increasing times by the cumulative
softplus reparameterization of eq. (1).

Training: SubsetCriterion solves the order-constrained subset-selection problem
by dynamic programming (Algorithm 1) and evaluates loss (8) under the resulting
sigma_hat held fixed.

Inference: no NMS and no postprocessing here -- the raw (class_logits, t_hat)
are returned so thresholds can be swept without rerunning the model, and
alignbeat/stitching.py handles decoding and fragment reassembly (Section 9.3).
"""
import torch
import torch.nn as nn

from alignbeat.beat_this_encoder import BeatThisEncoder
from alignbeat.progressive_downsample import ProgressiveDownsample
from alignbeat.subset_head import (
    SubsetSelectionHead,
    SubsetCriterion,
    intervals_to_events,
)


class AlignBeat(nn.Module):
    def __init__(self, audio_downsampling_factor=512, audio_sample_rate=22050, **kwargs):
        super().__init__()

        self.audio_downsampling_factor = audio_downsampling_factor
        self.audio_sample_rate = audio_sample_rate

        encoder_keys = ['spect_dim', 'transformer_dim', 'ff_mult', 'n_layers',
                        'head_dim', 'stem_dim', 'dropout', 'partial_transformers']
        encoder_kwargs = {k: kwargs[k] for k in encoder_keys if k in kwargs}
        self.encoder = BeatThisEncoder(**encoder_kwargs)
        transformer_dim = encoder_kwargs.get('transformer_dim', 512)

        # T (the encoder's output frame count) is fixed by the training window
        # length, so it must be known here to precompute the downsample schedule
        # when building the module. N is derived from the physical tempo bound
        # N_min := BPM_max * D_min, never chosen directly (Section 3).
        self.downsample = ProgressiveDownsample(
            d_model=transformer_dim,
            T=kwargs['encoder_input_frames'],
            N_min=kwargs.get('n_min', 100),
        )
        self.num_candidates = self.downsample.N

        # No FPN: a single level, so level_strides=(1,) makes the head's own
        # downsample conv a kernel-1 projection that does not change the length.
        self.subset_head = SubsetSelectionHead(
            feature_size=transformer_dim,
            num_candidates=self.num_candidates,
            level_strides=(1,),
            hidden_size=kwargs.get('subset_hidden_size', 256),
            # Section 10.2: candidate-level self-attention, classification
            # branch only. 0 disables it entirely.
            class_attention_layers=kwargs.get('class_attention_layers', 0),
            class_attention_heads=kwargs.get('class_attention_heads', 4),
            # Section 4.1.2: per-candidate precision b_j. Off by default.
            predict_precision=kwargs.get('predict_precision', False),
        )
        self.subset_criterion = SubsetCriterion(
            b_scale=kwargs.get('b_scale', 0.005),
            gamma=kwargs.get('gamma', 0.5),
            omega_downbeat=kwargs.get('omega_db', 2.0),
            learn_b=kwargs.get('learn_b', False),
            cont_weight=kwargs.get('cont_weight', 0.0),
            lambda_r=kwargs.get('lambda_r', 0.0),
            meter_length=kwargs.get('meter_length', 0),
            marginal=kwargs.get('marginal', False),
            marginal_background=kwargs.get('marginal_background', True),
            mu_meter=kwargs.get('mu_meter', 0.0),
            cont_windows=kwargs.get('cont_windows', 8),
            normalize_by_events=not kwargs.get('no_event_norm', False),
            # Sections 8.3 / 8.5: infer (sigma, phi_0) together rather than in sequence.
            joint_phase=kwargs.get('joint_phase', False),
            # Section 8.6: marginalize the meter too.
            marginal_meters=kwargs.get('marginal_meters', ()),
            precision_warmup=kwargs.get('precision_warmup', 2000),
            precision_prior_alpha=kwargs.get('precision_prior_alpha', 2.0),
        )

    def forward(self, inputs):
        # Dispatch on type, not on len(): a bare tensor of batch size 2 is not a
        # (audio, annotations) pair, and treating it as one silently corrupts the
        # eval path.
        if isinstance(inputs, (tuple, list)):
            audio_batch, annotations = inputs
        else:
            audio_batch, annotations = inputs, None

        encoder_out = self.encoder(audio_batch)          # (B, T, transformer_dim)
        num_frames = encoder_out.shape[1]                # T, the axis annotations are indexed on
        z = self.downsample(encoder_out)                 # (B, N, transformer_dim)

        # SubsetSelectionHead expects a list of channel-first (B, C, T_l) maps;
        # length 1 here since there is no FPN.
        head_out = self.subset_head([z.transpose(1, 2)])
        # The head returns a third output only when the section 4.1.2 precision head
        # is enabled; inference never needs it.
        class_logits, t_hat = head_out[0], head_out[1]
        raw_precision = head_out[2] if len(head_out) > 2 else None

        if not self.training:
            return class_logits, t_hat

        # Annotation frame indices are on the crop's own axis, the same unit as
        # T, so normalizing by T puts them on eq. (1)'s [0, 1] axis.
        targets = intervals_to_events(annotations, num_frames)

        # The loss math stays in float32 even under --amp. autocast's op allowlist would
        # cover log_softmax, but the DP is the part that matters: the cost matrix feeds a
        # running minimum and a log-sum-exp recursion whose numerics are load-bearing
        # (the LOG_PROB_FLOOR clamp in subset_head exists precisely because a -inf cost
        # makes backtracking impossible and loses the batch). Casting explicitly is
        # cheaper to reason about than trusting the allowlist to cover every op the
        # recursions use, including logcumsumexp.
        with torch.autocast('cuda', enabled=False):
            losses, _stats = self.subset_criterion(
                class_logits.float(), t_hat.float(), targets,
                None if raw_precision is None else raw_precision.float())

        # train.py sums the returned tuple and backpropagates through it, so
        # every term that should reach the optimizer must be in here.
        return (
            losses['class'].unsqueeze(0),
            losses['time'].unsqueeze(0),
            losses['background'].unsqueeze(0),
            losses['continuity'].unsqueeze(0),
            losses['periodicity'].unsqueeze(0),
        )


def create_alignbeat_model(args, **kwargs):
    return AlignBeat(**kwargs)
