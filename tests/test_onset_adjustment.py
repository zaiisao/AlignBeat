import librosa
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.integration.config import split_subset_kwargs
from alignbeat.model.head import (
    calibrate_event_logits,
    librosa_logmel_onset_strength,
    sample_onset_strength,
)
from beat_this.model.beat_tracker import BeatThis


def test_torch_onset_strength_matches_librosa_on_stored_logmel():
    rng = np.random.default_rng(12)
    spectrogram = torch.from_numpy(
        rng.uniform(0, 4, size=(2, 151, 128)).astype(np.float32)
    )
    actual = librosa_logmel_onset_strength(spectrogram).numpy()
    expected = np.stack([
        librosa.onset.onset_strength(
            S=item.T, sr=22050, hop_length=441, n_fft=1024, center=True
        )
        for item in spectrogram.numpy()
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_identity_calibrator_preserves_three_way_probabilities():
    torch.manual_seed(4)
    logits = torch.randn(2, 17, 3)
    onset = torch.rand(2, 17)
    calibrator = nn.Linear(2, 1)
    nn.init.zeros_(calibrator.weight)
    nn.init.zeros_(calibrator.bias)
    with torch.no_grad():
        calibrator.weight[0, 0] = 1

    adjusted = calibrate_event_logits(logits, onset, calibrator)
    torch.testing.assert_close(F.softmax(adjusted, dim=-1),
                               F.softmax(logits, dim=-1), rtol=1e-6, atol=1e-7)


def test_onset_sampling_and_calibrator_have_finite_gradients():
    onset = torch.zeros(2, 101)
    onset[:, 50] = 4
    times = torch.tensor([[0.5, 0.2], [0.5, 0.8]])
    sampled = sample_onset_strength(onset, times)
    assert sampled.shape == times.shape
    assert torch.isfinite(sampled).all()
    assert torch.all(sampled[:, 0] > sampled[:, 1])

    logits = torch.randn(2, 2, 3, requires_grad=True)
    calibrator = nn.Linear(2, 1)
    adjusted = calibrate_event_logits(logits, sampled, calibrator)
    F.cross_entropy(adjusted.reshape(-1, 3), torch.tensor([0, 2, 1, 2])).backward()
    assert calibrator.weight.grad is not None
    assert torch.isfinite(calibrator.weight.grad).all()
    assert calibrator.weight.grad.abs().sum() > 0


def test_onset_adjustment_option_routes_to_alignbeat_head():
    head, _decode, _criterion = split_subset_kwargs({
        "num_candidates": 8,
        "onset_adjustment": True,
    })
    assert head["onset_adjustment"] is True


@pytest.mark.slow
def test_compiled_alignbeat_head_keeps_onset_input_and_backpropagates():
    model = BeatThis(
        spect_dim=128, transformer_dim=64, ff_mult=2, n_layers=1,
        stem_dim=8, head_dim=8, partial_transformers=False,
        alignbeat=True,
        alignbeat_kwargs={
            "num_candidates": 8,
            "train_length": 64,
            "downsample_stages": 3,
            "onset_adjustment": True,
        },
    )
    model.task_heads = torch.compile(model.task_heads)
    output = model(torch.randn(2, 64, 128))
    loss = output["class_logits"].square().mean() + output["t_hat"].square().mean()
    loss.backward()
    original_head = model.task_heads._orig_mod
    assert original_head.event_calibrator.weight.grad is not None
    assert torch.isfinite(original_head.event_calibrator.weight.grad).all()
