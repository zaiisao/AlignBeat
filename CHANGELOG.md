# Changelog

## [1.0.0] - 2026-11-13

First AlignBeat release, built on the Beat This! beat-tracking framework.

### Added

- Candidate event head with classification and event-time regression.
- Order-preserving dynamic-programming alignment and training losses for beat,
  downbeat, and partially labeled beat-only annotations.
- AlignBeat decoding, fragment stitching, and training and evaluation integration.
- Training options for candidate attention, latent meter, tempo-relative timing,
  onset adjustment, and downbeat annotation dropout.

### Reused from Beat This!

- Spectrogram frontend, transformer blocks, data pipeline, augmentation, and
  beat-tracking evaluation metrics.
