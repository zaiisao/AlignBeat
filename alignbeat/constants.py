CLASS_UNKNOWN = -1
CLASS_DOWNBEAT = 0
CLASS_BEAT = 1
CLASS_BACKGROUND = 2

NUM_CLASSES = 3

# The standard beat-tracking F-measure tolerance, in SECONDS
# References:
#   [1] "Evaluation Methods for Musical Audio Beat Tracking Algorithms" by Davies et al., 2009
F_MEASURE_TOLERANCE = 0.07

# Fastest tempo the corpus contains, in BPM. Only an upper bound is needed:
# overshooting costs a few extra candidates that get classified as background, which
# the formulation expects anyway, while undershooting is unrecoverable -- an
# order-preserving injection needs N >= M, and SubsetCriterion skips any fragment where
# it does not hold. 15 of 5555 tracks exceed this, all of them asap, whose MIDI-derived
# annotations are note-level rather than beat-level (the densest has a 0.005 s gap,
# i.e. 11521 BPM); those are bad annotations, not fast music.
BPM_MAX = 340.0
