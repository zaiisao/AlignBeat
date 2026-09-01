"""Class labels, the beat tolerance, and the shared log-probability floor."""

DOWNBEAT = 0
BEAT = 1
BACKGROUND = 2
NUM_CLASSES = 3

# Beat-only annotations: "some beat occurred", B/DB unresolved (the B* label).
CLASS_UNKNOWN = -1

# The standard beat-tracking F-measure tolerance, in SECONDS. Everything downstream --
# the precision head's initial scale, the loss's dead zone -- is this same quantity,
# converted into whatever units that consumer works in. Nothing should hardcode it twice.
F_MEASURE_TOLERANCE = 0.07
