"""Class labels, the beat tolerance, and the shared log-probability floor."""

DOWNBEAT = 0
BEAT = 1
BACKGROUND = 2
NUM_CLASSES = 3

# Beat-only annotations: "some beat occurred", B/DB unresolved (the B* label).
CLASS_UNKNOWN = -1

# The standard beat-tracking F-measure tolerance, in SECONDS. Everything downstream --
# the loss's dead zone -- is this same quantity,
# converted into whatever units that consumer works in. Nothing should hardcode it twice.
F_MEASURE_TOLERANCE = 0.07

# Meter distribution of the corpus, from docs/METER_DISTRIBUTION.md ("% of tracks with
# downbeats", i.e. P(L | L was annotated) -- which is what the latent-meter posterior is
# estimating for the beat-only sets). Renormalised over whichever candidates are offered.
METER_PRIOR = {2: 0.0447, 3: 0.0838, 4: 0.8612, 5: 0.0011, 6: 0.0068, 8: 0.0017}
