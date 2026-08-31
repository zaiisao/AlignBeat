"""Class labels and the shared log-probability floor."""

DOWNBEAT = 0
BEAT = 1
BACKGROUND = 2
NUM_CLASSES = 3

# Beat-only annotations: "some beat occurred", B/DB unresolved (the B* label).
CLASS_UNKNOWN = -1

# -log p is clamped here before entering the DP, so one confident wrong candidate
# cannot dominate the matching cost.
LOG_PROB_FLOOR = -60.0
