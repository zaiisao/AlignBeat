CLASS_UNKNOWN = -1
CLASS_DOWNBEAT = 0
CLASS_BEAT = 1
CLASS_BACKGROUND = 2

NUM_CLASSES = 3

# Standard beat-tracking F-measure tolerance, in seconds (Davies et al., 2009).
F_MEASURE_TOLERANCE = 0.07

# Continuity tolerance is a relative radius (fraction of the beat interval), unlike
# the absolute F-measure tolerance. They coincide at 150 BPM (0.175 * 0.4 s = 0.07 s).
CONTINUITY_TOLERANCE = 0.175

# Upper tempo bound used to size the candidate set. Oversizing adds background
# candidates; undersizing can make a fragment infeasible when N is less than M.
BPM_MAX = 340.0
