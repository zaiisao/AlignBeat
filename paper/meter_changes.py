"""How often does the meter change inside a single piece?

Reads every two-column .beats annotation (time, beat-within-bar), takes the gaps
between consecutive downbeats as bar lengths, and counts the pieces that use more
than one. Writes paper/meter_changes.pdf.
"""
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANNOTATIONS = Path("data/annotations")


def bar_lengths(path):
    """Bar lengths in beats, or None if the file carries no downbeats."""
    beats = np.loadtxt(path, ndmin=2)
    if beats.shape[1] < 2:
        return None
    downbeats = np.flatnonzero(beats[:, 1] == 1)
    if downbeats.size < 4:
        return None
    # Drop the outer two bars: a pickup and a truncated final bar are not meter changes.
    return np.diff(downbeats)[1:-1]


def main():
    by_dataset = collections.defaultdict(list)
    for path in sorted(ANNOTATIONS.glob("*/annotations/beats/*.beats")):
        lengths = bar_lengths(path)
        if lengths is not None and lengths.size:
            by_dataset[path.parts[-4]].append(lengths)

    rows = []
    for dataset, pieces in by_dataset.items():
        varying = [p for p in pieces if len(set(p.tolist())) > 1]
        off_modal = [np.mean(p != np.bincount(p).argmax()) * 100 for p in varying]
        rows.append((dataset, len(pieces), 100 * len(varying) / len(pieces),
                     np.median(off_modal) if off_modal else 0.0))
    rows.sort(key=lambda r: r[2])

    print(f"{'dataset':16s} {'pieces':>7s} {'varying':>9s} {'median bars off modal':>22s}")
    for dataset, n, pct, off in rows:
        print(f"{dataset:16s} {n:7d} {pct:8.1f}% {off:21.1f}%")

    fig, ax = plt.subplots(figsize=(7, 0.34 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], height=0.6, color="#4C72B0")
    ax.set_yticks(y, [f"{r[0]}  (n={r[1]})" for r in rows])
    for i, (_, _, pct, _) in enumerate(rows):
        ax.text(pct + 1, i, f"{pct:.0f}%", va="center", fontsize=8, color="#444")
    ax.set_xlim(0, 100)
    ax.set_xlabel("pieces whose bar length is not constant (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color="#DDD", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("paper/meter_changes.pdf")
    fig.savefig("paper/meter_changes.png", dpi=150)
    print("wrote paper/meter_changes.pdf and .png")


if __name__ == "__main__":
    main()
