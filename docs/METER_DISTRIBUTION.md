# Meter distribution of the Beat This annotations

Computed from `data/annotations/*/annotations/beats/*.beats` (5,556 tracks across 16
datasets). Each track's meter is the **modal number of beats between successive
downbeats**, i.e. the most common gap between lines whose beat counter is `1`.
Tracks whose annotations carry no downbeat counter are reported as *unknown*.

Reproduce with:

```bash
cd data/annotations && python3 - <<'PY'
import glob, collections
tot = collections.Counter()
for f in glob.glob('*/annotations/beats/*.beats'):
    cnt = []
    for l in open(f):
        p = l.split()
        try:
            cnt.append(int(float(p[1])))
        except (IndexError, ValueError):
            cnt.append(None)
    if not cnt or any(c is None for c in cnt) or all(c == 0 for c in cnt):
        m = 'unknown'
    else:
        idx = [i for i, c in enumerate(cnt) if c == 1]
        gaps = collections.Counter(b - a for a, b in zip(idx, idx[1:]))
        m = gaps.most_common(1)[0][0] if gaps else max(cnt)
    tot[m] += 1
N = sum(tot.values())
for m, c in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"{m:>7}: {c:5d}  {100 * c / N:5.2f}%")
PY
```

## Overall

| Meter | Tracks | % of dataset | % of tracks with downbeats |
| --- | ---: | ---: | ---: |
| 4 | 4080 | 73.43% | 86.12% |
| 3 | 397 | 7.15% | 8.38% |
| 2 | 212 | 3.82% | 4.47% |
| 6 | 32 | 0.58% | 0.68% |
| 8 | 8 | 0.14% | 0.17% |
| 5 | 5 | 0.09% | 0.11% |
| 12 | 1 | 0.02% | 0.02% |
| 7 | 1 | 0.02% | 0.02% |
| 1 | 1 | 0.02% | 0.02% |
| unknown (beats only) | 819 | 14.74% | — |

The 819 *unknown* tracks are beat-only annotations: all of `simac` (595) and `smc`
(217), plus a handful in `gtzan` and `beatles`.

## Per dataset

Percentages are shares of that dataset's own tracks.

| Dataset | 4 | 3 | 2 | 6 | other | n |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| asap | 47% | 21% | 30% | – | 1% (8) | 473 |
| ballroom | 75% | 25% | – | – | – | 685 |
| beatles | 88% | 6% | 5% | 1% | 1% unknown | 180 |
| candombe | 100% | – | – | – | – | 35 |
| filosax | 100% | – | – | – | – | 48 |
| groove_midi | 96% | 1% | – | 1% | 1% (5) | 336 |
| gtzan | 93% | 5% | 1% | – | 1% unknown | 999 |
| guitarset | 100% | – | – | – | – | 180 |
| hainsworth | 92% | 5% | 3% | <1% | – | 222 |
| harmonix | 97% | 1% | – | 2% | <1% (7, 8) | 911 |
| hjdb | 100% | – | – | – | – | 235 |
| jaah | 96% | 2% | 2% | – | – | 113 |
| rwc | 65% | 13% | 19% | 2% | <1% (1, 5, 8) | 226 |
| simac | – | – | – | – | 100% unknown | 595 |
| smc | – | – | – | – | 100% unknown | 217 |
| tapcorrect | 87% | 6% | 1% | 5% | 1% (12) | 101 |

## Caveats

- One label per track: pieces with meter changes (common in `asap` and `rwc`)
  collapse to their modal meter.
- The meter here is a beats-per-bar count from the annotations, not a time
  signature — 6/8 notated in two dotted-quarter beats appears as meter 2.
- Counts are per annotation file, unweighted by track duration.
