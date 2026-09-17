#!/usr/bin/env python3
"""Per-dataset results as two LaTeX tables, one for beat and one for downbeat.

Both carry all three metrics for both systems. No delta column: with three metric
pairs a single delta would have to belong to one of them arbitrarily, and bolding
already shows which system wins. Rows are sorted by CMLt gain.

The GTZAN row is a different experiment from the rest -- models trained on all
cross-validation data, evaluated on the held-out test set, averaged over three
seeds -- so it sits below the pooled row and the caption says so.
"""
import csv, os, re, sys
from collections import defaultdict
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "results")
KINDS = {"beat": ("F-measure_beat", "CMLt_beat", "AMLt_beat"),
         "downbeat": ("F-measure_downbeat", "CMLt_downbeat", "AMLt_downbeat")}
NO_DOWNBEAT = {"simac", "smc"}
SHORT = {"rwc_royalty-free": "rwc_royalty", "rwc_classical": "rwc_class."}
esc = lambda s: SHORT.get(s, s).replace("_", r"\_")

# GTZAN, read off the scoring logs: ours is FX_all{,_s1,_s2}, theirs beat_this-final{0,1,2}
def gtzan(keys, pats):
    out = []
    for k in keys:
        vals = []
        for p in pats:
            t = open(p, errors="ignore").read()
            vals.append(float(re.search(rf"^{re.escape(k)}: ([0-9.]+)", t, re.M).group(1)) * 100)
        out.append(np.mean(vals))
    return out

def build(kind):
    keys = KINDS[kind]
    o = list(csv.DictReader(open(f"{OUT}/cv_ours.csv")))
    b = list(csv.DictReader(open(f"{OUT}/cv_bt.csv")))
    acc = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for ro, rb in zip(o, b):
        if kind == "downbeat" and ro["dataset"] in NO_DOWNBEAT:
            continue
        for k in keys:
            if ro[k] in ("", "nan"): continue
            acc[ro["dataset"]][k][0].append(float(ro[k]))
            acc[ro["dataset"]][k][1].append(float(rb[k]))

    def m(d, k):
        return np.mean(d[k][1]) * 100, np.mean(d[k][0]) * 100

    rows = []
    for ds, d in acc.items():
        cells = [m(d, k) for k in keys]
        gain = cells[1][1] - cells[1][0]
        rows.append((esc(ds), len(d[keys[0]][0]), cells, gain))
    rows.sort(key=lambda r: -r[3])

    pool = defaultdict(lambda: ([], []))
    for ds, d in acc.items():
        for k in d:
            pool[k][0].extend(d[k][0]); pool[k][1].extend(d[k][1])
    rows.append((r"\hline All", len(pool[keys[0]][0]), [m(pool, k) for k in keys], 0))

    g_ours = gtzan(keys, [f"/tmp/gtzan_{n}.log" for n in ("FX_all", "FX_all_s1", "FX_all_s2")])
    g_bt = gtzan(keys, [f"/tmp/gtzan_bt_final{i}.log" for i in range(3)])
    rows.append((r"\hline GTZAN", 993, list(zip(g_bt, g_ours)), 0))
    return rows

def pair(t, u):
    a, c = f"{t:.1f}", f"{u:.1f}"
    if abs(u - t) < 0.05: return a, c
    return (f"\\textbf{{{a}}}", c) if t > u else (a, f"\\textbf{{{c}}}")

for kind, num in (("downbeat", "I"), ("beat", "II")):
    note = ("\\texttt{simac} and \\texttt{smc} are omitted: they carry no downbeat annotations."
            if kind == "downbeat" else "All $18$ datasets carry beat annotations.")
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(f"  \\caption{{{kind.capitalize()} results per dataset, sorted by CMLt gain. {note}"
          "\n  The GTZAN row evaluates models trained on all cross-validation data.}")
    print(f"  \\label{{tab:perdataset{kind}}}")
    print(r"  \footnotesize")
    print(r"  \setlength{\tabcolsep}{3.5pt}")
    print(r"  \begin{tabular}{@{}lr rr rr rr@{}}")
    print(r"    \hline")
    print(r"    & & \multicolumn{2}{c}{F-measure} & \multicolumn{2}{c}{CMLt}"
          r" & \multicolumn{2}{c}{AMLt} \\")
    print(r"    \cline{3-4}\cline{5-6}\cline{7-8}")
    print(r"    Dataset & $n$ & BT & Ours & BT & Ours & BT & Ours \\")
    print(r"    \hline")
    for name, n, cells, _ in build(kind):
        cs = " & ".join(" & ".join(pair(t, u)) for t, u in cells)
        print(f"    {name} & {n} & {cs} \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()
