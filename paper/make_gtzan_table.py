#!/usr/bin/env python3
"""GTZAN table: ours against Beat This and the concurrent masked-diffusion system.

Beat This is scored by us (beat_this-final0/1/2); its figures match those quoted in
[foscarin2026masked] to within 0.07 on every metric. The masked-diffusion row is
quoted from that paper, including its two coherency heuristics.
"""
import re, numpy as np

KEYS = [("F-measure_beat", "bF"), ("CMLt_beat", "bCMLt"), ("AMLt_beat", "bAMLt"),
        ("F-measure_downbeat", "dF"), ("CMLt_downbeat", "dCMLt"), ("AMLt_downbeat", "dAMLt")]

def metrics(paths):
    out = {}
    for k, _ in KEYS:
        v = [float(re.search(rf"^{re.escape(k)}: ([0-9.]+)", open(p, errors="ignore").read(),
                             re.M).group(1)) * 100 for p in paths]
        out[k] = np.mean(v)
    return out

def coherency(paths):
    """consecutive downbeats and tempo doubling per track, from the diagnostic logs"""
    cd, td = [], []
    for p in paths:
        for line in open(p, errors="ignore"):
            if "ours (detect)" in line:
                f = line.split()
                cd.append(float(f[-3])); td.append(float(f[-2]))
    return np.mean(cd), np.mean(td)

ours = metrics([f"/tmp/gtzan_{n}.log" for n in ("FX_all", "FX_all_s1", "FX_all_s2")])
bt = metrics([f"/tmp/gtzan_bt_final{i}.log" for i in range(3)])
o_cd, o_td = coherency([f"/tmp/coh_fx{i}.log" for i in range(3)])
# Beat This measured by us on the same 993 tracks; MDM from their Section 4.4.
bt_cd, bt_td = 1.713, 0.976
mdm = {"F-measure_beat": 89.7, "CMLt_beat": 82.9, "AMLt_beat": 92.5,
       "F-measure_downbeat": 79.5, "CMLt_downbeat": 76.4, "AMLt_downbeat": 88.5}
mdm_cd, mdm_td = 0.02, 0.119

rows = [("Beat This~\\cite{foscarin2024beat}", bt, bt_cd, bt_td),
        ("MDM~\\cite{foscarin2026masked}", mdm, mdm_cd, mdm_td),
        ("Ours", ours, o_cd, o_td)]

# bold the best in each column; coherency columns are counts, so lower is better
best_m = {k: max(r[1][k] for r in rows) for k, _ in KEYS}
best_cd, best_td = min(r[2] for r in rows), min(r[3] for r in rows)
fmt = lambda v, b, d=1: (f"\\textbf{{{v:.{d}f}}}" if abs(v - b) < 10**-d/2 else f"{v:.{d}f}")

print(r"\begin{table}[t]")
print(r"  \centering")
print(r"  \caption{GTZAN, $993$ tracks. Coherency is the per-track count of consecutive"
      "\n  downbeats and of tempo doubling or halving events, lower being better; the GTZAN"
      "\n  annotations contain $0.000$ and $0.036$ respectively. Beat This is scored"
      "\n  by us and reproduces the figures quoted in~\\cite{foscarin2026masked}; the"
      "\n  masked-diffusion row (MDM) is quoted from that paper, where it is obtained"
      "\n  from an ensemble of three models at eight inference steps.}")
print(r"  \label{tab:gtzan}")
print(r"  \footnotesize")
print(r"  \setlength{\tabcolsep}{3pt}")
print(r"  \begin{tabular}{@{}l rr rrr rrr@{}}")
print(r"    \hline")
print(r"    & \multicolumn{2}{c}{Coherency} & \multicolumn{3}{c}{Beat}"
      r" & \multicolumn{3}{c}{Downbeat} \\")
print(r"    \cline{2-3}\cline{4-6}\cline{7-9}")
print(r"    & \makecell{consecutive\\downbeats} & \makecell{tempo\\$\times2$}"
      r" & F & CMLt & AMLt & F & CMLt & AMLt \\")
print(r"    \hline")
for name, m, cd, td in rows:
    cells = " & ".join(fmt(m[k], best_m[k]) for k, _ in KEYS)
    print(f"    {name} & {fmt(cd, best_cd, 2)} & {fmt(td, best_td, 2)} & {cells} \\\\")
print(r"    \hline")
print(r"  \end{tabular}")
print(r"\end{table}")
