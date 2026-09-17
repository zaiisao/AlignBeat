#!/usr/bin/env python3
"""Regenerate every number in the paper's tables and figures from checkpoints.

Two stages. `score` runs the models and writes per-piece metric CSVs; `aggregate`
reads those CSVs and prints the tables. They are separate because scoring needs a GPU and
takes hours, while aggregation is seconds and gets rerun whenever a number changes.

    python paper/make_tables.py score     --gpu 0
    python paper/make_tables.py aggregate

Checkpoints expected in checkpoints/ (see CV_RUNS and GTZAN_RUNS below). Beat This's
released per-fold checkpoints are downloaded by beat_this.inference on first use.
"""
import argparse, csv, glob, os, subprocess, sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "results")
PY_EXE = sys.executable
BT_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints")

# Our runs. CV_RUNS are the eight cross-validation folds (encoder initialised from
# Beat This's fold checkpoint, 20 epochs); GTZAN_RUNS are the three all-data seeds.
CV_RUNS = [f"RX_f{k} S0 fold{k} " for k in range(8)]
GTZAN_RUNS = ["FX_all S0 noval ", "FX_all_s1 S1 noval ", "FX_all_s2 S2 noval "]
EPOCH = "epoch=019"

CLASS_BEAT = ["F-measure_beat", "CMLt_beat", "AMLt_beat"]
DOWN = ["F-measure_downbeat", "CMLt_downbeat", "AMLt_downbeat"]
PRETTY = {"F-measure_beat": "F-measure", "CMLt_beat": "CMLt", "AMLt_beat": "AMLt",
          "F-measure_downbeat": "F-measure", "CMLt_downbeat": "CMLt",
          "AMLt_downbeat": "AMLt"}
# No downbeat annotations, so their downbeat cells are structural zeros rather than
# scores. They are dropped from downbeat rows and kept in beat rows.
NO_DOWNBEAT = {"simac", "smc"}


def find(prefix):
    hits = glob.glob(os.path.join(REPO, "checkpoints", f"{prefix}*{EPOCH}*.ckpt"))
    if not hits:
        sys.exit(f"missing checkpoint: {prefix}*{EPOCH}")
    return hits[0]


def score(gpu):
    """Run every model and dump per-piece metrics. Ours decode with Algorithm 3's
    detection stage; Beat This decodes with its own postprocessor. Both go through
    compute_paper_metrics, so the protocol (full pieces, 5 s trim, no DBN) is shared."""
    os.makedirs(OUT, exist_ok=True)
    shim = os.path.join(OUT, "_decode_shim.py")
    with open(shim, "w") as fh:
        fh.write(
            "import sys, runpy\n"
            f"sys.path.insert(0, {REPO!r})\n"
            "from beat_this.model.pl_module import PLBeatThis\n"
            "# these checkpoints predate --decode, so select it at load time\n"
            "PLBeatThis.decode = property(lambda s: 'detect', lambda s, v: None)\n"
            "PLBeatThis.detect_tau = property(lambda s: 0.5, lambda s, v: None)\n"
            "sys.argv = ['compute_paper_metrics.py'] + sys.argv[1:]\n"
            f"runpy.run_path({os.path.join(REPO, 'launch_scripts', 'compute_paper_metrics.py')!r},"
            " run_name='__main__')\n")

    def run(entry, models, split, dump, log):
        cmd = [PY_EXE, "-u", entry, "--models", *models, "--datasplit", split,
               "--aggregation-type", "k-fold", "--dump-piece-metrics", dump,
               "--gpu", str(gpu), "--num_workers", "2"]
        print(f"  -> {os.path.basename(dump)}", flush=True)
        with open(log, "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=REPO, env={**os.environ, "PYTHONPATH": REPO})

    plain = os.path.join(REPO, "launch_scripts", "compute_paper_metrics.py")
    run(shim, [find(p) for p in CV_RUNS], "val",
        f"{OUT}/cv_ours.csv", f"{OUT}/cv_ours.log")
    run(plain, [f"{BT_CACHE}/beat_this-fold{k}.ckpt" for k in range(8)], "val",
        f"{OUT}/cv_bt.csv", f"{OUT}/cv_bt.log")
    for i, prefix in enumerate(GTZAN_RUNS):
        run(shim, [find(prefix)], "test",
            f"{OUT}/gtzan_s{i}.csv", f"{OUT}/gtzan_s{i}.log")


def load(path, drop_no_downbeat_rows=True):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"empty: {path}")
    return rows


def paired(ours, theirs, key, exclude=()):
    a, b = [], []
    for ro, rb in zip(ours, theirs):
        if ro["piece"] != rb["piece"]:
            sys.exit("piece order differs between the two CSVs")
        if ro["dataset"] in exclude:
            continue
        if ro[key] in ("", "nan") or rb[key] in ("", "nan"):
            continue
        a.append(float(ro[key])); b.append(float(rb[key]))
    a, b = np.array(a) * 100, np.array(b) * 100
    d = a - b
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return a.mean(), b.mean(), d.mean(), ci, len(d)


def cross_validation(ours, bt):
    print("CROSS-VALIDATION  8 folds, ours vs Beat This")
    print(f"  {'':<10}{'metric':<11}{'BeatThis':>9}{'ours':>8}{'delta':>8}   95% CI")
    for label, keys, excl in (("Beat", CLASS_BEAT, ()), ("Downbeat", DOWN, NO_DOWNBEAT)):
        for i, k in enumerate(keys):
            o, b, d, ci, n = paired(ours, bt, k, excl)
            head = label if i == 0 else ""
            print(f"  {head:<10}{PRETTY[k]:<11}{b:9.2f}{o:8.2f}{d:+8.2f}"
                  f"   [{d-ci:+.2f}, {d+ci:+.2f}]")
    nb = paired(ours, bt, CLASS_BEAT[0])[4]
    nd = paired(ours, bt, DOWN[0], NO_DOWNBEAT)[4]
    print(f"  beat rows n={nb}, downbeat rows n={nd}\n")


def per_dataset(ours, bt):
    """Per-dataset downbeat CMLt delta, sorted -- the data behind the bar chart."""
    per = defaultdict(lambda: ([], []))
    for ro, rb in zip(ours, bt):
        if ro["dataset"] in NO_DOWNBEAT: continue
        if ro["CMLt_downbeat"] in ("", "nan"): continue
        per[ro["dataset"]][0].append(float(ro["CMLt_downbeat"]))
        per[ro["dataset"]][1].append(float(rb["CMLt_downbeat"]))
    print("PER-DATASET  downbeat CMLt")
    print(f"  {'dataset':<18}{'n':>5}{'BeatThis':>9}{'ours':>8}{'delta':>8}{'ci95':>7}")
    rows = []
    for ds, (a, b) in per.items():
        a, b = np.array(a) * 100, np.array(b) * 100
        d = a - b
        rows.append((ds, len(d), b.mean(), a.mean(), d.mean(),
                     1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    rows.sort(key=lambda r: -r[4])
    for r in rows:
        print(f"  {r[0]:<18}{r[1]:>5}{r[2]:9.2f}{r[3]:8.2f}{r[4]:+8.2f}{r[5]:7.2f}")
    print(f"  improved on {sum(1 for r in rows if r[4] > 0)}/{len(rows)} datasets\n")


def mechanism(ours, bt):
    """AMLt - CMLt isolates metrical-level and offbeat error."""
    print("MECHANISM  AMLt - CMLt (metrical-level / offbeat error)")
    for kind, keys in (("beat", ("AMLt_beat", "CMLt_beat")),
                       ("downbeat", ("AMLt_downbeat", "CMLt_downbeat"))):
        excl = () if kind == "beat" else NO_DOWNBEAT
        go = paired(ours, bt, keys[0], excl)[0] - paired(ours, bt, keys[1], excl)[0]
        gb = paired(ours, bt, keys[0], excl)[1] - paired(ours, bt, keys[1], excl)[1]
        print(f"  pooled {kind:<9} Beat This {gb:5.2f} -> ours {go:5.2f}")
    per = defaultdict(dict)
    for rows, who in ((ours, "ours"), (bt, "bt")):
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for k in ("AMLt_beat", "CMLt_beat"):
                if r[k] not in ("", "nan"): acc[r["dataset"]][k].append(float(r[k]))
        for ds in acc:
            per[ds][who] = (np.mean(acc[ds]["AMLt_beat"]) -
                            np.mean(acc[ds]["CMLt_beat"])) * 100
    for ds in sorted(per, key=lambda d: -per[d]["bt"]):
        if per[ds]["bt"] < 5: continue
        print(f"  {ds:<16} {per[ds]['bt']:6.1f} -> {per[ds]['ours']:6.1f}")
    print()


def gtzan():
    """GTZAN. Beat This and MDM are quoted from their papers; ours is mean +- std
    over three seeds trained on all cross-validation data."""
    seeds = [load(f"{OUT}/gtzan_s{i}.csv") for i in range(3)]
    print("GTZAN  993 tracks, mean +- std over 3 seeds")
    means = {}
    for k in CLASS_BEAT + DOWN:
        per_seed = [np.mean([float(r[k]) for r in s if r[k] not in ("", "nan")]) * 100
                    for s in seeds]
        means[k] = (np.mean(per_seed), np.std(per_seed))
    print(f"  {'metric':<22}{'mean':>8}{'std':>7}")
    for k in CLASS_BEAT + DOWN:
        kind = "beat" if k.endswith("beat") and "downbeat" not in k else "downbeat"
        print(f"  {kind + ' ' + PRETTY[k]:<22}{means[k][0]:8.2f}{means[k][1]:7.2f}")
    print("  Beat This and MDM rows are quoted from their published tables.\n")


def aggregate():
    ours, bt = load(f"{OUT}/cv_ours.csv"), load(f"{OUT}/cv_bt.csv")
    cross_validation(ours, bt)
    per_dataset(ours, bt)
    mechanism(ours, bt)
    if os.path.exists(f"{OUT}/gtzan_s0.csv"):
        gtzan()
    else:
        print("GTZAN CSVs absent; run the score stage first\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("score", "aggregate", "all"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.stage in ("score", "all"):
        score(args.gpu)
    if args.stage in ("aggregate", "all"):
        aggregate()
