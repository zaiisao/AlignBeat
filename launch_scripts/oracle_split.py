"""Split the classifier's oracle gap into its parts, on fold 0, one checkpoint.

oracle_ceiling.py replaces the WHOLE classifier and reports what that buys. Here each
decode replaces one piece at a time, all over the same predicted t_hat:

  oracle   time-only DP match -> matched candidates fire with their true class,
           everything else is background. The full ceiling.
  orfire   matched candidates ALL fire (oracle fire + oracle suppression), but the
           DB-vs-B choice is the model's own. What remains is the D stage.
  orsup    the model's real decode, with every fired candidate that is NOT a matched
           one removed. Oracle suppression only: what remains is the model's misses.
  orcls    the model's real decode, with every fired matched candidate given its true
           class. Oracle DB-vs-B only: beat F equals real by construction, so the
           downbeat column is the one to read.
  real     the model's own decode.

Plus a matching comparison: the training E-step matches on class + time (eq. 3 with
the section 8.4 correction); the diagnostics match on time alone. How often do they
pick different candidates, and does the classifier's choice look self-serving?
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import BACKGROUND, BEAT, DOWNBEAT
from alignbeat.dp import subset_select_dp
from launch_scripts.oracle_ceiling import load


@torch.no_grad()
def run(model, loader, device):
    from alignbeat.decode import decode_events
    crit = model.subset_criterion
    rows = []
    agg = dict(events=0, differ=0, differ_and_bg=0, differ_full_in_tol=0,
               c_argmax_time=0, c_argmax_full=0, c_thresh_time=0,
               fired_total=0, fired_matched=0, fired_in_tol=0, events_hit=0,
               db_events=0, db_fired=0, db_as_b=0, b_events=0, b_fired=0, b_as_db=0,
               gap_class=0.0, gap_time=0.0, gap_n=0, b_ms=float("nan"))
    TOL_S = 0.070
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        n_frames = batch["truth_beat"].shape[-1]
        window = n_frames / model.fps
        targets = model._subset_targets(batch)

        for i, target in enumerate(targets):
            truth_b = np.frombuffer(batch["truth_orig_beat"][i])
            if len(truth_b) < 3:
                continue
            logits = pred["class_logits"][i]
            t_hat_t = pred["t_hat"][i]
            t_hat = t_hat_t.cpu().numpy()
            gt_t = target["times"].cpu().numpy()
            gt_c = target["classes"].cpu().numpy()
            M, N = len(gt_t), len(t_hat)
            if M < 2 or M > N:
                continue

            # --- matchings -------------------------------------------------------
            sigma_time = subset_select_dp(np.abs(gt_t[:, None] - t_hat[None, :]))
            logp = F.log_softmax(logits, dim=-1)
            full_cost = crit.build_l_match(logp, t_hat_t, target["classes"], target["times"])
            time_cost = crit.l1(t_hat_t[None, :], target["times"][:, None])
            # build_l_match is the two terms added; the class channel is what is left.
            class_cost = full_cost - time_cost
            sigma_full = subset_select_dp(full_cost.cpu().numpy())

            argmax = logits.argmax(-1).cpu().numpy()
            probs = F.softmax(logits, dim=-1)
            score, _ = probs.max(-1)
            score = score.cpu().numpy()
            fires_thresh = (argmax != BACKGROUND) & (
                score >= model.tau)

            differ = sigma_time != sigma_full
            if differ.any():
                ev = np.nonzero(differ)[0]
                cc = class_cost.cpu().numpy(); tc = time_cost.cpu().numpy()
                bg = (-logp[:, BACKGROUND]).cpu().numpy()
                # how much class (incl. the 8.4 correction) the full pick saves, and how
                # much time it pays, relative to the nearest candidate
                agg["gap_class"] += float(((cc[ev, sigma_time[ev]] - crit.gamma * bg[sigma_time[ev]])
                                           - (cc[ev, sigma_full[ev]] - crit.gamma * bg[sigma_full[ev]])).sum())
                agg["gap_time"] += float((tc[ev, sigma_full[ev]] - tc[ev, sigma_time[ev]]).sum())
                agg["gap_n"] += len(ev)
            tol = TOL_S / window
            agg["differ_full_in_tol"] += int((differ & (
                np.abs(gt_t - t_hat[sigma_full]) <= tol)).sum())
            agg["events"] += M
            agg["differ"] += int(differ.sum())
            # classifier steering: time-only pick is background, full-cost pick is not
            agg["differ_and_bg"] += int((differ & (argmax[sigma_time] == BACKGROUND)
                                         & (argmax[sigma_full] != BACKGROUND)).sum())
            agg["c_argmax_time"] += int((argmax[sigma_time] != BACKGROUND).sum())
            agg["c_argmax_full"] += int((argmax[sigma_full] != BACKGROUND).sum())
            agg["c_thresh_time"] += int(fires_thresh[sigma_time].sum())

            # --- decodes ---------------------------------------------------------
            matched = np.zeros(N, dtype=bool); matched[sigma_time] = True
            true_class = np.full(N, BEAT); true_class[sigma_time] = np.where(
                gt_c == DOWNBEAT, DOWNBEAT, BEAT)
            model_db = (logp[:, DOWNBEAT] > logp[:, BEAT]).cpu().numpy()

            cls, times, _ = decode_events(logits, t_hat_t, model.tau)
            real_sec = (times * window).cpu().numpy()
            real_db = (cls == DOWNBEAT).cpu().numpy()
            fired = np.zeros(N, dtype=bool)
            fired[np.isin(t_hat, times.cpu().numpy())] = True
            agg["fired_total"] += int(fired.sum())
            agg["fired_matched"] += int((fired & matched).sum())
            # the metric's own view: any fired candidate within 70 ms of an event
            dist = np.abs(gt_t[:, None] - t_hat[None, :])
            agg["events_hit"] += int((dist[:, fired] <= tol).any(axis=1).sum())
            agg["fired_in_tol"] += int((dist[:, fired] <= tol).any(axis=0).sum())
            # DB-vs-B confusion on fired matched candidates
            fm = fired[sigma_time]
            is_db, is_b = gt_c == DOWNBEAT, gt_c == BEAT
            agg["db_events"] += int(is_db.sum()); agg["b_events"] += int(is_b.sum())
            agg["db_fired"] += int((is_db & fm).sum()); agg["b_fired"] += int((is_b & fm).sum())
            agg["db_as_b"] += int((is_db & fm & ~model_db[sigma_time]).sum())
            agg["b_as_db"] += int((is_b & fm & model_db[sigma_time]).sum())

            sec = t_hat * window
            decodes = {
                "oracle": (matched, true_class[matched] == DOWNBEAT),
                "orfire": (matched, model_db[matched]),
                "orsup":  (fired & matched, model_db[fired & matched]),
                "orcls":  (fired, np.where(matched[fired], true_class[fired] == DOWNBEAT,
                                           model_db[fired])),
                "real":   (None, None),
            }

            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            has_db = bool(batch["downbeat_mask"][i]) and len(truth_db) >= 3
            row = dict(corpus=str(batch["spect_path"][i]).split("/", 1)[0],
                       bpm=float(60.0 / np.median(np.diff(truth_b))))
            for tag, (keep, isdb) in decodes.items():
                if keep is None:
                    s, d = real_sec, real_db
                else:
                    s, d = sec[keep], isdb
                row[f"{tag}_F"] = model.metrics(truth_b, np.sort(s), step="test")["F-measure"]
                if has_db:
                    row[f"{tag}_dbF"] = model.metrics(
                        truth_db, np.sort(s[d]), step="test")["F-measure"]
            rows.append(row)
    return rows, agg


TAGS = ("oracle", "orfire", "orsup", "orcls", "real")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=args.num_workers, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    rows, agg = run(load(sorted(glob.glob(args.checkpoint))[0], device),
                    dm.val_dataloader(), device)

    def show(name, sel):
        if not sel: return
        db = [r for r in sel if "oracle_dbF" in r]
        cells = [f"{np.mean([r[f'{t}_F'] for r in sel]):7.3f}" for t in TAGS]
        dbc = [f"{np.mean([r[f'{t}_dbF'] for r in db]):7.3f}" if db else "    nan"
               for t in TAGS]
        print(f"  {name:<12}{len(sel):>5} |" + "".join(cells) + " |" + "".join(dbc))

    head = "".join(f"{t:>7}" for t in TAGS)
    print(f"\n{'':<12}{'n':>5} | beat F {head[7:]} | downbeat F {head[11:]}")
    print(f"  {'group':<12}{'':>5} |{head} |{head}")
    show("ALL", rows)
    for lo, hi in ((0,70),(70,100),(100,130),(130,160),(160,1e9)):
        show(f"bpm {lo}-{hi:.0f}" if hi < 1e9 else f"bpm {lo}+",
             [r for r in rows if lo <= r["bpm"] < hi])
    for c in sorted({r["corpus"] for r in rows}):
        show(f"ds:{c}", [r for r in rows if r["corpus"] == c])

    E = agg["events"]
    print(f"\nmatching, {E} events")
    print(f"  full-cost vs time-only pick differs      {agg['differ']/E:6.1%}")
    print(f"    ...and time pick is BG, full pick is not {agg['differ_and_bg']/E:6.1%}")
    print(f"  C (argmax fires) under time-only match   {agg['c_argmax_time']/E:6.1%}")
    print(f"  C (argmax fires) under full-cost match   {agg['c_argmax_full']/E:6.1%}")
    print(f"  C (decode fires) under time-only match   {agg['c_thresh_time']/E:6.1%}")
    print(f"    ...of which full pick is within 70 ms    {agg['differ_full_in_tol']/E:6.1%}")
    gn = max(agg['gap_n'], 1)
    print(f"  on differing events: class saved {agg['gap_class']/gn:5.2f} nats, "
          f"time paid {agg['gap_time']/gn:5.2f} nats  (E-step b = {agg['b_ms']:.0f} ms)")
    ft = max(agg['fired_total'], 1)
    print(f"\ndecode, metric's view (70 ms)")
    print(f"  recall: events with a fired candidate      {agg['events_hit']/E:6.1%}")
    print(f"  precision: fired candidates near an event  {agg['fired_in_tol']/ft:6.1%}")
    print(f"  fired {agg['fired_total']}, {agg['fired_matched']/ft:.1%} are the time-only pick")
    print(f"\nDB-vs-B on fired matched candidates")
    print(f"  downbeats: {agg['db_events']} events, {agg['db_fired']} fired, "
          f"{agg['db_as_b']/max(agg['db_fired'],1):.1%} of those called B")
    print(f"  beats:     {agg['b_events']} events, {agg['b_fired']} fired, "
          f"{agg['b_as_db']/max(agg['b_fired'],1):.1%} of those called DB")


if __name__ == "__main__":
    main()
