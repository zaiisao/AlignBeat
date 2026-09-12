"""What would F be if each stage in turn were perfect? An oracle ladder on fold 0.

The A/B/C/D breakdown says which STAGE fails; this says what that failure COSTS in the
metric the paper reports. A single oracle only gives the ceiling, so instead hand the
model one true quantity at a time and watch F climb. Each rung differs from the one
above it by exactly one oracle, so the step between them is that stage's price:

  real      decode_events: the model picks its own events by threshold, then each
            candidate independently takes argmax over DB/B. What we ship.
  real+lat  the model's OWN detected events, with downbeats from the bar-constrained
            posterior instead of the per-candidate argmax. No ground truth of any kind
            enters. This is what Algorithm 3 would actually buy; the "+latent" rung
            below is the same idea handed an oracle event set AND oracle event times,
            so it bounds this from above rather than predicting it.
  +detect   the event SET is oracled -- the time-only DP says which candidates are real
            events -- but classes still come from the head's argmax. The step from
            "real" is what detection and thresholding cost.
  +latent   same events, but downbeats come from r_i, the bar-constrained posterior,
            with L inferred. The step is what the periodicity constraint buys.
  +meter    same, with L pinned to the annotated meter. The step is what our meter
            inference costs us by getting L wrong.
  +class    every matched candidate gets its true label. The step is the head's
            remaining error, and the level is the ceiling candidate placement allows.

Beat F is identical from +detect down, since those rungs share an event set and differ
only in labelling; read the beat column for the detection price and the downbeat column
for everything after it. Timing is never oracled -- t_hat is the model's throughout --
so even the top rung carries whatever localisation error remains.
"""
import argparse, glob, math, os, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import BACKGROUND, CLASS_UNKNOWN, DOWNBEAT
from alignbeat.classes import BEAT
from alignbeat.dp import subset_select_dp

# The rungs, in the order they are reported. Each adds one oracle to the one before.
RUNGS = ("real", "real+latent", "+detect", "+latent", "+meter", "+class")


def load(ckpt_path, device):
    import inspect
    from beat_this.model.pl_module import PLBeatThis
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = {k: v for k, v in ck.get("hyper_parameters", {}).items()
          if k in set(inspect.signature(PLBeatThis.__init__).parameters)}
    m = PLBeatThis(**hp)
    missing, _ = m.load_state_dict(ck["state_dict"], strict=False)
    assert not [k for k in missing if "criterion" not in k], "architecture mismatch"
    return m.eval().to(device)


def true_meter(classes):
    """The annotated L: the modal gap between consecutive downbeats."""
    pos = (classes == DOWNBEAT).nonzero(as_tuple=False).flatten()
    if pos.numel() < 2:
        return 0
    return int(np.median(np.diff(pos.cpu().numpy())))


def downbeat_call(model, class_logits, t_hat, target, sigma, meter=None):
    """r_i > 0.5 on the matched events, with every class label hidden.

    This is the beat-only path run on labelled data: the E-step never sees a label, so
    the downbeats it returns come from the bar-phase posterior alone. meter=None leaves
    L to be inferred; an integer pins it, which prices our meter inference against the
    annotation. Returns None when no hypothesis is viable."""
    from launch_scripts.latent_meter import downbeat_mass
    crit = model.subset_criterion
    if meter is not None:
        if meter <= 1:
            return None
        saved = (crit.meter_candidates, crit.meter_prior)
        crit.meter_candidates = (meter,)
        crit.meter_prior = {meter: 1.0 / meter}    # the only hypothesis, so pi_M(L)=1
    try:
        masked = torch.full_like(target["classes"], CLASS_UNKNOWN)
        match = crit._e_step(class_logits, t_hat, masked, target["times"])
        if match.pi is None:
            return None
        r = downbeat_mass(crit, match.pi, int(target["classes"].numel()))
        # _e_step resolves its own sigma; it is the same time-only match the ladder
        # uses, but index by its own to stay consistent if that ever changes.
        return (r > 0.5).cpu().numpy()
    finally:
        if meter is not None:
            crit.meter_candidates, crit.meter_prior = saved


@torch.no_grad()
def run(model, loader, device):
    from alignbeat.decode import decode_events
    # Deferred: latent_meter imports load() from here, so a module-level import cycles.
    from launch_scripts.latent_meter import downbeat_mass
    rows = []
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
            t_hat = pred["t_hat"][i].cpu().numpy()
            gt_t = target["times"].cpu().numpy()
            gt_c = target["classes"].cpu().numpy()
            if len(gt_t) < 2 or len(gt_t) > len(t_hat):
                continue

            # The oracled event set, shared by every rung below "real": match on time
            # alone, so sigma is which candidates are genuinely events.
            sigma = subset_select_dp(np.abs(gt_t[:, None] - t_hat[None, :]))
            matched_sec = t_hat[sigma] * window
            true_is_db = gt_c == DOWNBEAT

            logits = pred["class_logits"][i].float()
            log_p = torch.log_softmax(logits, dim=-1)
            crit = model.subset_criterion
            span = log_p[torch.from_numpy(sigma).to(device)]
            head_is_db = (span[:, DOWNBEAT] > span[:, BEAT]).cpu().numpy()

            # r_i on the same events, with the labels hidden exactly as training hides
            # them on beat-only data. Free L, then L pinned to the annotation.
            L_true = true_meter(target["classes"])
            latent_is_db = downbeat_call(model, logits, pred["t_hat"][i].float(),
                                         target, sigma, meter=None)
            meter_is_db = downbeat_call(model, logits, pred["t_hat"][i].float(),
                                        target, sigma, meter=L_true)

            # ALGORITHM 3, honestly: score the bar hypotheses over the candidates the
            # model itself emitted. Same arithmetic as the E-step after sigma, but
            # sigma here is the decode's own keep-mask, so no gt time or count leaks in.
            real_latent_is_db = None
            probs = torch.softmax(pred["class_logits"][i].float(), dim=-1)
            top, arg = probs.max(dim=-1)
            keep = (arg != BACKGROUND) & (top >= model.tau)
            if int(keep.sum()) >= 4:
                span_r = log_p[keep]
                scores_h = crit._log_meter_phase_scores(crit._class_log_posterior(span_r))
                if scores_h:
                    flat = torch.cat([scores_h[L] for L in scores_h])
                    pi = torch.softmax(flat, dim=0)
                    r = downbeat_mass(crit, pi, int(keep.sum()))
                    real_latent_is_db = (r > 0.5).cpu().numpy()
                    real_latent_sec = (pred["t_hat"][i].float()[keep]
                                       * window).cpu().numpy()

            # REAL: the model's own decode, its own events and its own classes
            cls, times, _ = decode_events(pred["class_logits"][i].float(),
                                          pred["t_hat"][i].float(), model.tau)
            real_sec = (times * window).cpu().numpy()
            real_is_db = (cls == DOWNBEAT).cpu().numpy()

            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            has_db = bool(batch["downbeat_mask"][i]) and len(truth_db) >= 3
            row = dict(corpus=str(batch["spect_path"][i]).split("/", 1)[0],
                       bpm=float(60.0 / np.median(np.diff(truth_b))), L_true=L_true)
            # Upstream value, not a score: how often each rung's DB/B call is right.
            for tag, isdb in (("+detect", head_is_db), ("+latent", latent_is_db),
                              ("+meter", meter_is_db)):
                if isdb is not None and has_db:
                    row[f"{tag}_acc"] = float((isdb == true_is_db).mean())
            for tag, sec, isdb in (("real", real_sec, real_is_db),
                                   ("real+latent",
                                    real_latent_sec if real_latent_is_db is not None
                                    else None, real_latent_is_db),
                                   ("+detect", matched_sec, head_is_db),
                                   ("+latent", matched_sec, latent_is_db),
                                   ("+meter", matched_sec, meter_is_db),
                                   ("+class", matched_sec, true_is_db)):
                if isdb is None:
                    continue
                row[f"{tag}_F"] = model.metrics(truth_b, np.sort(sec), step="test")["F-measure"]
                if has_db:
                    row[f"{tag}_dbF"] = model.metrics(
                        truth_db, np.sort(sec[isdb]), step="test")["F-measure"]
            rows.append(row)
    return rows


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
    rows = run(load(sorted(glob.glob(args.checkpoint))[0], device), dm.val_dataloader(), device)

    def mean(sel, key):
        v = [r[key] for r in sel if key in r]
        return np.mean(v) if v else float("nan")

    def show(name, sel):
        if not sel:
            return
        cells = ""
        prev = None
        for rung in RUNGS:
            db = mean(sel, f"{rung}_dbF")
            step = "" if prev is None or not np.isfinite(db) else f"{db - prev:+6.3f}"
            cells += f"{db:>8.3f}{step:>7}"
            if np.isfinite(db):
                prev = db
        print(f"  {name:<14}{len(sel):>4}{mean(sel, 'real_F'):>8.3f}"
              f"{mean(sel, '+detect_F'):>8.3f}   {cells}")

    head = "".join(f"{r:>8}{'step':>7}" for r in RUNGS)
    print(f"\n{'':14}{'':4}{'beat F':>16}   {'downbeat F by rung':>16}")
    print(f"  {'group':<14}{'n':>4}{'real':>8}{'+detect':>8}   {head}")
    show("ALL", rows)
    for lo, hi in ((0,70),(70,100),(100,130),(130,160),(160,1e9)):
        show(f"bpm {lo}-{hi:.0f}" if hi < 1e9 else f"bpm {lo}+",
             [r for r in rows if lo <= r["bpm"] < hi])
    for c in sorted({r["corpus"] for r in rows}):
        show(f"ds:{c}", [r for r in rows if r["corpus"] == c])


if __name__ == "__main__":
    main()
