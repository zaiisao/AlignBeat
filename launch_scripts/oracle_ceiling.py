"""What would F be if the classifier were perfect? An oracle decode on fold 0.

The A/B/C/D breakdown says which STAGE fails; this says what that failure COSTS in the
metric the paper reports. Three decodes over the same predicted t_hat:

  oracle   every candidate the time-only DP matches to a ground-truth event is emitted
           with that event's true class, everything else is background. This is the
           ceiling candidate placement allows: perfect classification, real timing.
  real     the model's own argmax + threshold, i.e. what decode_events does.
  gap      what the classifier costs.

Timing is never oracled -- t_hat is the model's throughout -- so the ceiling already
includes whatever localisation error remains.
"""
import argparse, glob, os, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import BACKGROUND, DOWNBEAT
from alignbeat.dp import subset_select_dp


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


@torch.no_grad()
def run(model, loader, device):
    from alignbeat.decode import decode_events
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

            # ORACLE: match on time alone, then hand each matched candidate its true class
            sigma = subset_select_dp(np.abs(gt_t[:, None] - t_hat[None, :]))
            oracle_sec = t_hat[sigma] * window
            oracle_is_db = gt_c == DOWNBEAT

            # REAL: the model's own decode
            cls, times, _ = decode_events(pred["class_logits"][i].float(),
                                          pred["t_hat"][i].float(),
                                          model.tau_beat, model.tau_downbeat)
            real_sec = (times * window).cpu().numpy()
            real_is_db = (cls == DOWNBEAT).cpu().numpy()

            truth_db = np.frombuffer(batch["truth_orig_downbeat"][i])
            has_db = bool(batch["downbeat_mask"][i]) and len(truth_db) >= 3
            row = dict(corpus=str(batch["spect_path"][i]).split("/", 1)[0],
                       bpm=float(60.0 / np.median(np.diff(truth_b))))
            for tag, sec, isdb in (("oracle", oracle_sec, oracle_is_db),
                                   ("real", real_sec, real_is_db)):
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

    def show(name, sel):
        if not sel: return
        o  = np.mean([r["oracle_F"] for r in sel]); rl = np.mean([r["real_F"] for r in sel])
        db = [r for r in sel if "oracle_dbF" in r]
        od = np.mean([r["oracle_dbF"] for r in db]) if db else float("nan")
        rd = np.mean([r["real_dbF"] for r in db]) if db else float("nan")
        print(f"  {name:<12}{len(sel):>5}   {o:6.3f}{rl:8.3f}{o-rl:+8.3f}   "
              f"{od:8.3f}{rd:8.3f}{od-rd:+8.3f}")

    print(f"\n{'group':<12}{'n':>5}   {'oracle':>6}{'real':>8}{'cost':>8}   "
          f"{'or.db':>8}{'real db':>8}{'cost':>8}")
    show("ALL", rows)
    for lo, hi in ((0,70),(70,100),(100,130),(130,160),(160,1e9)):
        show(f"bpm {lo}-{hi:.0f}" if hi < 1e9 else f"bpm {lo}+",
             [r for r in rows if lo <= r["bpm"] < hi])
    for c in sorted({r["corpus"] for r in rows}):
        show(f"ds:{c}", [r for r in rows if r["corpus"] == c])


if __name__ == "__main__":
    main()
