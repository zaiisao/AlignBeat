"""Are the predicted downbeats spaced like bars?

Bar length = number of beats between consecutive downbeats. From the model's own decode
(Algorithm 10, per candidate) and from the annotation, per fragment: how many bars
differ from the fragment's modal annotated length, how often the length changes, and
whether wrong bars are too short (a half-bar called a downbeat) or too long. On
E_offset ep19 the annotation is regular (0.6% irregular bars, 0.09 changes/fragment)
and the prediction is not (19%, 3.2) -- and the frozen dense head, decoded with its own
minimal postprocessor, shows the same (17%, 2.0), so it is what per-position downbeat
classification produces on this encoder, not this head's pathology.
"""
import argparse, glob, os, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alignbeat.classes import DOWNBEAT
from alignbeat.decode import decode_events
from launch_scripts.oracle_ceiling import load


def bar_lengths(classes):
    idx = np.where(np.asarray(classes) == DOWNBEAT)[0]
    return np.diff(idx) if len(idx) > 1 else np.array([], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=args.fold)
    dm.setup(stage="fit")
    device = f"cuda:{args.gpu}"
    model = load(sorted(glob.glob(args.checkpoint))[0], device)

    rows = []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            if not bool(batch["downbeat_mask"][0]):
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model.model(batch["spect"])
            cls_pred, _, _ = decode_events(pred["class_logits"][0].float(),
                                           pred["t_hat"][0].float(), 0.2)
            gt_c = model._subset_targets(batch)[0]["classes"].cpu().numpy()
            Lp, Lt = bar_lengths(cls_pred.cpu().numpy()), bar_lengths(gt_c)
            if len(Lt) < 2 or len(Lp) < 2:
                continue
            mode = Counter(Lt.tolist()).most_common(1)[0][0]
            rows.append(dict(
                ds=batch["dataset"][0], t_mode=mode,
                t_irregular=float(np.mean(Lt != mode)), t_changes=int(np.sum(np.diff(Lt) != 0)),
                p_irregular=float(np.mean(Lp != mode)), p_changes=int(np.sum(np.diff(Lp) != 0)),
                p_mode=Counter(Lp.tolist()).most_common(1)[0][0],
                p_short=float(np.mean(Lp < mode)), p_long=float(np.mean(Lp > mode)),
                p_seq=Lp[:12].tolist(), t_seq=Lt[:12].tolist()))

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n{os.path.basename(args.checkpoint)[:30]}: {len(df)} fragments with >= 2 bars "
          f"in both annotation and prediction")
    print("bar length (beats between consecutive downbeats); 'irregular' = differs from the "
          "annotation's modal length")
    print(f"annotation: irregular {df.t_irregular.mean():.1%}, length changes/frag {df.t_changes.mean():.2f}")
    print(f"prediction: irregular {df.p_irregular.mean():.1%}, length changes/frag {df.p_changes.mean():.2f}; "
          f"too short {df.p_short.mean():.1%}, too long {df.p_long.mean():.1%}")
    print(f"fragments whose predicted modal length != annotated: {(df.p_mode != df.t_mode).mean():.1%}")
    print(f"fragments with >= 3 predicted length changes over a constant annotation: "
          f"{((df.p_changes >= 3) & (df.t_changes == 0)).mean():.1%}")
    print("\nper dataset (annotation irregular | prediction irregular | changes/frag | short | long):")
    for ds, g in df.groupby("ds"):
        print(f"   {ds:16s} n={len(g):3d}  {g.t_irregular.mean():5.1%} | {g.p_irregular.mean():5.1%} | "
              f"{g.p_changes.mean():4.1f} | {g.p_short.mean():5.1%} | {g.p_long.mean():5.1%}")
    print(f"\nworst {args.examples} fragments (predicted bar lengths vs annotated):")
    for _, r in df.sort_values("p_irregular", ascending=False).head(args.examples).iterrows():
        print(f"   {r.ds:12s} pred {r.p_seq}   true {r.t_seq}")


if __name__ == "__main__":
    main()
