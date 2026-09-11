"""Does the latent meter work?  Masked-label test on annotated data.

Two splits. --split test is held-out GTZAN, never trained or validated on, but 93.7%
of it is in 4, so a constant meter is nearly unbeatable there and the test says little
about metrically varied music. --split val is fold 0's validation set: also unseen by
this checkpoint, and it carries asap, hainsworth and the rest, which is where the
oracle test says the downbeat gap actually lives. Per-dataset numbers only mean
something on that split, so it reports them.

GTZAN is the test set -- never trained on, never validated on -- and it carries downbeat
annotations, so the true meter L and the true DB/B label of every event are known. We
hide both, run the beat-only path exactly as training would (sigma resolved from the
prior-combined marginal, with no label anywhere in the matching cost), and score what
the latent variable recovers against the truth we withheld.

Two numbers matter, each against the trivial baseline that beats it:
  L    argmax_L P(L | x)  vs  always answering the corpus mode, L=4
  r_i  (r_i > 0.5) as a downbeat call  vs  always answering B

r_i is the one that decides whether the surrogate is fit to train on: it is literally
the target every beat-only event is regressed toward.
"""
import argparse, collections, glob, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alignbeat.classes import BEAT, CLASS_UNKNOWN, DOWNBEAT
from launch_scripts.oracle_ceiling import load


def downbeat_mass(crit, pi, M):
    """r_i = P(event i is a downbeat): pi_{omega,L} summed over the hypotheses whose
    pattern c_i(omega, L) calls i a downbeat. The criterion no longer forms this --
    line 52 is a dot product over hypotheses -- but a per-event probability is what a
    diagnostic wants, so rebuild it here from the E-step's own pi."""
    i0 = torch.arange(M, device=pi.device)
    r = torch.zeros(M, device=pi.device, dtype=pi.dtype)
    start = 0
    for meter in crit.meter_candidates:
        meter = int(meter)
        if meter <= 1 or M < meter:
            continue
        block = pi[start:start + meter]
        start += meter
        r = r + block[(-i0) % meter]
    return r


def true_meter(classes):
    """The annotated L: the modal gap between consecutive downbeats."""
    pos = (classes == DOWNBEAT).nonzero(as_tuple=False).flatten()
    if pos.numel() < 2:
        return 0
    return int(np.median(np.diff(pos.cpu().numpy())))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="test = held-out GTZAN; val = fold 0 validation, per dataset")
    ap.add_argument("--flat-class-prior", action="store_true", dest="flat",
                    help="set pi_C to (0.5, 0.5), removing the per-event -1.008 nat "
                         "cost of claiming a downbeat. pi_C(DB)=E[1/L] is derived from "
                         "METER_PRIOR, the same prior already applied per hypothesis as "
                         "log P(L), so charging it again per claimed event double-counts "
                         "the downbeat base rate -- and does so proportionally to how "
                         "many downbeats a hypothesis claims, i.e. biased toward large L")
    ap.add_argument("--only-meter", type=int, default=None,
                    help="restrict the hypothesis set to this single L, leaving only "
                         "its phases. Scores phase inference with the meter handed to "
                         "it, so r_i's dependence on getting L right is isolated")
    ap.add_argument("--uniform", action="store_true",
                    help="null pi_M, as meter_posterior.py does, to separate the "
                         "network's own evidence from the prior's contribution")
    args = ap.parse_args()

    from beat_this.dataset import BeatDataModule
    dm = BeatDataModule(Path("data"), batch_size=1, train_length=1500, spect_fps=50,
                        num_workers=2, test_dataset="gtzan",
                        length_based_oversampling_factor=0.65, augmentations={},
                        hung_data=False, no_val=False, fold=0)
    dm.setup(stage="test" if args.split == "test" else "fit")
    device = f"cuda:{args.gpu}"
    path = sorted(glob.glob(args.checkpoint))[0]
    model = load(path, device)
    crit = model.subset_criterion
    if args.only_meter:
        L = args.only_meter
        crit.meter_candidates = (L,)
        crit.meter_prior = {L: 1.0 / L}           # the only hypothesis, so pi_M(L)=1
    if args.uniform:
        crit.meter_prior = None
    if args.flat:
        crit.class_prior = torch.tensor([0.5, 0.5], device=device)

    conf = collections.Counter(); n_frag = 0
    ev_tp = ev_fp = ev_fn = ev_tn = 0
    hd_tp = hd_fp = hd_fn = hd_tn = 0     # the head's own DB/B argmax, what decode uses
    both_right = r_only = head_only = 0
    r_on_db, r_on_b = [], []
    loader = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()
    # per dataset: L correct, L=4 correct, fragments, tp, fp, fn, tn, head correct
    per_ds = collections.defaultdict(lambda: [0] * 8)
    # Selective prediction: is confidence calibrated enough to abstain on?
    # per fragment (max_L P(L|x), L correct); per event (max(r,1-r), r right, head right)
    L_conf, ev_conf = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # GTZAN clips run ~1519 frames; the head is built for one 1500-frame window.
        # Crop spect and the frame-wise truths together so _subset_targets derives
        # window_seconds = 30.0 and keeps exactly the events inside it -- the same
        # fragment condition training sees, rather than a stitched whole-piece decode.
        W = 1500
        batch["spect"] = batch["spect"][:, :W]
        for k in ("truth_beat", "truth_downbeat"):
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k][..., :W]
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.model(batch["spect"])
        pred = {k: v.float() for k, v in pred.items()}
        for i, t in enumerate(model._subset_targets(batch)):
            gt_c, gt_t = t["classes"], t["times"]
            if gt_c.numel() < 4 or bool((gt_c == CLASS_UNKNOWN).any()):
                continue                       # need real DB/B labels to score against
            L_true = true_meter(gt_c)
            if L_true not in crit.meter_candidates:
                continue
            log_p = torch.log_softmax(pred["class_logits"][i], dim=-1)
            # THE MASK: every label hidden, so sigma is resolved by the beat-only
            # branch of build_l_match exactly as it is on real beat-only data.
            masked = torch.full_like(gt_c, CLASS_UNKNOWN)
            m = crit._e_step(pred["class_logits"][i], pred["t_hat"][i], masked, gt_t)
            if m.pi is None:
                continue
            r = downbeat_mass(crit, m.pi, gt_c.numel())
            n_frag += 1
            post = crit._meter_log_posterior(log_p[torch.from_numpy(m.sigma).to(device)])
            L_hat = max(post, key=lambda L: float(post[L]))
            conf[(L_true, L_hat)] += 1

            is_db = (gt_c == DOWNBEAT)
            called = r > 0.5
            ev_tp += int((called & is_db).sum());  ev_fp += int((called & ~is_db).sum())
            ev_fn += int((~called & is_db).sum()); ev_tn += int((~called & ~is_db).sum())
            r_on_db += r[is_db].tolist(); r_on_b += r[~is_db].tolist()

            # What decode_events does today: each matched candidate independently takes
            # argmax over DB/B, with no bar-phase constraint tying the events together.
            span = log_p[torch.from_numpy(m.sigma).to(device)]
            head = span[:, DOWNBEAT] > span[:, BEAT]
            hd_tp += int((head & is_db).sum());  hd_fp += int((head & ~is_db).sum())
            hd_fn += int((~head & is_db).sum()); hd_tn += int((~head & ~is_db).sum())
            agree_r = (called == is_db); agree_h = (head == is_db)
            both_right += int((agree_r & agree_h).sum())
            r_only += int((agree_r & ~agree_h).sum())
            head_only += int((agree_h & ~agree_r).sum())

            d = per_ds[batch["dataset"][i] if "dataset" in batch else "?"]
            d[0] += int(L_hat == L_true); d[1] += int(L_true == 4); d[2] += 1
            d[3] += int((called & is_db).sum());  d[4] += int((called & ~is_db).sum())
            d[5] += int((~called & is_db).sum()); d[6] += int((~called & ~is_db).sum())
            d[7] += int((head == is_db).sum())

            pmax = float(torch.softmax(torch.stack([post[L] for L in post]), 0).max())
            L_conf.append((pmax, L_hat == L_true))
            conf_r = torch.maximum(r, 1 - r)
            ev_conf += list(zip(conf_r.tolist(), agree_r.tolist(), agree_h.tolist()))

    print(f"\n{'GTZAN' if args.split == 'test' else 'fold 0 validation'}, labels masked, {'UNIFORM' if args.uniform else 'corpus'} pi_M, "
          f"{'FLAT' if args.flat else 'corpus'} pi_C")
    print(f"checkpoint: {Path(path).name}")
    print(f"fragments scored: {n_frag}\n")

    tot = sum(conf.values()); right = sum(v for (a, b), v in conf.items() if a == b)
    trues = collections.Counter({L: sum(v for (a, _), v in conf.items() if a == L)
                                 for L, _ in {(a, 0) for a, _ in conf}})
    mode = max(trues, key=trues.get)
    print(f"1  L: argmax correct {right}/{tot} = {100*right/tot:.1f}%"
          f"   |  always L={mode}: {100*trues[mode]/tot:.1f}%"
          f"   -> {100*(right-trues[mode])/tot:+.1f} pts")
    ls = sorted({a for a, _ in conf} | {b for _, b in conf})
    print("     true \\ pred " + "".join(f"{L:>6}" for L in ls) + "     n    acc")
    for a in sorted(trues):
        row = [conf.get((a, b), 0) for b in ls]
        na = sum(row)
        print(f"        {a:>6}    " + "".join(f"{v:>6}" for v in row) +
              f"  {na:>4}  {100*conf.get((a,a),0)/max(na,1):5.1f}%")

    ev = ev_tp + ev_fp + ev_fn + ev_tn
    acc = 100 * (ev_tp + ev_tn) / max(ev, 1)
    base = 100 * (ev_tn + ev_fp) / max(ev, 1)
    prec = 100 * ev_tp / max(ev_tp + ev_fp, 1); rec = 100 * ev_tp / max(ev_tp + ev_fn, 1)
    print(f"\n2  r_i as a downbeat call, over {ev} events")
    print(f"     accuracy {acc:.1f}%   |  always-B baseline {base:.1f}%   -> {acc-base:+.1f} pts")
    print(f"     precision {prec:.1f}%   recall {rec:.1f}%   (tp {ev_tp} fp {ev_fp} fn {ev_fn})")
    hacc = 100 * (hd_tp + hd_tn) / max(ev, 1)
    hprec = 100 * hd_tp / max(hd_tp + hd_fp, 1); hrec = 100 * hd_tp / max(hd_tp + hd_fn, 1)
    print(f"\n3  the head's own argmax on the same events -- what decode_events uses")
    print(f"     accuracy {hacc:.1f}%   precision {hprec:.1f}%   recall {hrec:.1f}%"
          f"   (tp {hd_tp} fp {hd_fp} fn {hd_fn})")
    print(f"     r_i - head: {acc - hacc:+.1f} pts")
    print(f"     r_i right where head wrong: {r_only}   head right where r_i wrong: {head_only}")

    print(f"\n4  mean r_i on true downbeats {np.mean(r_on_db):.3f} "
          f"(should approach 1) | on true beats {np.mean(r_on_b):.3f} (should approach 0)")
    und = np.mean([1.0 for v in r_on_db + r_on_b if max(v, 1 - v) < 0.7])
    print(f"     events the 0.7 confidence gate would refuse: "
          f"{100*sum(1 for v in r_on_db+r_on_b if max(v,1-v)<0.7)/len(r_on_db+r_on_b):.1f}%")

    def band(rows, edges, head=None):
        print(f"     {'confidence':>14}{'n':>7}{'share':>8}{'accuracy':>10}"
              + (f"{'head':>8}{'delta':>8}" if head else ""))
        for lo, hi in zip(edges, edges[1:]):
            sel = [x for x in rows if lo <= x[0] < hi]
            if not sel:
                continue
            acc = 100 * sum(x[1] for x in sel) / len(sel)
            line = (f"     [{lo:.2f}, {hi:.2f}){len(sel):>7}"
                    f"{100*len(sel)/len(rows):>7.1f}%{acc:>9.1f}%")
            if head:
                h = 100 * sum(x[2] for x in sel) / len(sel)
                line += f"{h:>7.1f}%{acc-h:>+8.1f}"
            print(line)

    print("\n5  is max_L P(L | x) calibrated?  (fragments)")
    band(L_conf, [0.0, 0.5, 0.7, 0.9, 0.99, 1.01])

    print("\n6  is max(r, 1-r) calibrated?  r_i vs the head within each band (events)")
    band(ev_conf, [0.5, 0.7, 0.9, 0.99, 1.01], head=True)

    if per_ds:
        print("\n7  per dataset -- L argmax vs always-4, and r_i vs the head")
        print(f"     {'dataset':<16}{'frag':>6}{'L acc':>8}{'always4':>9}"
              f"{'r_i acc':>9}{'head':>8}{'delta':>7}{'r_i rec':>9}")
        for ds in sorted(per_ds):
            ok, a4, n, tp, fp, fn, tn, hok = per_ds[ds]
            ev = tp + fp + fn + tn
            racc = 100 * (tp + tn) / max(ev, 1)
            hacc = 100 * hok / max(ev, 1)
            print(f"     {ds:<16}{n:>6}{100*ok/max(n,1):>7.1f}%{100*a4/max(n,1):>8.1f}%"
                  f"{racc:>8.1f}%{hacc:>7.1f}%{racc-hacc:>+7.1f}"
                  f"{100*tp/max(tp+fn,1):>8.1f}%")


if __name__ == "__main__":
    main()

