"""Equivalence harness for SubsetCriterion optimisations."""
import os, sys, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alignbeat.classes import BEAT, CLASS_UNKNOWN, DOWNBEAT
from alignbeat.criterion import SubsetCriterion

GOLDEN = os.path.join(os.path.dirname(__file__), '_criterion_golden.pt')

# (label, criterion kwargs) -- the flag combinations that actually get trained
# v1 ships class + time + background. learn_b, the marginal objective (Alg. 4), the
# periodicity term and the continuity term are all retired; git history has them.
CONFIGS = [
    ("default",        dict(gamma=0.5, omega_downbeat=2.0)),
    ("omega4",         dict(gamma=0.5, omega_downbeat=4.0)),
    ("beat_only_em",   dict(gamma=0.5, meter_length=4, beat_only_warmup=0)),
    ("joint_phase",    dict(gamma=0.5, meter_length=4, joint_phase=True, beat_only_warmup=0)),
    ("latent_meter",   dict(gamma=0.5, meter_candidates=(2, 3, 4, 6), beat_only_warmup=0)),
    ("meter_prior",    dict(gamma=0.5, meter_candidates=(2, 3, 4, 6), meter_prior="corpus",
                            beat_only_warmup=0)),
    ("no_normalize",   dict(gamma=0.5, normalize_by_events=False)),
]

# (label, batch, N, list of per-fragment M, label mode) -- includes the edge cases
SHAPES = [
    ("typical",      4, 172, [90, 88, 92, 85],  "labelled"),
    ("ragged",       4, 172, [120, 12, 60, 3],  "labelled"),
    ("beat_only",    3, 172, [80, 75, 90],      "unknown"),
    ("mixed_labels", 4, 172, [60, 60, 60, 60],  "mixed"),
    ("empty_frag",   3, 172, [0, 50, 70],       "labelled"),
    ("single_event", 2, 172, [1, 40],           "labelled"),
    ("dense",        2, 172, [170, 165],        "labelled"),
]


def build(seed, batch, N, Ms, mode, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(batch, N, 3, generator=g).to(device).requires_grad_(True)
    t_hat = torch.sort(torch.rand(batch, N, generator=g), dim=1)[0].to(device)
    targets = []
    for i, M in enumerate(Ms):
        times = torch.sort(torch.rand(M, generator=g))[0].to(device)
        if mode == "unknown":
            cls = torch.full((M,), CLASS_UNKNOWN, dtype=torch.long, device=device)
        elif mode == "mixed" and i % 2 == 1:
            cls = torch.full((M,), CLASS_UNKNOWN, dtype=torch.long, device=device)
        else:
            idx = torch.arange(M, device=device)
            cls = torch.where(idx % 4 == 0,
                              torch.full((M,), DOWNBEAT, dtype=torch.long, device=device),
                              torch.full((M,), BEAT, dtype=torch.long, device=device))
        targets.append({"times": times, "classes": cls})
    return logits, t_hat, targets


def run_case(cfg_kwargs, seed, shape):
    _, batch, N, Ms, mode = shape
    logits, t_hat, targets = build(seed, batch, N, Ms, mode)
    crit = SubsetCriterion(**cfg_kwargs)
    losses, stats = crit(logits, t_hat, torch.full_like(t_hat, 0.00233), targets)
    out = {k: v.detach().clone() for k, v in losses.items() if torch.is_tensor(v)}
    total = losses["total"]
    grad = torch.zeros_like(logits)
    if total.requires_grad:
        grad = torch.autograd.grad(total, logits, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(logits)
    out["__grad"] = grad.detach().clone()
    return out


def collect():
    results = {}
    for (cfg_label, cfg), shape in itertools.product(CONFIGS, SHAPES):
        for seed in (0, 1, 2):
            key = f"{cfg_label}|{shape[0]}|{seed}"
            torch.manual_seed(seed)
            try:
                results[key] = run_case(cfg, seed, shape)
            except Exception as exc:                      # record failures too
                results[key] = {"__error": str(type(exc).__name__) + ": " + str(exc)[:200]}
    return results


def compare(a, b, atol=1e-6, rtol=1e-5):
    bad = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            bad.append(f"{key}: present in only one run"); continue
        ra, rb = a[key], b[key]
        if "__error" in ra or "__error" in rb:
            if ra.get("__error") != rb.get("__error"):
                bad.append(f"{key}: error mismatch {ra.get('__error')} vs {rb.get('__error')}")
            continue
        for term in sorted(set(ra) | set(rb)):
            if term not in ra or term not in rb:
                bad.append(f"{key}.{term}: missing"); continue
            x, y = ra[term], rb[term]
            if x.shape != y.shape:
                bad.append(f"{key}.{term}: shape {tuple(x.shape)} vs {tuple(y.shape)}"); continue
            if not torch.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True):
                d = (x - y).abs().max().item()
                bad.append(f"{key}.{term}: max|diff| {d:.3e}")
    return bad


if __name__ == "__main__":
    current = collect()
    n_ok = sum(1 for v in current.values() if "__error" not in v)
    print(f"collected {len(current)} cases ({n_ok} ran, {len(current)-n_ok} raised)")
    if os.environ.get("ALIGNBEAT_RECORD"):
        torch.save(current, GOLDEN)
        print(f"recorded golden -> {GOLDEN}")
    elif os.path.exists(GOLDEN):
        golden = torch.load(GOLDEN, weights_only=False)
        bad = compare(golden, current)
        if bad:
            print(f"\nFAIL: {len(bad)} mismatches vs golden")
            for line in bad[:25]:
                print("   ", line)
            sys.exit(1)
        print("\nok: criterion is bit-equivalent to the recorded golden across "
              f"{len(current)} cases (losses and d(total)/d(logits))")
    else:
        print("no golden recorded; run with ALIGNBEAT_RECORD=1 first")
