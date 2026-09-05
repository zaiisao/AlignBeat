"""Decoding for the continuous-phase arm: candidates -> beat and downbeat times.

The counterpart of alignbeat/decode.py's Algorithm 10 for the phase head. The two arms
decide "is this candidate an event" in fundamentally different ways, and that difference
is the whole of the difference between them:

  subset  a learned posterior. p(BACKGROUND) is read from the candidate's own features,
          so the model can decline a candidate because the AUDIO says nothing is there.
  phase   a geometric test. A candidate fires when its predicted phase lands within tau
          of a grid point of the inferred meter, so nothing acoustic can veto it.

The second has no acoustic veto by construction, which is worth keeping in view when
comparing the two arms' firing accuracy (the 'C fire' column of diagnose_bottleneck.py).
"""
import math

import torch

from alignbeat.classes import METER_PRIOR
from alignbeat.phase_criterion import circ_dist

CANDIDATE_METERS = (2, 3, 4, 6)
TAU = 0.2
TAU_PRIME = 1.5 * TAU


def fit_ell(phi_hat, ell):
    """fit(l) = sum_j min_k d_circ(phi_hat_j, k/l): how well a meter of l explains phi."""
    grid = torch.arange(ell, dtype=phi_hat.dtype, device=phi_hat.device) / ell
    return circ_dist(phi_hat[:, None], grid[None, :]).min(dim=1).values.sum()


def infer_meter(phi_hat, candidate_meters=CANDIDATE_METERS, lambda_phi=3.0,
                meter_prior=None):
    """The meter L that best explains the predicted phases, prior-weighted.

    The -N/(4l) term is the bias correction: a larger meter has more grid points, so it
    fits any phase distribution better by chance, and without the correction the choice
    would run away to the largest candidate.
    """
    prior = METER_PRIOR if meter_prior is None else meter_prior
    N = phi_hat.shape[0]

    best_score, best_ell = float("inf"), candidate_meters[0]
    for ell in candidate_meters:
        score = (float(fit_ell(phi_hat, ell)) - N / (4 * ell)
                 - (1.0 / lambda_phi) * math.log(prior.get(ell, 0.0) + 1e-12))
        if score < best_score:
            best_score, best_ell = score, ell
    return best_ell


def decode(phi_hat, t_hat, meter, tau=TAU):
    """Nearest-grid-point decoding at a known meter.

    Returns (positions, distances, accepted), where positions[j] is the bar position the
    candidate claims, distances[j] its disagreement with that position, and accepted the
    (position, time) pairs within tau. Already in time order, since t_hat is strictly
    increasing by construction.
    """
    positions = (torch.round(phi_hat * meter).long() % meter)
    distances = circ_dist(phi_hat, positions.to(phi_hat.dtype) / meter)
    accepted = [(int(positions[j]), float(t_hat[j]))
                for j in range(phi_hat.shape[0]) if float(distances[j]) <= tau]
    return positions.tolist(), distances.tolist(), accepted


def meter_consistency_correction(accepted, meter, positions, distances, times,
                                 tau=TAU, tau_prime=TAU_PRIME):
    """Gap filling and pruning against the inferred bar length.

    A gap between consecutive downbeats much larger than one bar suggests a downbeat was
    missed, and one much smaller suggests a spurious one; the first is filled from the
    nearest candidate under a relaxed threshold, the second pruned. This is a repair pass
    over a decision the model already made, not part of the model.
    """
    downbeats = sorted((j for j in range(len(positions))
                        if positions[j] == 0 and distances[j] <= tau),
                       key=lambda j: times[j])
    if len(accepted) < 2 or len(downbeats) < 2:
        return accepted

    span = max(t for _, t in accepted) - min(t for _, t in accepted)
    bar = meter * (span / max(len(accepted) - 1, 1))
    known = {(positions[j], times[j]) for j in downbeats}
    corrected = list(accepted)

    for k in range(len(downbeats) - 1):
        left, right = downbeats[k], downbeats[k + 1]
        gap = times[right] - times[left]

        if gap > 1.5 * bar:
            between = [j for j in range(len(positions))
                       if times[left] < times[j] < times[right]
                       and (positions[j], times[j]) not in known]
            if between:
                nearest = min(between, key=lambda j: abs(times[j] - (times[left] + bar)))
                if positions[nearest] == 0 and distances[nearest] <= tau_prime:
                    corrected.append((0, times[nearest]))
        elif gap < 0.5 * bar:
            weakest = left if distances[left] >= distances[right] else right
            entry = (positions[weakest], times[weakest])
            if entry in corrected and distances[weakest] > tau:
                corrected.remove(entry)

    return sorted(set(corrected), key=lambda pair: pair[1])


def decode_events(phi_hat, t_hat, tau=TAU, tau_prime=TAU_PRIME,
                  candidate_meters=CANDIDATE_METERS, lambda_phi=3.0):
    """One excerpt -> (beat_times, downbeat_times) on t_hat's own (0, 1] axis.

    Mirrors alignbeat.decode.decode_events' contract so pl_module can treat the two arms
    identically: downbeats are a SUBSET of beats, as the dense arm's targets also have
    it, since bar position 0 is still a beat.
    """
    meter = infer_meter(phi_hat, candidate_meters, lambda_phi)
    positions, distances, accepted = decode(phi_hat, t_hat, meter, tau)
    accepted = meter_consistency_correction(accepted, meter, positions, distances,
                                            t_hat.tolist(), tau, tau_prime)

    beats = [t for _, t in accepted]
    downbeats = [t for p, t in accepted if p == 0]
    return beats, downbeats, meter
