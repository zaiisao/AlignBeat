"""The correspondence problem (sections 4-5, 8): Algorithm 1 and its variants."""
import os

import numpy as np
import torch

from alignbeat.classes import BACKGROUND, BEAT, DOWNBEAT


# Optional compiled kernel for recursion (7). The numpy path below is already
# vectorised over j (np.minimum.accumulate), so its cost is not arithmetic but ~5 numpy
# calls per row -- roughly 3600 tiny calls per training step at M=90, N=172, B=8. That
# call overhead is what collapses under multi-process load: three concurrent arms went
# to 1200 ms/step against a single arm's 251 ms, while the dense baseline (pure GPU
# tensor ops) showed no degradation at all. A compiled scalar loop removes the call
# storm entirely and touches no BLAS thread pool.
#
# Falls back to numpy transparently when numba is absent; the two produce BIT-IDENTICAL
# sigma (verified on 300 random cost matrices including forced ties, and by
# tests/test_criterion_equivalence.py).
try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=False)
    def _dp_kernel(cost, choice):
        """Recursion (7) as a compiled scalar loop."""
        M, N = cost.shape
        previous = np.zeros(N + 1, dtype=np.float64)
        current = np.empty(N + 1, dtype=np.float64)
        for i in range(1, M + 1):
            best = np.inf
            best_index = 0
            for j in range(N):
                a = previous[j] + cost[i - 1, j]
                if a <= best:
                    best = a
                    best_index = j + 1
                current[j + 1] = best
                choice[i, j + 1] = best_index
            current[0] = np.inf
            previous, current = current, previous
        return previous[N]

    # ALIGNBEAT_NO_NUMBA=1 forces the numpy path -- used to A/B the two
    # implementations against each other, which must agree bit for bit.
    _HAVE_NUMBA = not os.environ.get("ALIGNBEAT_NO_NUMBA")
except ImportError:                                          # pragma: no cover
    _HAVE_NUMBA = False


def subset_select_dp(cost, return_cost=False):
    """Algorithm 1 - exact O(N*M) order-constrained subset selection."""
    cost = np.asarray(cost, dtype=np.float64)
    M, N = cost.shape
    if M == 0:
        return np.empty(0, dtype=np.int64)
    if M > N:
        raise ValueError(
            f"infeasible correspondence: {M} ground-truth events but only {N} "
            f"candidates. Increase num_candidates or drop the fragment.")

    if _HAVE_NUMBA:
        choice = np.zeros((M + 1, N + 1), dtype=np.int32)
        final = _dp_kernel(np.ascontiguousarray(cost), choice)
        sigma = np.empty(M, dtype=np.int64)
        j = N
        for i in range(M, 0, -1):
            j_star = int(choice[i, j])
            if j_star < 1:
                raise RuntimeError(
                    "backtracking failed; cost matrix contains non-finite values")
            sigma[i - 1] = j_star - 1
            j = j_star - 1
        return (sigma, float(final)) if return_cost else sigma

    # D[i, j], 1-indexed in both axes. D[0, j] = 0 (empty injection costs nothing),
    # D[i, 0] = +inf for i >= 1 (no candidates cannot cover a nonempty domain).
    previous = np.zeros(N + 1, dtype=np.float64)
    choice = np.zeros((M + 1, N + 1), dtype=np.int32)
    candidate_index = np.arange(1, N + 1, dtype=np.int32)

    for i in range(1, M + 1):
        # A[j] = D[i-1, j-1] + cost[i-1, j-1] for j = 1..N
        a = previous[0:N] + cost[i - 1]
        accumulated = np.minimum.accumulate(a)
        # argmin of the running minimum: a[j] attains it exactly when it is a new
        # (weak) minimum, and a later tie is just as optimal as an earlier one.
        is_new_minimum = a <= accumulated
        arg = np.maximum.accumulate(np.where(is_new_minimum, candidate_index, 0))

        current = np.empty(N + 1, dtype=np.float64)
        current[0] = np.inf
        current[1:] = accumulated
        choice[i, 1:] = arg
        previous = current

    sigma = np.empty(M, dtype=np.int64)
    j = N
    for i in range(M, 0, -1):
        j_star = int(choice[i, j])
        if j_star < 1:
            raise RuntimeError("backtracking failed; cost matrix contains non-finite values")
        sigma[i - 1] = j_star - 1
        j = j_star - 1
    if return_cost:
        # previous[] now holds D[M, .]; D[M, N] is the optimal total cost, which
        # eq. (19) compares across phase hypotheses.
        return sigma, float(previous[N])
    return sigma


def subset_select_dp_meter(cost, downbeat_positions, t_hat, meter_length, mu):
    """Section 4.2, equation (6): known-meter spacing folded into the SELECTION."""
    M, N = cost.shape
    if mu <= 0.0 or len(downbeat_positions) < 2:
        return subset_select_dp(cost)
    if M > N:
        raise ValueError(f"infeasible: M={M} > N={N}")

    NONE = N
    inf = np.inf
    is_db = np.zeros(M, dtype=bool)
    is_db[np.asarray(downbeat_positions, dtype=np.int64)] = True
    # Equation (17) defines Delta_bar over MATCHED candidates, which are unknown before
    # sigma is chosen. Take one EM step: solve the plain DP, measure Delta_bar on its
    # matched set, then run the augmented DP against that target. Using the span of all
    # N candidates instead biased the target high by 2.8% (ballroom) to 10% (harmonix),
    # which pushed spacing the WRONG way wherever the constraint bit.
    sigma0 = subset_select_dp(cost)
    delta_bar = float(np.mean(np.diff(t_hat[sigma0]))) if M > 1 else 0.0
    target = meter_length * delta_bar

    D = np.full((N + 1, N), inf)
    prev_j = np.full((M, N + 1, N), -1, dtype=np.int32)
    prev_s = np.full((M, N + 1, N), -1, dtype=np.int32)
    if is_db[0]:
        for j in range(N):
            D[j, j] = cost[0, j]
    else:
        D[NONE, :] = cost[0, :]

    # run[s, j] = min_{j' < j} D[s, j'], the running minimum along the candidate axis,
    # with run_arg the j' achieving it. This is the same quantity subset_select_dp
    # computes with minimum.accumulate, batched over the j* state rows -- written as
    # two vectorised numpy calls rather than the O(N) Python loop it replaces, which
    # was the dominant cost here (M * N interpreter iterations per fragment, ~25k at
    # N=160, on top of the array work the recursion genuinely requires).
    candidate_axis = np.arange(N, dtype=np.int32)[None, :]
    for i in range(1, M):
        run = np.full((N + 1, N), inf)
        run_arg = np.full((N + 1, N), -1, dtype=np.int32)
        if N > 1:
            shifted = D[:, : N - 1]
            accumulated = np.minimum.accumulate(shifted, axis=1)
            run[:, 1:] = accumulated
            # argmin of a running minimum: j' attains it exactly when it is a new (weak)
            # minimum, and a later tie is just as optimal as an earlier one.
            is_new_minimum = shifted <= accumulated
            run_arg[:, 1:] = np.maximum.accumulate(
                np.where(is_new_minimum, candidate_axis[:, : N - 1], 0), axis=1)

        newD = np.full((N + 1, N), inf)
        if not is_db[i]:
            newD = run + cost[i][None, :]
            prev_j[i] = run_arg
            prev_s[i] = np.arange(N + 1, dtype=np.int32)[:, None]
        else:
            gap = t_hat[None, :] - np.concatenate([t_hat, [0.0]])[:, None]
            penalty = mu * (gap - target) ** 2
            penalty[NONE, :] = 0.0
            cand = run + cost[i][None, :] + penalty
            chosen = np.argmin(cand, axis=0).astype(np.int32)
            vals = cand[chosen, np.arange(N)]
            for j in range(N):
                newD[j, j] = vals[j]
                prev_j[i, j, j] = run_arg[chosen[j], j]
                prev_s[i, j, j] = chosen[j]
        D = newD

    flat = int(np.argmin(D))
    js, j = np.unravel_index(flat, D.shape)
    if not np.isfinite(D[js, j]):
        return subset_select_dp(cost)
    sigma = np.empty(M, dtype=np.int64)
    for i in range(M - 1, 0, -1):
        sigma[i] = j
        pj, ps = int(prev_j[i, js, j]), int(prev_s[i, js, j])
        if pj < 0:
            return subset_select_dp(cost)
        j, js = pj, ps
    sigma[0] = j
    return sigma


def event_is_downbeat_under(M, L, p, device=None):
    """Which of the M matched events are downbeats under the hypothesis phi_0 = p."""
    i0 = torch.arange(M, device=device)
    return ((p + i0) % L) == 0


def phase_star(M, L, device=None):
    """p*_i = (1 - i) mod L: the unique hypothesis under which event i is a downbeat."""
    i0 = torch.arange(M, device=device)
    return (-i0) % L


def phase_class_nll(log_probabilities, M, L, p):
    """The class term of eq. (18): -log p_hat_j(phi_j = (p + i - 1) mod L | x; theta)."""
    is_db = event_is_downbeat_under(M, L, p, log_probabilities.device)
    downbeat = -log_probabilities[:, DOWNBEAT]                            # (N,)
    beat = -log_probabilities[:, BEAT]                                    # (N,)
    return torch.where(is_db[:, None], downbeat[None, :], beat[None, :])


def subset_select_dp_joint_phase(costs):
    """Equation (19): the joint MAP of (sigma, phi_0), by L reruns of Algorithm 1."""
    costs = np.asarray(costs, dtype=np.float64)
    L = costs.shape[0]
    best_sigma, best_p, best_cost = None, 0, np.inf
    for p in range(L):
        sigma, total = subset_select_dp(costs[p], return_cost=True)
        if total < best_cost:
            best_sigma, best_p, best_cost = sigma, p, total
    return best_sigma, best_p


def subset_select_dp_phase_segments(cost_fn, M, N, segments):
    """Equation (20): the mixed-meter joint DP, phase carried as augmented state."""
    if M == 0:
        return np.empty(0, dtype=np.int64)
    if M > N:
        raise ValueError(
            f"infeasible correspondence: {M} ground-truth events but only {N} candidates.")

    segment_of, meter_of = np.empty(M, dtype=np.int64), np.empty(M, dtype=np.int64)
    starts = [start for start, _ in segments]
    for k, (start, meter) in enumerate(segments):
        end = segments[k + 1][0] if k + 1 < len(segments) else M
        segment_of[start:end] = k
        meter_of[start:end] = meter
    max_meter = int(meter_of.max())

    # previous[phi, j] = D[i-1, j, phi]; phi >= L_k entries stay +inf and never win.
    previous = np.zeros((max_meter, N + 1), dtype=np.float64)
    choice = np.zeros((M + 1, max_meter, N + 1), dtype=np.int32)
    back_phase = np.zeros((M + 1, max_meter, N + 1), dtype=np.int32)
    candidate_index = np.arange(1, N + 1, dtype=np.int32)

    for i0 in range(M):
        meter = int(meter_of[i0])
        starts_segment = i0 in starts
        current = np.full((max_meter, N + 1), np.inf, dtype=np.float64)
        for phi in range(meter):
            if starts_segment:
                # New segment: its phase origin is free, so the match branch may
                # arrive from whatever phase the previous segment ended on.
                source = previous[:, 0:N].min(axis=0)
                source_phase = previous[:, 0:N].argmin(axis=0)
            else:
                previous_phase = (phi - 1) % meter
                source = previous[previous_phase, 0:N]
                source_phase = np.full(N, previous_phase, dtype=np.int64)
            a = source + cost_fn(i0, phi)
            accumulated = np.minimum.accumulate(a)
            is_new_minimum = a <= accumulated
            arg = np.maximum.accumulate(np.where(is_new_minimum, candidate_index, 0))
            current[phi, 0] = np.inf
            current[phi, 1:] = accumulated
            choice[i0 + 1, phi, 1:] = arg
            back_phase[i0 + 1, phi, 1:] = source_phase[np.maximum(arg - 1, 0)]
        previous = current

    final_meter = int(meter_of[M - 1])
    phi = int(np.argmin(previous[:final_meter, N]))
    if not np.isfinite(previous[phi, N]):
        raise RuntimeError("backtracking failed; cost contains non-finite values")

    sigma = np.empty(M, dtype=np.int64)
    j = N
    for i0 in range(M - 1, -1, -1):
        j_star = int(choice[i0 + 1, phi, j])
        if j_star < 1:
            raise RuntimeError("backtracking failed; cost contains non-finite values")
        sigma[i0] = j_star - 1
        phi = int(back_phase[i0 + 1, phi, j])
        j = j_star - 1
    return sigma


def subset_select_logsumexp(cost, lengths=None):
    """Equation (13) - the marginalised counterpart of the DP, log Z(theta, x)."""
    batched = cost.dim() == 3
    if not batched:
        cost = cost.unsqueeze(0)
    B, M, N = cost.shape
    device, dtype = cost.device, cost.dtype
    if M == 0:
        out = torch.zeros(B, device=device, dtype=dtype)
        return out if batched else out[0]
    if M > N:
        raise ValueError(f"infeasible: M={M} > N={N}")

    neg_inf = torch.finfo(dtype).min
    previous = torch.zeros(B, N + 1, device=device, dtype=dtype)
    for i in range(1, M + 1):
        a = previous[:, 0:N] - cost[:, i - 1]
        accumulated = torch.logcumsumexp(a, dim=1)     # running logsumexp along j
        current = torch.full((B, N + 1), neg_inf, device=device, dtype=dtype)
        current[:, 1:] = accumulated
        if lengths is not None:
            # padded steps are exact no-ops: a fragment that has consumed all M_b of its
            # events simply stops advancing, landing on the same table entry the
            # unpadded recursion would.
            current = torch.where((lengths >= i).unsqueeze(1), current, previous)
        previous = current
    out = previous[:, N]
    return out if batched else out[0]


def subset_posterior_marginals(cost):
    """The exact posterior P(sigma(i) = j | y, theta, x) over eq. (21)'s distribution."""
    M, N = cost.shape
    device, dtype = cost.device, cost.dtype
    neg_inf = torch.finfo(dtype).min

    # forward: D[i, j] over 0..M, 0..N  (D[0, j] = 0, D[i, 0] = -inf for i >= 1)
    D = torch.full((M + 1, N + 1), neg_inf, device=device, dtype=dtype)
    D[0, :] = 0.0
    for i in range(1, M + 1):
        prev = D[i - 1, 0:N] - cost[i - 1]              # take candidate j (1-indexed j)
        D[i, 1:] = torch.logcumsumexp(prev, dim=0)

    # backward: E[i, j] = mass of assigning events i+1..M to candidates j+1..N
    E = torch.full((M + 1, N + 1), neg_inf, device=device, dtype=dtype)
    E[M, :] = 0.0
    for i in range(M - 1, -1, -1):
        for j in range(N - 1, -1, -1):
            take = E[i + 1, j + 1] - cost[i, j]         # y_{i+1} -> y^_{j+1}, 0-indexed
            E[i, j] = torch.logaddexp(E[i, j + 1], take)

    log_z = D[M, N]
    w = torch.empty((M, N), device=device, dtype=dtype)
    for i in range(1, M + 1):
        w[i - 1] = D[i - 1, 0:N] - cost[i - 1] + E[i, 1:] - log_z
    return w.exp()


def joint_phase_log_partition(costs):
    """Equations (25) and (30): log Z_joint = log sum_p Z^(p)."""
    per_phase = torch.stack([subset_select_logsumexp(costs[p]) for p in range(costs.shape[0])])
    return torch.logsumexp(per_phase, dim=0), per_phase


def joint_phase_posterior(log_z_per_phase):
    """Equation (26): P(phi_0 = p | x; theta) = Z^(p) / Z_joint, in log space."""
    return log_z_per_phase - torch.logsumexp(log_z_per_phase, dim=0)


def downbeat_marginal_from_phase_posterior(log_phase_posterior, M, L):
    """Equations (27) and (14): r_i = P(C_i = DB | x; theta) = P(phi_0 = p*_i | x; theta)."""
    return log_phase_posterior[phase_star(M, L, log_phase_posterior.device)].exp()


def joint_phase_matching_marginals(costs, log_phase_posterior):
    """Equation (29): P(sigma(i) = j | x; theta), mixed over the phase posterior."""
    weights = log_phase_posterior.exp()
    return sum(weights[p] * subset_posterior_marginals(costs[p])
               for p in range(costs.shape[0]))


def meter_joint_log_partition(cost_builder, meters):
    """Equation (32): log Z_joint = log sum_{L in M} sum_p Z^(L,p)."""
    hypotheses = []
    for meter in meters:
        for p in range(meter):
            hypotheses.append((meter, p, subset_select_logsumexp(cost_builder(meter, p))))
    log_z_all = torch.stack([h[2] for h in hypotheses])
    return torch.logsumexp(log_z_all, dim=0), hypotheses


def meter_posterior(hypotheses, log_z_joint):
    """Equation (33): P(L | x; theta), by summing the joint posterior over p."""
    posterior = {}
    for meter, _p, log_z in hypotheses:
        contribution = (log_z - log_z_joint).exp()
        posterior[meter] = posterior.get(meter, 0.0) + contribution
    return posterior


def downbeat_marginal_over_meters(hypotheses, log_z_joint, M, device=None):
    """Equation (34): r_i = sum_{L in M} P(L, phi_0 = p*_i(L) | x; theta)."""
    i0 = torch.arange(M, device=device)
    r = torch.zeros(M, device=device, dtype=log_z_joint.dtype)
    for meter, p, log_z in hypotheses:
        # this hypothesis contributes to event i exactly when p == p*_i(L)
        contributes = ((-i0) % meter) == p
        r = r + torch.where(contributes, (log_z - log_z_joint).exp(),
                            torch.zeros((), device=device, dtype=r.dtype))
    return r
