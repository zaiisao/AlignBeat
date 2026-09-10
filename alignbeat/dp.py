"""The correspondence problem: Algorithm 1 (SubsetSelectDP) and its soft counterpart."""
import os

import numpy as np
import torch


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


def subset_select_dp(cost):
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

        _dp_kernel(np.ascontiguousarray(cost), choice)
        sigma = np.empty(M, dtype=np.int64)
        j = N
        for i in range(M, 0, -1):
            j_star = int(choice[i, j])
            if j_star < 1:
                raise RuntimeError(
                    "backtracking failed; cost matrix contains non-finite values")
            sigma[i - 1] = j_star - 1
            j = j_star - 1
        return sigma

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



def subset_select_logz(cost):
    """log sum_sigma exp(-cost(sigma)): the partition function over ALL order-preserving
    injections, by the same recursion as subset_select_dp with min replaced by logsumexp.

    Section 4.1's denominator. subset_select_dp returns the single cheapest sigma; this
    returns what that sigma is competing against, so the two together give section 4.2's

        log m(theta, x) = -cost(sigma_hat) - log Z

    i.e. sigma_hat's own share of the timing posterior. cost must be the TIMING cost
    alone -- P_1 is defined under timing evidence only -- so pass lambda_l1 * |t - t_hat|
    without the class term.
    """
    M, N = cost.shape
    if M == 0:
        return 0.0
    if M > N:
        return float("-inf")

    # previous[j] = log sum over ways to match the first i events into candidates < j.
    # The empty prefix has exactly one such way (the empty matching) at every j.
    previous = np.zeros(N + 1, dtype=np.float64)
    current = np.empty(N + 1, dtype=np.float64)
    for i in range(1, M + 1):
        running = -np.inf
        for j in range(N):
            # Either event i-1 takes candidate j, or it does not and j is skipped.
            running = np.logaddexp(running, previous[j] - cost[i - 1, j])
            current[j + 1] = running
        current[0] = -np.inf
        previous, current = current, previous
    return float(previous[N])
