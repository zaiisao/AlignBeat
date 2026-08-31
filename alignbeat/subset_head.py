"""Order-preserving alignment head ("Beat Tracking as Latent Order-Preserving
Alignment").

[What] A fixed set of N candidate *point* predictors. Each candidate emits a
3-way class distribution over {DB, B, none} and one raw scalar; a global
cumulative-softplus reparameterization (eq. 1) turns those scalars into a
strictly increasing time sequence. Because both the ground truth and the
candidate times are sorted by construction, deciding which candidate is
responsible for which event reduces to choosing an order-preserving subset,
solved exactly by an O(N*M) dynamic program (Alg. 1). No anchors, no NMS.

[Why the candidates cannot collapse] A DETR-style head with learned queries and
a decoder collapses. Here candidates come from Downsample(encoder features) and
their times are monotone by construction, so there is nothing to collapse and
nothing to sort: the DP indexes candidates in their natural order.

[Class index convention] 0 = downbeat, 1 = beat, 2 = background.

Equation numbers below refer to the PDF.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DOWNBEAT = 0
BEAT = 1
BACKGROUND = 2
NUM_CLASSES = 3

# Ground-truth marker for an event that is known to be a beat but whose B/DB
# distinction was never annotated (section 8; SMC is the case in practice). It is a
# label for the TARGET, never a class the network predicts - the head always emits the
# same three-way distribution over {DB, B, empty}.
CLASS_UNKNOWN = -1

# Floor for log-probabilities entering the DP cost. Once the model becomes confident,
# p(background) for some candidate underflows to 0 in float32, -log gives +inf, and the
# section 8.4 corrected cost (which SUBTRACTS gamma * that term) becomes -inf; the
# running-minimum DP then cannot backtrack and the whole batch is lost. Observed live:
# 2 failures in 11,200 iterations, both at epoch 29, i.e. it appears only once training
# has progressed and would grow more frequent. exp(-60) ~ 1e-26 is far below any
# probability that matters here, so clamping changes nothing except that the cost stays
# finite.
LOG_PROB_FLOOR = -60.0


# ---------------------------------------------------------------------------
# Prediction architecture (section 3)
# ---------------------------------------------------------------------------

class SubsetSelectionHead(nn.Module):
    """FPN levels -> N candidates -> (class logits, monotone times).

    Each FPN level is downsampled to exactly N positions by a strided conv whose
    stride is the level's own size ratio (P1: T/N, P2: (T/2)/N, P3: (T/4)/N), then
    the three N-length sequences are fused into one candidate feature z_j. This is
    the multi-scale story that motivated keeping the FPN: evidence for a beat lives
    at a different timescale depending on tempo, and each candidate should see all
    of them. Nothing in sections 4-9 depends on how z_j is produced (the paper calls
    Downsample an implementation choice orthogonal to the correspondence problem),
    so a single-level variant is a drop-in ablation - pass one feature map instead
    of three.

    Note on monotonicity: t_hat is computed from z_j only. If a candidate-level
    self-attention pass (section 10.2) is later added it must feed the *classification*
    branch only, otherwise equation (1)'s guarantee is unaffected but the argument in
    9.2 for why it is safe no longer applies.
    """

    def __init__(self, feature_size=256, num_candidates=160, level_strides=(8, 4, 2),
                 hidden_size=256, dropout=0.0, class_prior=(0.10, 0.30, 0.60),
                 class_attention_layers=0, class_attention_heads=4,
                 predict_precision=False):
        super(SubsetSelectionHead, self).__init__()
        self.num_candidates = num_candidates
        self.level_strides = tuple(level_strides)
        # N is baked into the checkpoint as a buffer because NO parameter shape
        # depends on it (the downsample convs are shaped by stride, the heads are
        # per-candidate) - so a checkpoint trained at N=160 loads cleanly into a
        # model built with N=100 under strict=True and evaluates silently wrong
        # (adversarial audit finding, reproduced). forward() compares the loaded
        # buffer against the constructed value and refuses to run on a mismatch.
        self.register_buffer('trained_num_candidates',
                             torch.tensor(num_candidates, dtype=torch.long))

        # One downsampling conv per FPN level. kernel = stride so the receptive
        # fields tile the level exactly with no overlap and no gap; each candidate
        # summarises a disjoint span of that level.
        self.downsample = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size=s, stride=s)
            for s in self.level_strides
        ])

        # LayerNorm before the heads is load-bearing, not decoration. The FPN emits
        # unnormalised features with |max| around 40, so without it the two final
        # Linear layers see large activations and produce a gradient norm of ~130
        # while the entire backbone contributes ~0 (measured). Gradient clipping then
        # scales everything by 1/130 and the encoder effectively never trains: a batch
        # of four fragments failed to overfit at all, while the same model with the
        # backbone frozen trained fine. Normalising the candidate features puts the
        # head's gradients on the same scale as everything upstream.
        self.input_norm = nn.LayerNorm(feature_size)
        self.trunk = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        # Section 9.2: a shallow self-attention pass over the N candidate features,
        # used by the CLASSIFICATION branch only.
        #
        # [why] Measured on a trained checkpoint, the mean predicted p(DB) is 0.854 on
        # candidates matched to a downbeat, 0.029 on background - both fine - but 0.141
        # on candidates matched to an ordinary BEAT. With ~3x as many beats as
        # downbeats, that tail is the entire source of the false downbeats (precision
        # 0.430). Every other explanation was eliminated by measurement: timing (14 ms
        # median residual), the decode threshold (a full 9x9 tau sweep moves Joint by
        # +0.001), the DP assignment (only 6.7% of downbeats go to a beat-preferring
        # candidate), and the class weighting (the gradient pull:push ratio on the
        # downbeat logit is 0.81, i.e. balanced). What remains is discriminability:
        # z_j summarises its own local span, and which beat of the bar this is simply
        # is not a local property - exactly the argument in section 9.1.
        #
        # [why classification only] t_hat is computed from z_j, never from z_tilde, so
        # equation (1)'s monotonicity guarantee is untouched. Section 9.2 makes the
        # same split for the same reason: placing a matched time is a local decision,
        # whereas bar position needs beat-grid-scale context.
        self.candidate_attention = None
        if class_attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=class_attention_heads,
                dim_feedforward=hidden_size * 2, dropout=dropout,
                batch_first=True, norm_first=True)
            self.candidate_attention = nn.TransformerEncoder(layer, class_attention_layers)

        self.class_head = nn.Linear(hidden_size, NUM_CLASSES)
        self.regression_head = nn.Linear(hidden_size, 1)
        # Section 4.1.2: a small head reading z_j so the model can express reduced
        # confidence in acoustically ambiguous passages rather than assuming a single
        # dataset-wide noise level. Off by default -- section 4.1's own analysis is
        # that this reopens two failure modes the shared global b closes, and the
        # mitigations of 4.1.3 (a floor by construction, a Gamma prior, a stop-gradient
        # and a warm-up) are what make it safe to enable.
        self.precision_head = nn.Linear(hidden_size, 1) if predict_precision else None

        self._initialize_weights(class_prior)

    def _initialize_weights(self, class_prior):
        # candidate_attention (section 10.2) is skipped: nn.TransformerEncoder ships its
        # own Xavier init, and sweeping every nn.Linear here overwrote out_proj/linear1/
        # linear2 with Kaiming-relu while in_proj_weight escaped (it is a raw Parameter,
        # not a Linear). Measured: weight std 0.088 vs the stock 0.036, and the block
        # multiplied the activation scale by 2.24 instead of preserving it - so the two
        # halves of the same attention block were initialised inconsistently.
        attention_modules = set()
        if self.candidate_attention is not None:
            attention_modules = {id(m) for m in self.candidate_attention.modules()}
        for m in self.modules():
            if id(m) in attention_modules:
                continue
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero the regression output so every r_k starts equal: softplus(0) is the
        # same positive constant for all k, so equation (1) gives t_hat_j = j/N at
        # initialization - candidates uniformly spread over the window, which is the
        # best possible starting point and costs nothing to arrange.
        nn.init.zeros_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

        # Same treatment for the classifier, for the same reason a detector sets a
        # focal-loss prior on its final layer. Left at a plain Kaiming init the logits
        # start at magnitude ~35, i.e. near-certain and arbitrary, so the background
        # term of loss (8) opens in the hundreds and step 0 is a large meaningless
        # gradient. Zero weights plus a log-prior bias makes every candidate start at
        # the class prior instead: with N = 160 candidates and typically 60-70 real
        # events per fragment, roughly 60% of candidates should predict background.
        nn.init.zeros_(self.class_head.weight)
        prior = torch.tensor(class_prior, dtype=torch.float32)
        self.class_head.bias.data = torch.log(prior / prior.sum())

    def forward(self, features):
        """features: list of (B, C, T_l) FPN maps, finest first.

        Returns class_logits (B, N, 3) and t_hat (B, N), strictly increasing in j.
        """
        if len(features) != len(self.downsample):
            raise ValueError(
                f"expected {len(self.downsample)} feature levels, got {len(features)}")
        if int(self.trained_num_candidates) != self.num_candidates:
            raise RuntimeError(
                f"checkpoint was trained with num_candidates="
                f"{int(self.trained_num_candidates)} but this model was built with "
                f"{self.num_candidates}; pass --num_candidates "
                f"{int(self.trained_num_candidates)} (no parameter shape depends on N, "
                f"so this mismatch would otherwise evaluate silently wrong)")

        pooled = None
        for level, (feature, downsample, stride) in enumerate(
                zip(features, self.downsample, self.level_strides)):
            # Check the INPUT length, not the output length. A strided conv silently
            # drops the remainder - length 1281 with stride 8 still yields 160
            # candidates while quietly discarding the last frame - so validating the
            # output would never fire. The fragment length must divide exactly.
            expected = self.num_candidates * stride
            if feature.size(dim=2) != expected:
                raise ValueError(
                    f"FPN level {level} has length {feature.size(dim=2)}, expected "
                    f"{expected} (= num_candidates {self.num_candidates} * stride "
                    f"{stride}). A strided conv would silently truncate the remainder.")
            z = downsample(feature)  # (B, C, N)
            pooled = z if pooled is None else pooled + z

        z = pooled.transpose(1, 2)  # (B, N, C)
        z = self.trunk(self.input_norm(z))

        # Regression reads z directly and is therefore unaffected by section 10.2's
        # attention pass; only the classifier sees the contextualised features.
        r = self.regression_head(z).squeeze(dim=2)      # (B, N)
        t_hat = monotonic_times(r)

        z_class = z if self.candidate_attention is None else self.candidate_attention(z)
        class_logits = self.class_head(z_class)         # (B, N, 3)

        if self.precision_head is None:
            return class_logits, t_hat
        # Raw output u_j; the criterion applies b_j = b_min + softplus(u_j).
        return class_logits, t_hat, self.precision_head(z).squeeze(dim=2)


def monotonic_times(r, epsilon=1e-4):
    """Equation (1): t_hat_j = sum_{k<=j} softplus(r_k) / sum_{k<=N} softplus(r_k).

    Every summand is strictly positive, so in exact arithmetic t_hat is strictly
    increasing for any r and lands in (0, 1]. Monotonicity is an architectural
    invariant - the network cannot represent a crossing hypothesis - which is what
    reduces section 4's correspondence problem to a subset choice, and what makes NMS
    unnecessary at inference.

    [Finite precision] The cumulative sum is what actually decides ordering, and in
    float32 an increment smaller than the ULP of the running total leaves it
    unchanged, producing a tie. This needs a pathological spread in r (measured:
    ~46/159 ties once r ~ N(0, 20), none at realistic scales) but it is worth being
    precise about what survives: the sequence is *non-decreasing* in any precision,
    because it is a cumulative sum of positive numbers. Non-crossing - the property
    the no-NMS argument actually rests on - therefore always holds. Only strictness
    can degrade, and its only consequence is two detections reported at an identical
    time rather than out of order. SubsetCriterion reports min_gap so a run drifting
    toward that regime is visible rather than silent.

    The floor is *relative* to the mean increment, which is what makes strictness
    hold rather than merely being hoped for. After normalisation the increments are
    weights w_k summing to 1 with mean 1/N, so strict increase in float32 needs
    min(w) above the ULP near 1.0 (about 1.2e-7), i.e. a min/mean ratio above
    N * 1.2e-7 ~ 2e-5 at N = 160. Flooring at epsilon = 1e-4 of the mean clears that
    by 5x while perturbing a healthy increment by 0.01%. An absolute floor is not
    enough: whether 1e-6 is large or small depends entirely on how big Z has grown.

    A tiny absolute term additionally keeps the degenerate case graceful - if every
    softplus(r_k) underflows, all increments become equal and the normalisation
    returns the uniform grid j/N rather than 0/0.

    Note t_hat[..., -1] is exactly 1.0 by construction: the last candidate is pinned
    to the end of the window. If it goes unmatched it simply takes the background
    loss, so this costs a candidate slot rather than correctness.
    """
    # Bound r before softplus. softplus(r) ~ r for large r, so one extreme value makes
    # the cumulative sum overflow to inf and the normalisation return inf/inf = NaN.
    # That NaN reaches the DP cost through the lambda_L1 * |t - t_hat| term, which
    # LOG_PROB_FLOOR does not cover, and the batch is lost. Observed live: subset
    # training silently stopped at epoch ~74 after 9,749 consecutive 'backtracking
    # failed' skips, with the saved weights themselves still perfectly finite.
    # +-30 is far outside any healthy range (softplus(30) = 30, so the cumulative sum
    # stays under 160*30); a model sitting at r = 30 is already broken, so the
    # saturated gradient there costs nothing.
    r = r.clamp(-30.0, 30.0)
    increments = F.softplus(r)
    relative_floor = epsilon * increments.mean(dim=-1, keepdim=True)
    increments = increments + relative_floor + 1e-30
    cumulative = torch.cumsum(increments, dim=-1)
    total = cumulative[..., -1:].clamp(min=1e-30)
    return cumulative / total


# ---------------------------------------------------------------------------
# The correspondence problem (sections 4-5)
# ---------------------------------------------------------------------------

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
        """Recursion (7) as a compiled scalar loop.

        Tie-breaking must match the numpy path exactly. numpy does
            accumulated = minimum.accumulate(a)
            is_new = a <= accumulated
            arg = maximum.accumulate(where(is_new, index, 0))
        i.e. the LATEST index attaining the running weak minimum. The scalar form of
        that is `if a <= best` (weak inequality), which updates on ties.
        """
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
    """Algorithm 1 - exact O(N*M) order-constrained subset selection.

    cost: (M, N) numpy array of per-pair costs, ground truth by candidate, both in
    ascending time order. Returns sigma, an (M,) int array of candidate indices,
    strictly increasing, minimising sum_i cost[i, sigma(i)] over all order-preserving
    injections (Definition 1 / equation 4).

    Recursion (7): D[i, j] = min(D[i, j-1], D[i-1, j-1] + cost[i, j]).

    The inner loop over j is written as a running minimum rather than a Python loop:
    for a fixed i, D[i, j] = min_{j' <= j} (D[i-1, j'-1] + cost[i, j']), which is
    exactly np.minimum.accumulate over that expression. That turns an O(N*M) Python
    loop (~25k iterations per sample at N=160) into M vectorised numpy ops.
    """
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
    """Section 4.2, equation (6): known-meter spacing folded into the SELECTION.

    (6) penalises the gap between consecutive matched downbeats against L * Delta_bar.
    It depends on TWO matched positions jointly, breaking recursion (7)'s first-order
    structure, so the most recent downbeat's matched candidate is carried as extra
    state: D[i, j_star, j]. j_star = N means "no downbeat matched yet". mu = 0 reduces
    exactly to subset_select_dp, and exactness is checked against brute-force
    enumeration in tests/test_regressions.py.

    Complexity, stated honestly. The paper derives O(N^2 M / L) on the grounds that
    j_star need only be tracked BETWEEN consecutive downbeat indices, so the O(N^2)
    work is incurred once per inter-downbeat segment rather than once per event. This
    implementation does not realize that saving: the (N+1)-wide j_star axis is carried
    through every event, giving O(N^2 M). The array work is fully vectorised (the
    O(N) Python inner loop this used to run at every i is gone), so the constants are
    now reasonable, but the asymptotic saving would need the intermediate events of a
    segment collapsed into a transfer matrix computed once per segment -- a genuine
    restructure, not a constant-factor tidy, and one whose exactness would need its own
    proof. Left undone deliberately rather than claimed.

    This is the only mechanism in the paper that can change WHICH candidates are
    selected; 9.4's regulariser and 7.1's imputation act after sigma_hat is fixed.
    """
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


# ---------------------------------------------------------------------------
# Phase (sections 8, 8.1, 8.3, 8.5, 8.6)
# ---------------------------------------------------------------------------
#
# One fact underpins every construction below: phase propagation is deterministic.
# Fixing phi_0 fixes the whole sequence, phi_i = (phi_0 + i - 1) mod L, because the
# increment counts MATCHED EVENTS, never candidates -- an unmatched candidate is
# simply not counted, exactly as CTC's blank contributes nothing to its own
# alignment. There is no branching latent past phi_0, which is what makes joint
# treatment of (sigma, phi_0) cost a constant factor L rather than a new algorithm.


def event_is_downbeat_under(M, L, p, device=None):
    """Which of the M matched events are downbeats under the hypothesis phi_0 = p.

    Event i (1-indexed) carries phase (p + i - 1) mod L, and is a downbeat exactly
    when that is 0. Returned 0-indexed, so entry i0 is event i0 + 1.
    """
    i0 = torch.arange(M, device=device)
    return ((p + i0) % L) == 0


def phase_star(M, L, device=None):
    """p*_i = (1 - i) mod L: the unique hypothesis under which event i is a downbeat.

    Section 8 verifies this is a singleton rather than a general subset -- for fixed
    i and L exactly one p satisfies (p + i - 1) mod L = 0 -- which is what makes
    r_i a direct lookup of the phase posterior (eq. 14, 27, 34) rather than a sum.
    """
    i0 = torch.arange(M, device=device)
    return (-i0) % L


def phase_class_nll(log_probabilities, M, L, p):
    """The class term of eq. (18): -log p_hat_j(phi_j = (p + i - 1) mod L | x; theta).

    Section 8's phase-valued query on the classification head: phase 0 reads the DB
    probability, every nonzero phase reads the same B probability, since the head
    only ever distinguishes downbeat from non-downbeat and never one non-downbeat
    phase from another.

    log_probabilities: (N, 3). Returns (M, N).
    """
    is_db = event_is_downbeat_under(M, L, p, log_probabilities.device)
    downbeat = -log_probabilities[:, DOWNBEAT]                            # (N,)
    beat = -log_probabilities[:, BEAT]                                    # (N,)
    return torch.where(is_db[:, None], downbeat[None, :], beat[None, :])


def subset_select_dp_joint_phase(costs):
    """Equation (19): the joint MAP of (sigma, phi_0), by L reruns of Algorithm 1.

    Once phi_0 = p is fixed, every event's hypothesized phase is determined, so
    L^p_match is an ordinary per-pair cost of exactly the shape Algorithm 1 already
    minimises. Run it once per hypothesis and keep whichever run achieved the lowest
    D^(p)[M, N], together with the matching that run produced.

    This is what makes the coupling genuine rather than cosmetic: sigma is re-derived
    under every hypothesis, so a matching that looks cheap under one phase but
    metrically implausible under the phase that wins is never the one reported. A
    candidate whose class prediction favours DB is a cheap match for an event one
    hypothesis calls a downbeat and an expensive match for the same event under a
    hypothesis that calls it a beat -- so sigma genuinely differs across p.

    costs: (L, M, N) array of phase-conditioned costs. Returns (sigma, p_hat).
    Cost O(L * N * M), a constant-factor increase over the phase-blind O(N * M).
    """
    costs = np.asarray(costs, dtype=np.float64)
    L = costs.shape[0]
    best_sigma, best_p, best_cost = None, 0, np.inf
    for p in range(L):
        sigma, total = subset_select_dp(costs[p], return_cost=True)
        if total < best_cost:
            best_sigma, best_p, best_cost = sigma, p, total
    return best_sigma, best_p


def subset_select_dp_phase_segments(cost_fn, M, N, segments):
    """Equation (20): the mixed-meter joint DP, phase carried as augmented state.

    Section 8.1's segments do not reduce to independent reruns the way a single
    meter does, because order-preservation is a global constraint: which candidates
    remain available to segment k+1 depends on where segment k's matching ended. So
    the phase is carried as extra state instead, generalizing Section 4.2's own
    augmented-state device.

        D[i, j, phi] = min( D[i, j-1, phi],
                            D[i-1, j-1, phi'] + cost(i, j-1, phi) )

    where phi' is the unique value with (phi' + 1) mod L_k = phi when event i is not
    the first of its segment (propagation within a segment is deterministic, so there
    is genuinely one predecessor, not a minimum over several), and min over all phi'
    when i starts a new segment, whose phase origin is chosen independently of
    whatever phase the previous segment happened to end on.

    cost_fn(i0, phi) -> (N,) array: the cost of matching event i0 (0-indexed) to each
    candidate, given that event's own phase is phi.
    segments: list of (start_event_index, L_k), ascending, first starting at 0.
    Returns sigma. Cost O(N * M * L_max).
    """
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
    """Equation (13) - the marginalised counterpart of the DP, log Z(theta, x).

    Same recursion with min replaced by logsumexp on the negated cost, giving the
    total probability mass over every order-preserving injection instead of the
    single cheapest one (the forward algorithm to Algorithm 1's Viterbi). Provided
    for the section 8.4 training mode; not used by the default hard path. Operates
    on torch tensors because, unlike Algorithm 1, this one is differentiated through.
    """
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
    """The exact posterior P(sigma(i) = j | y, theta, x) over eq. (21)'s distribution.

    The direct --marginal path never needs this -- it differentiates -log Z directly --
    but section 8.4's equivalent E-step/M-step perspective does, it is the only way to
    SEE the soft correspondence rather than infer it, and eq. (29) mixes it over the
    phase posterior to get the fully joint matching marginal. Needs the backward
    companion (28) to the forward recursion (22):

        E~[i, j] = logsumexp( E~[i, j+1], E~[i+1, j+1] - L'match(y_{i+1}, y^_{j+1}) )
        E~[M, j] = 0,   E~[i, N] = -inf for i < M

    filled by decreasing i and j, same O(N M) cost. Combining forward, edge and backward:

        w_ij = exp( D~[i-1, j-1] - L'match(y_i, y^_j) + E~[i, j] - D~[M, N] )

    cost: (M, N) corrected cost. Returns (M, N) posterior; rows sum to 1.
    """
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
    """Equations (25) and (30): log Z_joint = log sum_p Z^(p).

    For fixed p the inner sum over sigma in eq. (24) is exactly Section 8.4's own
    partition function with L'^p_match substituted, so the joint object needs no new
    machinery: run the log-sum-exp recursion (22) once per hypothesis and add the
    results in log space. This is the soft analogue of eq. (19) -- there the joint MAP
    was the MINIMUM of L hard costs, here the joint marginal likelihood is the SUM of
    L soft partition functions, the same Viterbi/forward relationship one level up.

    costs: (L, M, N) tensor of phase-conditioned corrected costs, differentiable.
    Returns (log_z_joint, log_z_per_phase) with log_z_per_phase of shape (L,).
    Every quantity is differentiable; eq. (30) needs no stop-gradient anywhere.
    """
    per_phase = torch.stack([subset_select_logsumexp(costs[p]) for p in range(costs.shape[0])])
    return torch.logsumexp(per_phase, dim=0), per_phase


def joint_phase_posterior(log_z_per_phase):
    """Equation (26): P(phi_0 = p | x; theta) = Z^(p) / Z_joint, in log space."""
    return log_z_per_phase - torch.logsumexp(log_z_per_phase, dim=0)


def downbeat_marginal_from_phase_posterior(log_phase_posterior, M, L):
    """Equations (27) and (14): r_i = P(C_i = DB | x; theta) = P(phi_0 = p*_i | x; theta).

    No forward-backward pass is needed. The hypothesized phase (p + i - 1) mod L
    depends only on (p, i), never on which candidate sigma assigns to event i, so
    sigma was never part of what determines a hypothesis's own downbeat pattern --
    and since {p : (p + i - 1) mod L = 0} is the singleton {p*_i}, event i's downbeat
    marginal is a direct lookup rather than a sum.
    """
    return log_phase_posterior[phase_star(M, L, log_phase_posterior.device)].exp()


def joint_phase_matching_marginals(costs, log_phase_posterior):
    """Equation (29): P(sigma(i) = j | x; theta), mixed over the phase posterior.

    Unlike r_i, the MATCHING marginal genuinely couples to sigma, so it needs a real
    forward-backward pass (eq. 28) within each phase hypothesis, mixed afterwards by
    eq. (26)'s own posterior weights.

    costs: (L, M, N). Returns (M, N); rows sum to 1.
    """
    weights = log_phase_posterior.exp()
    return sum(weights[p] * subset_posterior_marginals(costs[p])
               for p in range(costs.shape[0]))


def meter_joint_log_partition(cost_builder, meters):
    """Equation (32): log Z_joint = log sum_{L in M} sum_p Z^(L,p).

    Section 8.6 removes the last standing assumption -- that L is known external
    metadata. L belongs in the same category as sigma and phi_0, a genuine unobserved
    quantity, and is marginalized the same way: one log-sum-exp recursion per (L, p)
    pair, summed in log space. No prior over L is introduced; every triple is weighted
    by its own cost alone, an implicitly uniform prior over the candidate set, with all
    discrimination between meters coming from how well each one's phase-consistent
    class predictions fit the data.

    cost_builder(L, p) -> (M, N) differentiable cost.
    meters: iterable of candidate meter lengths, e.g. range(2, 13).
    Returns (log_z_joint, hypotheses) where hypotheses is a list of
    (L, p, log_z) triples covering every pair.
    Cost O(N * M * sum_L L) -- 77 recursions for M = {2, ..., 12}.
    """
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
    """Equation (34): r_i = sum_{L in M} P(L, phi_0 = p*_i(L) | x; theta).

    For each fixed L the inner sum over p is again the singleton lookup at
    p*_i(L) = (1 - i) mod L -- now depending on L, since which phase makes event i a
    downbeat shifts with the hypothesized meter -- leaving only the outer sum over
    candidate meters genuine.
    """
    i0 = torch.arange(M, device=device)
    r = torch.zeros(M, device=device, dtype=log_z_joint.dtype)
    for meter, p, log_z in hypotheses:
        # this hypothesis contributes to event i exactly when p == p*_i(L)
        contributes = ((-i0) % meter) == p
        r = r + torch.where(contributes, (log_z - log_z_joint).exp(),
                            torch.zeros((), device=device, dtype=r.dtype))
    return r


# ---------------------------------------------------------------------------
# Training loss (sections 4, 7)
# ---------------------------------------------------------------------------

class SubsetCriterion(nn.Module):
    """Per-pair cost (3), the selection DP, and the training loss (8).

    lambda_l1 = 1/b is the precision of the Laplace time-observation model in (2),
    not a free balancing weight. b defaults to a fixed scalar; enable learn_b to use
    the closed-form MLE (5), the mean absolute matched residual, held fixed while
    theta is updated (block coordinate ascent, not EM).

    Scale warning worth watching: times are normalised to (0, 1) over the whole
    window, so the +-70 ms evaluation tolerance is only 70ms/D ~ 0.0024 in these
    units. With b at that scale lambda_l1 is in the hundreds while -log p is order 1,
    and the selection collapses to nearest-in-time regardless of class. The returned
    stats include cost_class_mean / cost_time_mean so this is visible from epoch 1.
    """

    def __init__(self, b_scale=0.005, gamma=0.5, omega_downbeat=2.0, omega_beat=1.0,
                 learn_b=False, b_momentum=0.9, b_min=1e-4, normalize_by_events=True,
                 tol_flat=0.0,
                 diagnostic_every=200, beat_only_warmup=2000, beat_only_confidence=0.7,
                 cont_weight=0.0, cont_windows=8, lambda_r=0.0, meter_length=0,
                 marginal=False, marginal_background=True, fragment_seconds=29.7215,
                 mu_meter=0.0,
                 joint_phase=False, marginal_meters=(),
                 precision_warmup=2000, precision_prior_alpha=2.0,
                 precision_prior_beta=None):
        super(SubsetCriterion, self).__init__()
        self.gamma = gamma
        # cont_weight > 0: penalise the variance of the log expected event count across
        # sub-windows of the fragment. Non-separable over candidates, so unlike the
        # per-candidate background term it can charge for structure rather than count.
        # NOTE: invariant to a GLOBAL doubling (log 2n = log n + const), so it targets
        # tempo INSTABILITY, not octave errors. Never trained yet.
        self.cont_weight = cont_weight
        self.cont_windows = cont_windows
        # Section 10.4, equations (36)-(37). Penalises deviation of consecutive predicted
        # DOWNBEAT spacing from L beat periods, where the beat period is estimated
        # differentiably from the model's own matched candidates:
        #     Delta_bar = mean_i ( t_sigma(i+1) - t_sigma(i) )  over all M matched  (36)
        #     R = sum_k ( t_sigma(i_{k+1}) - t_sigma(i_k) - L * Delta_bar )^2       (37)
        # Added after sigma_hat is fixed (it depends on the whole matched downbeat
        # sequence, so it cannot be a per-pair term in L_match). lambda_r = 0 is off.
        # meter_length = 0 derives L per fragment from the ground truth as the median
        # number of events between consecutive downbeats - the paper assumes L known
        # at dataset/track level, and the annotations already carry it (measured:
        # ballroom 3.98, harmonix 4.03, carnatic 6.01 beats per bar).
        self.lambda_r = lambda_r
        self.meter_length = meter_length
        # Section 8.4: train on the marginal likelihood over EVERY order-preserving
        # injection instead of one hard-selected sigma. Writing g_j = -log p_j(empty),
        # the full loss of section 7 is
        #     sum_i L_match(i, sigma(i)) + gamma * sum_{j not in im(sigma)} g_j
        #   = sum_i [ L_match(i, sigma(i)) - gamma * g_sigma(i) ] + gamma * sum_j g_j
        # and the bracketed quantity is exactly build_cost's `corrected`. So the
        # marginal loss (23) is -log Z over that corrected cost, plus the same
        # sigma-independent gamma * sum_j g_j. No stop-gradient: unlike the hard path,
        # every quantity here is differentiable.
        # Measured motivation (temp_wide, val): the posterior over sigma is a near
        # point mass on ballroom (~1.2 effective assignments) but broad on carnatic
        # (~18, no song deterministic), so hard EM commits arbitrarily on exactly the
        # dataset where we are weakest.
        self.marginal = marginal
        # Marginalising the FULL loss (8) gives gamma*sum_j g_j - log Z', because
        # sum_{j not in im(sigma)} g_j = sum_j g_j - sum_i g_sigma(i) and the second half
        # is already folded into build_cost's `corrected`. The paper's equation (23) is
        # -log Z alone; with the term omitted, build_cost still subtracts gamma*g_j and
        # nothing compensates, so dL/dg_j <= 0 for every j and the objective is unbounded
        # below. Default to the exact marginalisation; set False for (14) literally.
        self.marginal_background = marginal_background
        # t_hat is normalised to (0, 1] within the fragment: a residual is a FRACTION of
        # the fragment, not milliseconds. residual*1000 printed 0.0008 as "1ms" when it
        # is 0.0008 * 29.72 s = 24 ms - the figure compared against mir_eval's +-70 ms.
        self.fragment_seconds = fragment_seconds
        # Section 4.2: weight of equation (6) inside the SELECTION. 0 = plain Algorithm 1.
        self.mu_meter = mu_meter
        # Section 8.3, equation (19): infer (sigma, phi_0) JOINTLY rather than
        # sequentially. The sequential decomposition of section 8.2 has two named
        # failure modes, both tracing to the same cause -- L_match (3) is evaluated
        # before any phase hypothesis exists, so it cannot use one. A candidate that is
        # a middling match by class and timing alone but would sit at a metrically
        # implausible phase has no way to be penalised for that; and once sigma_hat is
        # fixed, no later phase evidence can revise it. Running Algorithm 1 once per
        # hypothesis and keeping the cheapest run resolves both, at O(L * N * M) --
        # a bounded constant-factor cost, not an asymptotic one. Requires a meter.
        self.joint_phase = joint_phase
        # Section 8.6: treat the meter itself as latent, marginalizing over a finite
        # candidate set M = {2, ..., L_max} alongside sigma and phi_0. Empty tuple
        # keeps L fixed external metadata (sections 8 through 8.5). The cost rises to
        # O(N * M * sum_L L) -- 77 recursions for {2, ..., 12}, a larger but still
        # tractable constant. No prior over L is introduced: all discrimination comes
        # from how well each meter's phase-consistent class predictions fit the data.
        self.marginal_meters = tuple(marginal_meters)
        # Section 4.1.3, the three mitigations for per-candidate precision. Each closes
        # one specific way b_j can be used to cheat rather than to localise:
        #
        #   (1) b_j = b_min + softplus(u_j) bounds b_j away from zero BY CONSTRUCTION,
        #       the same device equation (1) uses for monotonicity, rather than hoping a
        #       penalty suppresses collapse. Without it, a candidate that already fits
        #       well can drive b_j toward 0 and reduce its loss without bound.
        #   (2) A Gamma prior on the precision 1/b_j -- the conjugate prior for a
        #       Laplace scale -- adds a MAP term pulling b_j toward a data-informed
        #       default, damping runaway inflation on hard candidates rather than only
        #       bounding it. precision_prior_beta defaults to the shared b, so the
        #       default IS what the data says.
        #   (3) A stop-gradient on b_j when computing dL/dt_hat_j, so one step cannot
        #       simultaneously widen b_j and leave t_hat_j unimproved; b_j is updated
        #       only through its own separate term.
        #
        # A warm-up on a fixed global b completes the set: it forces accurate
        # localisation to be learned before the network has an uncertainty channel
        # available to substitute for it.
        self.precision_warmup = precision_warmup
        self.precision_prior_alpha = precision_prior_alpha
        self.precision_prior_beta = precision_prior_beta
        # Printed from inside the criterion rather than returned to train.py because
        # under DataParallel the forward runs on a replica, so an attribute set here
        # never reaches the parent module.
        self.diagnostic_every = diagnostic_every
        self._call_count = 0
        # Diagnostic-only stats are recomputed only on steps that will actually print
        # them; every one costs a GPU->CPU sync, and pl_module discards `stats`
        # entirely. Off-steps reuse the last computed values so logging still reads
        # something meaningful rather than zeros.
        self._diag_cache = {'cost_class_mean': 0.0, 'class_spread': 0.0,
                            'residual_mean': 0.0, 'min_gap': float('inf')}
        # Section 8.4 confirmation-bias mitigations: train beat-only events on the
        # marginal (9) alone for the first `beat_only_warmup` steps so the shared head
        # first learns B vs DB from fully-labelled data, and only trust the pseudo-label
        # (11) when the head is actually decided.
        self.beat_only_warmup = beat_only_warmup
        self.beat_only_confidence = beat_only_confidence
        self.omega_downbeat = omega_downbeat
        self.omega_beat = omega_beat
        self.learn_b = learn_b
        self.b_momentum = b_momentum
        self.b_min = b_min
        self.normalize_by_events = normalize_by_events
        # Flat-bottomed time loss: residuals below tol_flat (in NORMALISED window units)
        # cost nothing and produce NO gradient.
        #
        # Why: d|x|/dx = sign(x) is scale-free, so the time channel pushes with constant
        # magnitude however small the error -- it never converges, it dithers. Measured:
        # the N=T memorisation probe reached F 0.974 at step 1000 then fell back to 0.883
        # with the loss RISING (0.141 -> 0.382), while N=188 variants held at ~1.000.
        # Meanwhile the evaluation metric is flat within +/-70 ms, so every unit of
        # gradient spent below that tolerance is spent in a direction where the true risk
        # has zero derivative. Setting tol_flat to the tolerance aligns the surrogate's
        # curvature with the risk it stands in for.
        self.tol_flat = tol_flat
        # buffer so it round-trips through checkpoints with the model
        self.register_buffer('b', torch.tensor(float(b_scale)))

        # Print the settings that actually reached this object. Four separate flags have
        # silently failed to arrive here (cont_weight via a zeroed loss slot, 9.2's
        # attention default, lambda_r at an inert scale, mu_meter never plumbed through
        # model_module), each time producing a full training run that read as a clean
        # negative result. Cheap insurance; placed last so every field exists.
        print(f"[subset-criterion] gamma={self.gamma} omega_db={self.omega_downbeat} "
              f"b={float(self.b):.5f} learn_b={self.learn_b} marginal={self.marginal} "
              f"marginal_bg={self.marginal_background} lambda_r={self.lambda_r} "
              f"cont_weight={self.cont_weight} mu_meter={self.mu_meter} "
              f"joint_phase={self.joint_phase} "
              f"marginal_meters={self.marginal_meters or 'off'} "
              f"normalize_by_events={self.normalize_by_events}", flush=True)


    @property
    def lambda_l1(self):
        return 1.0 / float(self.b.clamp(min=self.b_min))

    def _periodicity_term(self, matched_times, event_classes):
        """Equation (17). matched_times = t_hat[sigma] in event order, (M,)."""
        if matched_times.numel() < 3:
            return None
        downbeat_positions = (event_classes == DOWNBEAT).nonzero(as_tuple=False).flatten()
        if downbeat_positions.numel() < 2:
            return None                                  # no consecutive downbeat pair
        delta_bar = (matched_times[1:] - matched_times[:-1]).mean()
        if self.meter_length > 0:
            L = float(self.meter_length)
        else:
            gaps = (downbeat_positions[1:] - downbeat_positions[:-1]).float()
            if not bool((gaps >= 1.0).all()):
                return None
            # Per-pair, not one median for the fragment: a fragment can span a metre
            # change or start mid-bar, and a single median then charges every pair
            # against the wrong target (measured on harmonix, 4.1x of the residual at
            # the GROUND-TRUTH times was this artifact).
            L = gaps
        predicted = matched_times[downbeat_positions]
        L_vec = L if torch.is_tensor(L) else torch.full_like(predicted[1:], float(L))
        residual = (predicted[1:] - predicted[:-1]) - L_vec * delta_bar
        return (residual ** 2).sum()

    def _continuity_term(self, log_probabilities_b, t_hat_b):
        """Var_w( log sum_{j in w} q_j ), q_j = 1 - p_j(background).

        Under a constant tempo the expected number of emitted events per sub-window is
        constant, so the variance is a tempo-consistency penalty that needs no tempo
        label. It is not separable over candidates, which is exactly what lets it charge
        for a coherent doubled pulse train."""
        q = 1.0 - log_probabilities_b[:, BACKGROUND].exp()
        span_end = float(t_hat_b.detach().max())
        if not (span_end > 0.0):
            return q.sum() * 0.0
        W = int(self.cont_windows)
        index = (t_hat_b.detach() / span_end * W).long().clamp(0, W - 1)
        counts = torch.zeros(W, device=q.device, dtype=q.dtype).index_add_(0, index, q)
        log_counts = torch.log(counts + 1e-3)
        return ((log_counts - log_counts.mean()) ** 2).mean()

    def build_cost(self, log_probabilities, t_hat, event_classes, event_times):
        """Per-pair cost (3) plus the section 8.4 background correction.

        (3) is L_match = -log p_j(c_i) + lambda_l1 * |t_i - t_hat_j|, term for term the
        negative log-likelihood of the observation model (2). The paper is explicit
        that this is -log p_j, not the -p_j inherited from the DETR matching-cost
        literature.

        The corrected cost L'_match = L_match - gamma * g_j subtracts the background
        term a candidate would otherwise have paid. Section 8.4 shows the full loss
        (8) contains gamma * sum_{j not in im(sigma)} g_j, and since |im(sigma)| = M is
        fixed, minimising (8) over sigma is minimising sum_i [L_match - gamma *
        g_sigma(i)]. Without this subtraction the DP minimises only part of the loss it
        is supposed to be selecting for - a real gap in the hard formulation, not only
        in the marginalised one.

        Returns the corrected cost (M, N) plus the two raw terms for diagnostics.
        """
        # The DP needs a FINITE cost to backtrack, so the floor lives here rather than on
        # the log-probabilities used by the loss (clamping there zeroes the gradient of
        # any confidently-wrong candidate). Floored only for selection; the loss sees the
        # ungated values.
        floored = log_probabilities.clamp(min=LOG_PROB_FLOOR)
        class_cost = self.class_nll(floored, event_classes)                     # (M, N)
        time_cost = self.lambda_l1 * (event_times[:, None] - t_hat[None, :]).abs()
        background = -floored[:, BACKGROUND]                                    # (N,)
        corrected = class_cost + time_cost - self.gamma * background[None, :]
        return corrected, class_cost, time_cost

    def build_phase_cost(self, log_probabilities, t_hat, event_classes, event_times,
                         meter, p):
        """Equation (18), and its section 8.5 background-corrected form L'^p_match.

        Generalizes build_cost by conditioning the CLASS term on a phase hypothesis:
        candidate j's cost against event i reads the probability of the phase j would
        carry were it matched to i under hypothesis p, rather than of an observed
        class. This is what lets the DP prefer a candidate that is metrically
        consistent with the pattern the other events establish.

        Only events whose label was never observed use the phase-conditioned term. For
        a fully-labeled event c_i is observed directly rather than hypothesized, so
        equation (3) is used unchanged and L^p_match plays no role, exactly as section
        8.3 specifies. A fragment mixing both kinds therefore gets each event scored
        against whatever its annotation actually supports.
        """
        M = event_times.shape[0]
        phase_cost = phase_class_nll(log_probabilities, M, meter, p)          # (M, N)
        observed_cost = self.class_nll(log_probabilities, event_classes)      # (M, N)
        unknown = (event_classes == CLASS_UNKNOWN)[:, None]
        class_cost = torch.where(unknown, phase_cost, observed_cost)

        time_cost = self.lambda_l1 * (event_times[:, None] - t_hat[None, :]).abs()
        background = -log_probabilities[:, BACKGROUND]
        return class_cost + time_cost - self.gamma * background[None, :]

    def _fragment_meter(self, event_classes):
        """The meter L in force for this fragment, or 0 if none is available.

        Explicit metadata wins. Failing that, derive it from the annotation as the
        median number of events between consecutive downbeats -- the paper assumes L
        known at dataset or track level, and a fully-labeled annotation already
        carries it. A beat-only fragment has no downbeats to measure, so it has no
        fallback and depends on --meter_L being set.
        """
        if self.meter_length > 0:
            return int(self.meter_length)
        positions = (event_classes == DOWNBEAT).nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            return 0
        return int(np.median(np.diff(positions.cpu().numpy())))

    @staticmethod
    def class_nll(log_probabilities, event_classes):
        """-log p_j(c_i) for every (event, candidate) pair, handling unlabelled classes.

        For a normally annotated event this is just the log-probability of its class.
        For a beat-only event (CLASS_UNKNOWN, section 8) the class is known to lie in
        {B, DB} but which one was never observed, so the likelihood of what is ACTUALLY
        known is the marginal over that superset - equation (9):

            l_i = -log( p_j(B) + p_j(DB) ) = -log( 1 - p_j(empty) )

        This is the exact likelihood of the observation "this candidate is some kind of
        beat", not an approximation. Note what it cannot do (the paper is explicit):
        its gradient with respect to any redistribution of mass between B and DB that
        holds their sum fixed is exactly zero, so on its own it teaches nothing about
        telling downbeats from beats - it only pushes the candidate away from
        background. Section 8's EM over the shared phase latent (equations 12-15) is what recovers
        that signal, and SubsetCriterion applies it once warm-up has passed.
        """
        unknown = event_classes == CLASS_UNKNOWN
        safe = torch.where(unknown, torch.zeros_like(event_classes), event_classes)
        cost = -log_probabilities[:, safe].transpose(0, 1)                      # (M, N)
        if bool(unknown.any()):
            # log(1 - p(empty)) computed as logsumexp over the two active classes so it
            # stays stable when p(empty) approaches 1.
            active = torch.logsumexp(
                log_probabilities[:, [DOWNBEAT, BEAT]], dim=-1)                 # (N,)
            cost = torch.where(unknown[:, None], (-active)[None, :], cost)
        return cost

    def forward(self, class_logits, t_hat, targets, raw_precision=None):
        """class_logits (B, N, 3), t_hat (B, N), targets: list of B dicts with
        'classes' (M_b,) long and 'times' (M_b,) float in [0, 1].

        Returns (losses, stats) where losses is a dict of differentiable tensors
        ('class', 'time', 'background', 'total') and stats is floats for logging.
        The components are kept separate because train.py unpacks a five-tuple of
        losses per iteration and logs each one.
        """
        batch_size, num_candidates, _ = class_logits.shape
        # Ungated log-probabilities: the loss must receive the full cross-entropy
        # gradient. clamp(min=LOG_PROB_FLOOR) has ZERO gradient below the floor, so a
        # confidently WRONG candidate received exactly no class gradient instead of the
        # maximal one -- the opposite of what (8) prescribes. The floor exists for the
        # DP cost's numerical stability (a non-finite cost makes the DP unable to
        # backtrack), so it is applied there, in build_cost, and not here.
        log_probabilities = F.log_softmax(class_logits, dim=-1)
        # Section 4.1.3's closing recommendation: even when the training loss uses a
        # per-candidate b_j, the cost driving the SELECTION uses the shared, globally
        # updated b. build_cost reads self.lambda_l1 = 1/b, so that holds by
        # construction here. It matters because L_match is not only a loss term but the
        # cost the DP minimizes, formed BEFORE any gradient exists: an inflated b_j
        # would make a badly localised candidate look artificially cheap and corrupt
        # sigma itself, not merely the loss computed after it.
        precision_scales = self._precision_scales(raw_precision)
        precision_terms = []
        # Matches the print gate at the bottom of this function (which tests the
        # POST-increment count), so the numbers printed are always freshly computed.
        want_diag = bool(self.diagnostic_every) and (
            (self._call_count + 1) % self.diagnostic_every == 1)

        # Per-fragment normalized terms. Normalization is PER FRAGMENT, not by the
        # batch-wide event count: with a shared batch denominator an M=0 fragment's
        # background term was divided by the OTHER fragments' event counts, so the
        # identical fragment received a 10x larger gradient in an all-empty batch than
        # when batched with a dense one (adversarial audit finding, reproduced). Here
        # each fragment is scaled by its own denominator and the batch is averaged, so
        # a fragment's effective weight never depends on what it is batched with.
        class_terms, time_terms, background_terms = [], [], []
        continuity_terms, periodicity_terms = [], []
        matched_residuals = []
        total_events = 0
        contributing_fragments = 0
        cost_class_sum, cost_cells = 0.0, 0
        class_spread_sum, class_spread_count = 0.0, 0
        infeasible = 0
        non_finite = 0
        non_finite_logits = 0
        unlabelled_events = 0

        for b in range(batch_size):
            event_classes = targets[b]['classes']
            event_times = targets[b]['times']
            M = int(event_classes.numel())

            log_probabilities_b = log_probabilities[b]
            background = -log_probabilities_b[:, BACKGROUND]

            if M == 0:
                # No events in this fragment: every candidate is background. The
                # fragment normalizer is N (there is no event count to scale by), which
                # puts the per-candidate background pressure on the same order as a
                # dense fragment's per-event terms.
                denominator = float(num_candidates) if self.normalize_by_events else 1.0
                background_terms.append(background.sum() / denominator)
                contributing_fragments += 1
                continue
            if M > num_candidates:
                # The paper's loss (8) is undefined for M > N (no feasible
                # order-preserving injection exists). Skip the fragment ENTIRELY -
                # contributing a background term here would push every candidate on a
                # fully-annotated fragment toward the empty-set class, i.e. train in
                # the opposite direction of the annotation (adversarial audit finding,
                # reproduced). N is chosen from measured annotation density (see
                # train.py --num_candidates) so this should never fire at defaults;
                # warn loudly if it does, because it means N is set too small.
                infeasible += 1
                print(f"[subset] WARNING: fragment with M={M} events > N={num_candidates} "
                      f"candidates skipped entirely (loss undefined; raise --num_candidates)",
                      flush=True)
                continue

            with torch.no_grad():
                corrected, class_cost, time_cost = self.build_cost(
                    log_probabilities_b, t_hat[b], event_classes, event_times)
                # A non-finite cost makes the running-minimum DP unable to backtrack and
                # subset_select_dp raises. That exception propagates to train.py, which
                # skips the whole BATCH -- and once it happens on every batch, training
                # stops while the epoch loop keeps spinning and validation keeps
                # reporting a frozen model. Skip just this fragment instead, and surface
                # it in the stats so a run that starts degrading is visible.
                cost_is_finite = bool(torch.isfinite(corrected).all())
                if want_diag:
                    cost_class_sum += float(class_cost.sum())
                    cost_cells += class_cost.numel()
                # Spread of the class term between neighbouring candidates: the amount
                # of class evidence available to overcome one slot of time cost.
                if want_diag:
                    class_spread_sum += float(
                        (class_cost[:, 1:] - class_cost[:, :-1]).abs().mean())
                    class_spread_count += 1
                # Selection is a non-differentiable combinatorial step evaluated at the
                # current parameters; sigma enters the loss as data (Algorithm 3).
                fragment_meter = self._fragment_meter(event_classes)
                phi_hat = None   # set only by the joint (sigma, phi_0) MAP below
                if not cost_is_finite:
                    sigma_np = None
                elif self.joint_phase and fragment_meter > 1:
                    # Equation (19): search over (sigma, phi_0) together. Each
                    # hypothesis gets its own phase-conditioned cost, and the cheapest
                    # run wins -- so a matching attractive under a losing hypothesis is
                    # never the one reported.
                    phase_costs = torch.stack([
                        self.build_phase_cost(log_probabilities_b, t_hat[b],
                                              event_classes, event_times,
                                              fragment_meter, p)
                        for p in range(fragment_meter)])
                    sigma_np, phi_hat = subset_select_dp_joint_phase(
                        phase_costs.detach().cpu().numpy())
                elif self.mu_meter > 0.0:
                    _dbp = (event_classes == DOWNBEAT).nonzero(as_tuple=False).flatten().cpu().numpy()
                    _L = float(fragment_meter)
                    sigma_np = (subset_select_dp_meter(
                                    corrected.detach().cpu().numpy(), _dbp,
                                    t_hat[b].detach().cpu().numpy(), _L, self.mu_meter)
                                if _L >= 1.0 else
                                subset_select_dp(corrected.detach().cpu().numpy()))
                else:
                    sigma_np = subset_select_dp(corrected.detach().cpu().numpy())

            # Must be OUTSIDE the no_grad block above: a term appended inside it is
            # detached, and if every fragment in the batch took that path the total
            # would carry no grad_fn at all and backward() would raise
            # "element 0 of tensors does not require grad" -- which is exactly how the
            # first version of this guard killed a run.
            if not cost_is_finite:
                # The cost can be non-finite for two very different reasons and they
                # need opposite handling.
                #
                #   (a) the cost blew up but the probabilities are still finite -- the
                #       DP cannot backtrack, but a background term is safe and keeps
                #       the fragment contributing something sane.
                #
                #   (b) the LOGITS themselves are NaN, so `background` is NaN too.
                #       Appending it puts NaN straight into the total; backward() then
                #       writes NaN into every weight and the run is dead while the
                #       epoch loop keeps spinning on a frozen model. Observed live:
                #       subset_nosmc, epoch 0 iteration 194 -- "BG: nan" with CLS still
                #       finite (that fragment's NaN, its batchmates' CLS fine), then
                #       every subsequent iteration 0.00000/nan for seven epochs.
                #
                # So drop the fragment entirely in case (b).
                non_finite += 1
                if bool(torch.isfinite(background).all()):
                    background_terms.append(background.sum() / (float(M) if self.normalize_by_events else 1.0))
                    contributing_fragments += 1
                else:
                    non_finite_logits += 1
                    print(f"[subset] WARNING: NaN/Inf class logits in fragment {b} "
                          f"(M={M}); fragment dropped to keep NaN out of backward",
                          flush=True)
                continue

            if self.marginal:
                denominator = float(M) if self.normalize_by_events else 1.0
                # -log Z over the corrected cost, plus gamma * sum_j g_j. Which
                # partition function depends on how much is being marginalized:
                # (23) sigma alone, (30) sigma with phi_0, (32) sigma with phi_0 and L.
                if self.marginal_meters and fragment_meter != 0:
                    # (31)-(32): L is latent too. One log-sum-exp recursion per (L, p)
                    # pair, summed in log space -- the same construction as (25), one
                    # level further marginalized, with an implicitly uniform prior over
                    # the candidate meter set.
                    log_z = meter_joint_log_partition(
                        lambda meter, p: self.build_phase_cost(
                            log_probabilities_b, t_hat[b], event_classes,
                            event_times, meter, p),
                        self.marginal_meters)[0]
                elif self.joint_phase and fragment_meter > 1:
                    # (24)-(25), (30): marginalize sigma and phi_0 TOGETHER. For fixed
                    # p the inner sum over sigma is exactly (22) with L'^p_match
                    # substituted, so this is L calls to the same recursion, summed in
                    # log space. Fully differentiable, no stop-gradient.
                    phase_costs = torch.stack([
                        self.build_phase_cost(log_probabilities_b, t_hat[b],
                                              event_classes, event_times,
                                              fragment_meter, p)
                        for p in range(fragment_meter)])
                    log_z = joint_phase_log_partition(phase_costs)[0]
                else:
                    # (23): -log Z over sigma alone, phase left out of the
                    # marginalization entirely.
                    # REBUILD the cost here: the `corrected` computed above lives inside
                    # a `with torch.no_grad()` block that exists for the hard path, whose
                    # DP only needs a detached matrix. The marginal path differentiates
                    # THROUGH the cost, so reusing it yields a class term with
                    # requires_grad=False - the loss still falls via the background term
                    # while the class logits never learn, and every candidate stays at
                    # its initialisation prior and decodes nothing. That is exactly what
                    # killed three marginal arms.
                    corrected = self.build_cost(
                        log_probabilities_b, t_hat[b], event_classes, event_times)[0]
                    log_z = subset_select_logsumexp(corrected)
                class_terms.append(-log_z / denominator)
                if self.marginal_background:
                    background_terms.append(background.sum() / denominator)
                if self.lambda_r > 0.0 and sigma_np is not None:
                    # -log Z marginalises sigma away, but the periodicity term needs a
                    # concrete assignment; use the hard DP's, which is already computed
                    # above for diagnostics. Cheap and consistent with how 9.4 is defined
                    # (it is applied after an assignment is fixed).
                    sigma_hard = torch.from_numpy(sigma_np).to(class_logits.device)
                    r_term = self._periodicity_term(t_hat[b][sigma_hard], event_classes)
                    if r_term is not None:
                        periodicity_terms.append(r_term / denominator)
                if self.cont_weight > 0.0:
                    continuity_terms.append(self._continuity_term(
                        log_probabilities_b, t_hat[b]))
                total_events += M
                contributing_fragments += 1
                # -log Z already covers the matched class and time terms, so those are
                # skipped below - but the shared blocks above must NOT be, or
                # --lambda_r / --cont_weight become silent no-ops under --marginal while
                # still printing 0.000 in the loss table.
                continue

            sigma = torch.from_numpy(sigma_np).to(class_logits.device)

            # --- loss (8), matched terms -------------------------------------
            denominator = float(M) if self.normalize_by_events else 1.0

            matched_log = log_probabilities_b[sigma]                      # (M, 3)
            unknown = event_classes == CLASS_UNKNOWN
            safe_classes = torch.where(unknown, torch.zeros_like(event_classes), event_classes)
            per_event = -matched_log.gather(1, safe_classes[:, None]).squeeze(1)

            if bool(unknown.any()):
                # Section 8.1: a fragment may carry known meter change-points as
                # [(first_event_index, L_k), ...]. Absent that, the whole fragment is
                # one segment under the fragment's own meter, which is the single-meter
                # derivation of section 8 unchanged.
                segments = targets[b].get('segments')
                beat_only_term, beat_only_r = self._beat_only_term(
                    matched_log, segments, phi_hat=phi_hat, meter=fragment_meter)
                per_event = torch.where(unknown, beat_only_term, per_event)
                unlabelled_events += int(unknown.sum())
            else:
                beat_only_r = None

            omega = torch.where(
                event_classes == DOWNBEAT,
                torch.full_like(event_times, self.omega_downbeat),
                torch.full_like(event_times, self.omega_beat))
            # NOTE: beat-only events keep omega_beat (default 1.0), NOT an r-weighted
            # blend of omega_B/omega_DB. Section 10.3 scopes omega_c to eq. (8), and
            # Algorithm 5 line 19's loss line carries no omega, so with the default
            # omega_beat = 1.0 this is the unweighted term the paper specifies. An
            # r-interpolated weight was tried here and reverted: it reads as principled
            # but is a term no algorithm lists, and it silently re-weights the 3% of
            # training that is beat-only.
            class_terms.append((omega * per_event).sum() / denominator)

            residual = (event_times - t_hat[b][sigma]).abs()
            if self.tol_flat > 0.0:
                residual = (residual - self.tol_flat).clamp(min=0.0)
            # Eq. (8) brackets omega_{c_i} around BOTH the class and the time term:
            #   sum_i omega_{c_i} [ -log p_hat(c_i) + lambda_L1 |t_i - t_hat| ] + gamma sum g_j
            # This previously weighted the class term only. The prose calls omega "a
            # class-specific weight on the matched classification term", which is why the
            # narrower reading was defensible, but the displayed equation is the
            # specification and it scopes omega over the bracket.
            if precision_scales is None:
                time_terms.append(
                    (omega * self.lambda_l1 * residual).sum() / denominator)
            else:
                b_j = precision_scales[b][sigma]
                time_terms.append(
                    (omega * self._per_candidate_time_term(residual, b_j)).sum()
                    / denominator)
                precision_terms.append(self._precision_prior(b_j) / denominator)
            matched_residuals.append(residual.detach())

            # --- loss (8), background term -----------------------------------
            unmatched = torch.ones(num_candidates, dtype=torch.bool, device=class_logits.device)
            unmatched[sigma] = False
            background_terms.append(background[unmatched].sum() / denominator)

            if self.lambda_r > 0.0:
                r_term = self._periodicity_term(t_hat[b][sigma], event_classes)
                if r_term is not None:
                    periodicity_terms.append(r_term / denominator)

            if self.cont_weight > 0.0:
                continuity_terms.append(self._continuity_term(
                    log_probabilities_b, t_hat[b]))

            total_events += M
            contributing_fragments += 1

        device = class_logits.device
        # NOT torch.zeros(): if every fragment in the batch was dropped, a constant
        # zero total carries no grad_fn and backward() raises "element 0 of tensors
        # does not require grad", turning a skippable batch into a dead run. Zeroing a
        # sum of the logits keeps the graph attached and contributes exactly zero
        # gradient.
        # nan_to_num first: a plain class_logits.sum() is NaN when the batch is the
        # very thing this guards against, and NaN * 0.0 is still NaN.
        zero = torch.nan_to_num(class_logits).sum() * 0.0
        loss_class = torch.stack(class_terms).sum() if class_terms else zero
        loss_time = torch.stack(time_terms).sum() if time_terms else zero
        loss_background = torch.stack(background_terms).sum() if background_terms else zero

        # The paper writes (8) as a plain unnormalized sum over ONE fragment; the
        # per-fragment /M above plus this batch mean is a deliberate, documented
        # deviation (keeps dense fragments from dominating the gradient and keeps the
        # batch loss scale independent of batch size). normalize_by_events=False gives
        # the literal equation: plain sums, no denominators anywhere.
        batch_denominator = max(contributing_fragments, 1) if self.normalize_by_events else 1
        losses = {
            'class': loss_class / batch_denominator,
            'time': loss_time / batch_denominator,
            'background': self.gamma * loss_background / batch_denominator,
        }
        if continuity_terms:
            losses['continuity'] = (self.cont_weight * torch.stack(continuity_terms).sum()
                                    / batch_denominator)
        else:
            losses['continuity'] = zero
        losses['periodicity'] = ((self.lambda_r * torch.stack(periodicity_terms).sum()
                                  / batch_denominator) if periodicity_terms else zero)
        # The Gamma MAP term rides along with the time channel it regularizes.
        if precision_terms:
            losses['time'] = losses['time'] + (torch.stack(precision_terms).sum()
                                               / batch_denominator)
        losses['total'] = (losses['class'] + losses['time'] + losses['background']
                           + losses['continuity'] + losses['periodicity'])

        if self.learn_b and matched_residuals:
            self._update_b(torch.cat(matched_residuals))

        stats = {
            'cls': float(losses['class']),
            'time': float(losses['time']),
            'bg': float(losses['background']),
            'total': float(losses['total']),
            'num_events': total_events,
            'infeasible': infeasible,
            'non_finite': non_finite,
            'non_finite_logits': non_finite_logits,
            'unlabelled_events': unlabelled_events,
            'b': float(self.b),
            'lambda_l1': self.lambda_l1,
            # Diagnostics for the scale warning above.
            'cost_class_mean': (cost_class_sum / max(cost_cells, 1)) if want_diag
                               else self._diag_cache['cost_class_mean'],
            'class_spread': (class_spread_sum / max(class_spread_count, 1)) if want_diag
                            else self._diag_cache['class_spread'],
            'slot_time_cost': self.lambda_l1 / num_candidates,
        }
        if matched_residuals:
            stats['residual_mean'] = (float(torch.cat(matched_residuals).mean())
                                      if want_diag else self._diag_cache['residual_mean'])
        with torch.no_grad():
            # Smallest spacing between consecutive candidate times. Equation (1)
            # guarantees this is >= 0 always; if it reaches exactly 0 the sequence has
            # stopped being strictly increasing in float32 (see monotonic_times) and
            # duplicate detection times become possible. (Guarded: at N=1 there are no
            # gaps and .min() of an empty tensor would crash.)
            if want_diag:
                gaps = t_hat[:, 1:] - t_hat[:, :-1]
                stats['min_gap'] = float(gaps.min()) if gaps.numel() else float('inf')
            else:
                stats['min_gap'] = self._diag_cache['min_gap']

        if want_diag:
            for _k in self._diag_cache:
                if _k in stats:
                    self._diag_cache[_k] = stats[_k]
        self._call_count += 1
        if self.diagnostic_every and self._call_count % self.diagnostic_every == 1:
            # The scale warning light. If cost_time dwarfs cost_class the DP has become
            # nearest-in-time matching and the class term no longer influences which
            # candidate is selected - lower lambda_l1 (raise b) if so.
            # The actionable comparison is NOT the mean over the whole (M, N) cost
            # matrix - that is dominated by pairs at opposite ends of the window and
            # says nothing. What decides whether class influences the selection is the
            # cost of moving a match one candidate slot sideways, lambda_L1 * (1/N),
            # against how far apart the class costs of neighbouring candidates are. If
            # slot_cost greatly exceeds class_spread the DP is pure nearest-in-time.
            slot_cost = stats['lambda_l1'] / num_candidates
            print(f"[subset] b={stats['b']:.5f} lambda_L1={stats['lambda_l1']:.1f} | "
                  f"one-slot time cost={slot_cost:.3f} vs class spread={stats['class_spread']:.3f} "
                  f"({'TIME DOMINATES' if slot_cost > 4 * stats['class_spread'] else 'balanced'}) | "
                  f"residual={stats.get('residual_mean', float('nan')):.5f} "
                  f"({stats.get('residual_mean', 0.0) * self.fragment_seconds * 1000:.0f}ms) "
                  f"min_gap={stats['min_gap']:.2e} events={stats['num_events']} "
                  f"infeasible={stats['infeasible']}", flush=True)
        return losses, stats

    def _beat_only_term(self, matched_log, segments=None, phi_hat=None, meter=0):
        """Section 8 class term for events whose B/DB label was never observed.

        The single-event marginal (9), l_i = -log(p(B) + p(DB)), is exactly correct
        but structurally unable to teach the distinction: both classes enter only
        through their sum, so its gradient along any B-versus-DB redistribution is
        identically zero. What recovers that signal is pooling evidence across the
        fragment's M matched events, coupled by the meter, which is what the EM of
        equations (12)-(15) does.

        E-step (12): the posterior over the fragment's one unresolved latent, the
        phase of the first matched event,

            pi_p  propto  prod_i p_hat_sigma(i)( phi = (p + i - 1) mod L )

        computed at theta_old and detached, which is the unique choice of q making
        Jensen's inequality tight.

        M-step (15): each event's own weighted cross-entropy, with

            r_i = pi_{p*_i},     p*_i = (1 - i) mod L
            l_i = -[ r_i log p_hat(DB) + (1 - r_i) log p_hat(B) ]

        r_i is a direct lookup rather than a sum because exactly one hypothesis makes
        event i a downbeat. Every r_i is coupled to every OTHER matched event's
        prediction through the shared posterior pi -- precisely what (9) lacked by
        never looking beyond event i's own prediction. By Fisher's identity a gradient
        step on this surrogate coincides with one on the true marginal (11).

        Under mixed meter (section 8.1) the derivation is unchanged in form: eq. (16)
        factors the joint marginal across segments before any sum is taken, so each
        segment gets its own posterior (17) computed from its own events alone, and
        the per-event loss formula never changes. Only how r_i is computed changes.

        Confidence gating guards against bootstrapping early mistakes: if the pooled
        posterior is still undecided, fall back to (9) rather than force-fitting an
        unreliable pseudo-label. Warm-up does the same, one level up.
        """
        active = torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)     # log(1-p(empty))
        marginal = -active
        if self._call_count < self.beat_only_warmup:
            return marginal, None

        if phi_hat is not None and meter > 1:
            # Joint path (eq. 19 / Algorithm 7): sigma and phi_0 were selected TOGETHER,
            # so phi_hat is already the MAP phase for this fragment under the chosen
            # matching. Re-deriving a soft r from the sequential E-step here would
            # discard it and apply Algorithm 5's matching-then-phase target on top of a
            # jointly-decoded sigma -- neither algorithm as written. Use the hard target
            # phi_hat implies; eq. (15) with r in {0, 1} is exactly that hard target.
            with torch.no_grad():
                r = event_is_downbeat_under(
                    matched_log.shape[0], int(meter), int(phi_hat), matched_log.device
                ).to(matched_log.dtype)
            weighted = -(r * matched_log[:, DOWNBEAT] + (1.0 - r) * matched_log[:, BEAT])
            return weighted, r

        posterior = self._phase_posterior_marginal(matched_log, segments)
        if posterior is None:
            # No usable meter: (9) is all the annotation supports.
            return marginal, None
        r, valid = posterior

        with torch.no_grad():
            confident = valid & (torch.maximum(r, 1.0 - r) >= self.beat_only_confidence)
        weighted = -(r * matched_log[:, DOWNBEAT] + (1.0 - r) * matched_log[:, BEAT])
        # (15) exactly, and Algorithm 5 line 19's complete loss line: the weighted
        # term ALONE. It previously read `weighted + marginal`, on the reasoning that
        # (15) supervises which of B/DB while (9) keeps the candidate off the
        # background class. That double-counts: log p(DB) and log p(B) here are
        # unconditional, so each already contains one factor of log p(event) --
        # weighted + marginal = [conditional CE] + 2 * (-log p(event)), weighting
        # event-detection twice and the B-vs-DB decision once. Background pressure on
        # unmatched candidates is separately supplied by the gamma term of loss (8).
        return torch.where(confident, weighted, marginal), torch.where(
            confident, r, torch.zeros_like(r))

    def _phase_posterior_marginal(self, matched_log, segments=None):
        """Equations (12)/(14), and (17) per segment: r_i = P(phi_i = 0 | y, x; theta).

        Returns (M,) detached posterior downbeat probabilities, or None if no meter is
        available. This is the E-step, evaluated at theta_old, so it is computed under
        no_grad and recomputed fresh every forward pass -- caching a stale E-step is
        exactly what would break the monotonicity guarantee.
        """
        M = matched_log.shape[0]
        if segments is None:
            L = int(self.meter_length)
            if L <= 1 or M < L:
                return None
            segments = [(0, L)]

        with torch.no_grad():
            r = torch.zeros(M, device=matched_log.device, dtype=matched_log.dtype)
            valid = torch.zeros(M, device=matched_log.device, dtype=torch.bool)
            for k, (start, meter) in enumerate(segments):
                end = segments[k + 1][0] if k + 1 < len(segments) else M
                span = matched_log[start:end]
                length = end - start
                if meter <= 1 or length == 0:
                    # No usable meter for this segment: leave `valid` False so the
                    # caller falls back to eq. (9). Leaving r at 0 here instead would
                    # read as max(r, 1-r) = 1 >= confidence, i.e. a CONFIDENT "this is
                    # a beat" pseudo-label, asserting exactly what we do not know.
                    continue
                # (12)/(17): unnormalized log posterior of each phase hypothesis,
                # from this segment's own matched events only.
                log_pi = torch.stack([
                    torch.where(event_is_downbeat_under(length, meter, p, span.device),
                                span[:, DOWNBEAT], span[:, BEAT]).sum()
                    for p in range(meter)])
                pi = torch.softmax(log_pi, dim=0)
                # (14)/(34): singleton lookup at p*_i, indexed within the segment.
                r[start:end] = pi[phase_star(length, meter, span.device)]
                valid[start:end] = True
        return r, valid

    def _precision_scales(self, raw_precision):
        """Equation of section 4.1.3: b_j = b_min + softplus(u_j).

        The floor rules out precision collapse by construction. Returns None when no
        precision head is present or the warm-up has not elapsed, in which case the
        caller uses the shared global b -- which is mitigation four, forcing accurate
        localisation to be learned before an uncertainty channel exists to substitute
        for it.
        """
        if raw_precision is None or self._call_count < self.precision_warmup:
            return None
        return self.b_min + F.softplus(raw_precision)

    def _per_candidate_time_term(self, residual, b_j):
        """The time channel under per-candidate precision, with the 4.1.3 stop-gradient.

        The per-example loss is |t_i - t_hat_j| / b_j + log(2 b_j). Written naively,
        one gradient step can widen b_j and leave t_hat_j unimproved -- the cheating
        direction, since inflating precision is a much smaller penalty than the harder
        work of localising correctly, and it is self-reinforcing, because the residual
        gradient scales as 1/b_j.

        So the two roles b_j plays are decoupled: t_hat_j sees b_j only through a
        DETACHED value, and b_j is updated only through its own separate term. Neither
        term is dropped -- both are the same loss -- but no single step can trade one
        against the other.
        """
        localisation = residual / b_j.detach()
        precision = residual.detach() / b_j + torch.log(2.0 * b_j)
        return localisation + precision

    def _precision_prior(self, b_j):
        """Mitigation two: a Gamma prior on the precision 1/b_j, as a MAP term.

        Gamma(alpha, beta) on tau := 1/b_j has log-density (alpha - 1) log tau - beta *
        tau up to a constant, so the negative log prior contributed to the objective is

            -(alpha - 1) * log(1 / b_j) + beta / b_j
             = (alpha - 1) * log b_j + beta / b_j,

        convex in b_j with its minimum at b_j = beta / (alpha - 1). Setting beta from
        the shared, data-estimated b makes that default what the data actually says
        rather than an arbitrary constant, so the prior damps runaway inflation toward
        the dataset-wide scale instead of toward an unrelated one.
        """
        alpha = self.precision_prior_alpha
        beta = (self.precision_prior_beta if self.precision_prior_beta is not None
                else float(self.b) * max(alpha - 1.0, 1e-6))
        return ((alpha - 1.0) * torch.log(b_j) + beta / b_j).sum()

    def _update_b(self, residuals):
        """Equation (5): b_hat = mean absolute residual over matched pairs, the
        maximum-likelihood Laplace scale. Kept as an EMA across minibatches for
        stability and held fixed while theta is updated.
        """
        batch_estimate = residuals.mean().clamp(min=self.b_min)
        self.b.mul_(self.b_momentum).add_((1.0 - self.b_momentum) * batch_estimate)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def targets_to_events(target, num_frames=None):
    """Frame-grid target (2, T) -> event list for one fragment.

    The dataloader rasterises annotations onto a frame grid: channel 0 marks every
    beat and channel 1 marks downbeats, and because load_annot appends a downbeat to
    *both* lists, every downbeat is also set in channel 0. The three classes here are
    exclusive, so downbeat wins wherever both fire.

    Times are normalised by the frame count, matching equation (1)'s (0, 1) range.
    Quantisation to the frame grid costs up to half a frame (11.6 ms at 43.07 fps) of
    target precision; the annotation grid has always had the same quantisation, so this
    keeps the comparison apples-to-apples. Passing exact annotation seconds through
    the dataloader would remove it and is the obvious later refinement.
    """
    if num_frames is None:
        num_frames = target.shape[-1]
    beat_frames = torch.nonzero(target[0] > 0, as_tuple=False).flatten()
    downbeat_frames = torch.nonzero(target[1] > 0, as_tuple=False).flatten()

    downbeat_set = set(downbeat_frames.tolist())
    beat_only = [f for f in beat_frames.tolist() if f not in downbeat_set]

    frames = sorted(downbeat_set.union(beat_only))
    if len(frames) == 0:
        return {
            'classes': torch.zeros(0, dtype=torch.long, device=target.device),
            'times': torch.zeros(0, dtype=torch.float32, device=target.device),
        }

    classes = torch.tensor(
        [DOWNBEAT if f in downbeat_set else BEAT for f in frames],
        dtype=torch.long, device=target.device)
    times = torch.tensor(frames, dtype=torch.float32, device=target.device) / float(num_frames)
    return {'classes': classes, 'times': times}


def batch_targets_to_events(targets, num_frames=None):
    return [targets_to_events(targets[b], num_frames=num_frames) for b in range(targets.shape[0])]


def intervals_to_events(annotations, num_frames):
    """Collated (M, 3) interval annotations -> event list. This is the path the real
    pipeline uses.

    BeatDataset.__getitem__ does not hand back the (2, T) frame grid; it returns
    make_intervals(target), an (M, 3) array of [start_frame, end_frame, class_id]
    where class_id 0 spans consecutive downbeats and class_id 1 spans consecutive
    beats. collater pads the batch with rows of -1.

    Events are recovered exactly: for a chain of abutting intervals the event times
    are every start plus the final end, since end_k == start_{k+1}. The beat chain is
    built from all beats (channel 0 of the grid contains downbeats too), so the
    downbeat set is a subset of the beat set and downbeat wins on the overlap - the
    same exclusivity rule as targets_to_events.

    [Inherited quirk, deliberately not worked around] make_intervals returns an empty
    array whenever a crop holds fewer than two downbeats OR fewer than two beats, so
    such a crop contributes no targets at all rather than a partial set. The interval
    has always trained under exactly this rule, so reproducing it keeps the comparison
    conversion is honest. It does mean very slow or near-silent crops are silently
    empty; SubsetCriterion counts them as all-background.

    annotations: (M, 3) for one fragment, or (B, M, 3) for a batch.
    Returns one dict, or a list of dicts for a batch.
    """
    if annotations.dim() == 3:
        return [intervals_to_events(annotations[b], num_frames) for b in range(annotations.shape[0])]

    device = annotations.device
    empty = {
        'classes': torch.zeros(0, dtype=torch.long, device=device),
        'times': torch.zeros(0, dtype=torch.float32, device=device),
    }
    if annotations.numel() == 0:
        return empty

    valid = annotations[annotations[:, 2] >= 0]
    if valid.numel() == 0:
        return empty

    def endpoints(rows):
        if rows.numel() == 0:
            return torch.zeros(0, device=device)
        return torch.unique(torch.cat((rows[:, 0], rows[:, 1])))

    # class_id 2 marks a beat-only dataset (dataloader.CLASS_BEAT_ONLY): the event is
    # certainly a beat, but whether it is a downbeat was never annotated. Such a
    # fragment carries ONLY these rows, so handle it before the normal two-chain case.
    beat_only = endpoints(valid[valid[:, 2] == 2])
    if beat_only.numel() > 0:
        beat_only = beat_only[(beat_only >= 0) & (beat_only <= num_frames)]
        return {
            'classes': torch.full((beat_only.numel(),), CLASS_UNKNOWN,
                                  dtype=torch.long, device=device),
            'times': beat_only.float() / float(num_frames),
        }

    downbeat_frames = endpoints(valid[valid[:, 2] == DOWNBEAT])
    beat_frames = endpoints(valid[valid[:, 2] == BEAT])

    frames = torch.unique(torch.cat((downbeat_frames, beat_frames)))
    # Defensive: an annotation frame outside [0, num_frames] would produce an event
    # time outside (0, 1] that the criterion would silently accept (the cost and DP
    # are happy to match it, just badly). The dataloader's crop slices the frame grid
    # before make_intervals so this should not occur; drop rather than clamp if it
    # ever does, since a clamped time would be a fabricated event position.
    frames = frames[(frames >= 0) & (frames <= num_frames)]
    if frames.numel() == 0:
        return empty

    is_downbeat = torch.isin(frames, downbeat_frames)
    classes = torch.where(
        is_downbeat,
        torch.full_like(frames, DOWNBEAT, dtype=torch.long),
        torch.full_like(frames, BEAT, dtype=torch.long))

    return {'classes': classes, 'times': frames.float() / float(num_frames)}


# ---------------------------------------------------------------------------
# Inference (section 9.2, Algorithm 5)
# ---------------------------------------------------------------------------

def estimate_beat_period(times, scores, threshold=0.2):
    """Equation (36)'s Delta_bar, estimated from PREDICTIONS rather than matches.

    Eq. (36) averages the spacing of the matched candidates t_hat_sigma(i), which needs
    sigma and therefore M. At inference neither exists, so the period is taken from a
    provisional Algorithm 5 decode instead -- the same one-EM-step device section 4.2's
    own implementation uses, which solves the plain DP first purely to obtain a
    Delta_bar for the augmented one.

    The MEDIAN gap is used rather than the mean: a provisional decode drops events, and
    a single missed beat doubles one gap, which drags a mean far more than a median.

    Returns None when there is too little to estimate from, which the caller must treat
    as "fall back to Algorithm 5" rather than guessing a period.
    """
    kept = times[scores >= threshold]
    if kept.shape[0] < 4:
        return None
    gaps = np.diff(np.sort(kept))
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else None


def decode_events_coupled(class_logits, t_hat, beat_period, gamma=0.5, mu=1.0,
                          threshold_beat=0.2, threshold_downbeat=0.2):
    """Decoding that couples neighbouring candidates, in place of Algorithm 5.

    THE PROBLEM. Training selects sigma jointly over O_{M,N} -- order-preserving,
    exactly M candidates, every candidate's cost coupled through one global choice --
    while Algorithm 5 decodes each candidate in isolation. Measured on a checkpoint at
    epoch 13, closing that gap is worth 0.158 joint F, the single largest loss in the
    pipeline.

    WHY THE PAPER'S OWN SUGGESTIONS DO NOT APPLY. Section 8.4 proposes decoding either
    by Algorithm 1 or by thresholding the per-candidate matching marginal. Algorithm 1's
    table is indexed i = 1..M and the marginal P(sigma(i) = j) is defined over O_{M,N};
    both presuppose M, which inference does not have -- as section 9.2 itself states.

    WHY A FREE-M DP IS NOT ENOUGH ON ITS OWN. If the score is a sum of per-candidate
    terms, the problem is separable: each candidate independently takes
    min(-log p_j(c), -log p_j(empty)), which IS the arg max of Algorithm 5. Monotonicity
    is already free, since eq. (1) sorts t_hat by construction. Structured decoding buys
    nothing unless the score couples neighbours.

    WHAT COUPLES THEM HERE. Spacing regularity, the one structure available without
    ground truth. State is (candidate j, index k of the last candidate that fired):

        fire:      D[j, k] -> D[j+1, j]  costs  -log p_j(c*) + mu (t_j - t_k - Delta)^2
        stay:      D[j, k] -> D[j+1, k]  costs  -gamma log p_j(empty)

    O(N^2) states, O(1) transitions.

    WHAT THIS ADHERES TO, AND WHAT IT DOES NOT. The recursion shape is section 5's; the
    emission costs are eq. (3)'s class term and section 8.4's own -gamma g_j background
    correction; the penalty is eq. (6)'s (gap - Delta)^2 with eq. (36)'s estimator.
    It diverges in three ways, deliberately: M is free rather than fixed by the
    annotation; the spacing term is applied at DECODE where sections 4.2 and 10.4 apply
    it during training; and it constrains beat-to-beat spacing rather than eq. (6)'s
    downbeat-to-downbeat L * Delta_bar, because beat spacing is the stronger and more
    available signal when no meter is known.

    ONE CONSEQUENCE WORTH STATING. Section 5 argues skip cost 0 is "a free normalization
    choice, not a load-bearing one", and that argument depends on M being FIXED: every
    feasible sigma then has exactly M diagonal and N - M horizontal steps, so a constant
    added to skip shifts every candidate solution equally. With M free that no longer
    holds -- the non-firing cost directly controls how many events are reported. It is
    therefore the principled -gamma log p_j(empty), not 0.

    mu = 0 reduces exactly to Algorithm 5 (verified in tests), so this is a strict
    generalisation rather than a replacement.

    Returns (classes, times, scores) ascending in time, as decode_events does.
    """
    probabilities = F.softmax(class_logits, dim=-1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    N = t_hat.shape[0]

    if beat_period is None or N == 0:
        return decode_events(class_logits, t_hat, threshold_beat, threshold_downbeat)

    times = t_hat.detach().cpu().numpy().astype(np.float64)
    logp = log_probabilities.detach().cpu().numpy().astype(np.float64)
    # Fire as whichever active class the candidate itself prefers; the spacing term is
    # about WHETHER a candidate fires, not which kind it is.
    fire_class = np.where(logp[:, DOWNBEAT] >= logp[:, BEAT], DOWNBEAT, BEAT)
    fire_cost = -logp[np.arange(N), fire_class]
    stay_cost = -gamma * logp[:, BACKGROUND]

    NONE = N                      # "nothing has fired yet" state
    INF = np.inf
    best = np.full(N + 1, INF)
    best[NONE] = 0.0
    back = np.full((N, N + 1), -1, dtype=np.int64)

    for j in range(N):
        gap = times[j] - np.concatenate([times, [0.0]])
        penalty = mu * (gap - beat_period) ** 2
        penalty[NONE] = 0.0       # the first event has no predecessor to be spaced from
        fire_from = best + fire_cost[j] + penalty
        source = int(np.argmin(fire_from))
        new = best + stay_cost[j]                     # stay: last-fired index unchanged
        back[j, :] = np.arange(N + 1)                 # provisional: everything stayed
        if fire_from[source] < new[j]:
            new[j] = fire_from[source]
            back[j, j] = source
        best = new

    fired = []
    k = int(np.argmin(best))
    for j in range(N - 1, -1, -1):
        if k == j:
            fired.append(j)
            k = int(back[j, j])
    fired = np.array(sorted(fired), dtype=np.int64)
    if fired.size == 0:
        return decode_events(class_logits, t_hat, threshold_beat, threshold_downbeat)

    keep = torch.as_tensor(fired, device=t_hat.device)
    classes = torch.as_tensor(fire_class[fired], device=t_hat.device, dtype=torch.long)
    scores = 1.0 - probabilities[keep, BACKGROUND]
    return classes, t_hat[keep], scores


def decode_events(class_logits, t_hat, threshold_beat=0.2, threshold_downbeat=0.2,
                  literal_argmax=False, db_margin=0.0):
    """Algorithm 5 for one fragment: decide per candidate, then threshold.

    No NMS and no de-duplication: exactly one classification decision is made per
    candidate and t_hat is strictly increasing by equation (1), so two reported
    detections can never coincide or cross. The predicted times are points, and the
    property that matters for a point sequence is strict ordering, not disjointness.

    Two decision rules, differing only in how a candidate is judged to be an event at
    all. Algorithm 5 as written takes the argmax over {DB, B, empty} and discards the
    candidate if empty wins. That throws away a candidate at, say,
    (DB 0.30, B 0.31, empty 0.39) even though it assigns 0.61 to SOMETHING being there
    -- the empty class only has to beat each active class separately, not their sum,
    so an underconfident model loses recall for no good reason.

    The default here instead thresholds p(event) = 1 - p(empty), the same quantity
    equation (9) uses to express "some beat occurred", and only then picks DB versus B
    by their relative mass. That is the decision the three-way distribution actually
    supports: whether an event is present, and separately which kind. Measured at epoch
    5 of a training run: joint F 0.514 -> 0.525, almost all of it beat F
    (0.615 -> 0.636), with downbeat F unchanged -- a small, free gain, larger while the
    model is underconfident and shrinking as it sharpens.

    literal_argmax=True restores Algorithm 5 exactly.

    class_logits (N, 3), t_hat (N,). Returns (classes, times, scores), ascending in
    time, with times still normalised to (0, 1).
    """
    probabilities = F.softmax(class_logits, dim=-1)

    if literal_argmax:
        scores, predicted = probabilities.max(dim=-1)
        thresholds = torch.where(
            predicted == DOWNBEAT,
            torch.full_like(scores, threshold_downbeat),
            torch.full_like(scores, threshold_beat))
        keep = (predicted != BACKGROUND) & (scores >= thresholds)
        return predicted[keep], t_hat[keep], scores[keep]

    # p(event) = 1 - p(empty): is anything here at all?
    event_probability = 1.0 - probabilities[:, BACKGROUND]
    # ... and if so, which kind? The two thresholds stay per-class, so beat and
    # downbeat can still be swept independently (their confidence distributions differ).
    log_probabilities = F.log_softmax(class_logits, dim=-1)
    # db_margin: call DOWNBEAT only if log p(DB) - log p(B) > db_margin. A deliberate
    # deviation from Algorithm 10 line 4: section 10.3's omega_DB trains the classifier
    # under a class-weighted loss, so p-hat converges toward an importance-weighted
    # posterior with DB odds inflated by up to omega, and the minimum-error decision
    # divides that back out (db_margin = log omega). Measured at omega_DB=4 (r1_learnb
    # ep14, fold 0): downbeat F 0.710 -> 0.740 at log 2 on 150 val songs, beat F
    # untouched. The margin relabels B<->DB; since the per-class threshold below is
    # selected BY that label, it also changes the firing set whenever threshold_beat
    # != threshold_downbeat (with the default equal taus it cannot).
    # Default 0.0 keeps the plain argmax; the knob retires itself if omega_DB
    # returns to 1.
    predicted = torch.where(
        log_probabilities[:, DOWNBEAT] - log_probabilities[:, BEAT] > db_margin,
        torch.full_like(event_probability, DOWNBEAT, dtype=torch.long),
        torch.full_like(event_probability, BEAT, dtype=torch.long))
    thresholds = torch.where(
        predicted == DOWNBEAT,
        torch.full_like(event_probability, threshold_downbeat),
        torch.full_like(event_probability, threshold_beat))
    keep = event_probability >= thresholds
    return predicted[keep], t_hat[keep], event_probability[keep]
