"""Order-constrained subset selection head ("Beat Tracking as an Order-Constrained
Subset Selection Problem", beat_dp_matching-1.pdf).

[What] Replaces the FCOS anchor + interval-regression + Soft-NMS pipeline with a
fixed set of N candidate *point* predictors. Each candidate emits a 3-way class
distribution over {DB, B, none} and one raw scalar; a global cumulative-softplus
reparameterization (eq. 1) turns those scalars into a strictly increasing time
sequence. Because both the ground truth and the candidate times are sorted by
construction, deciding which candidate is responsible for which event reduces to
choosing an order-preserving subset, solved exactly by an O(N*M) dynamic program
(Alg. 1). No anchors, no clusters, no NMS.

[Why it is not the abandoned hungarian_head.py] That head's candidates were DETR
learned queries passed through a decoder, which collapsed. Here candidates come
from Downsample(FPN features) and their times are monotone by construction, so
there is nothing to collapse and nothing to sort: the DP indexes candidates in
their natural order. hungarian_head.monotonic_match is the same recursion, but it
sorts by predicted centre first and uses the DETR raw-probability cost (-p) that
section 4 of the paper explicitly corrects to -log p.

[Class index convention] 0 = downbeat, 1 = beat, 2 = background. Indices 0/1 match
the existing repo convention (see get_jth_targets / model_module.py: "class_id 0 =
downbeat, class_id 1 = beat"), so downstream code that already assumes it keeps
working.

Equation numbers below refer to the PDF.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DOWNBEAT = 0
BEAT = 1
BACKGROUND = 2
NUM_CLASSES = 3

# Ground-truth marker for an event that is known to be a beat but whose B/DB
# distinction was never annotated (section 7.1; SMC is the case in practice). It is a
# label for the TARGET, never a class the network predicts - the head always emits the
# same three-way distribution over {DB, B, empty}.
CLASS_UNKNOWN = -1

# Floor for log-probabilities entering the DP cost. Once the model becomes confident,
# p(background) for some candidate underflows to 0 in float32, -log gives +inf, and the
# section 7.2 corrected cost (which SUBTRACTS gamma * that term) becomes -inf; the
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
    self-attention pass (section 9.2) is later added it must feed the *classification*
    branch only, otherwise equation (1)'s guarantee is unaffected but the argument in
    9.2 for why it is safe no longer applies.
    """

    def __init__(self, feature_size=256, num_candidates=160, level_strides=(8, 4, 2),
                 hidden_size=256, dropout=0.0, class_prior=(0.10, 0.30, 0.60)):
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
        self.class_head = nn.Linear(hidden_size, NUM_CLASSES)
        self.regression_head = nn.Linear(hidden_size, 1)

        self._initialize_weights(class_prior)

    def _initialize_weights(self, class_prior):
        for m in self.modules():
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

        # Same treatment for the classifier, for the same reason the FCOS path sets a
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

        class_logits = self.class_head(z)               # (B, N, 3)
        r = self.regression_head(z).squeeze(dim=2)      # (B, N)
        t_hat = monotonic_times(r)
        return class_logits, t_hat


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
    increments = F.softplus(r)
    relative_floor = epsilon * increments.mean(dim=-1, keepdim=True)
    increments = increments + relative_floor + 1e-30
    cumulative = torch.cumsum(increments, dim=-1)
    total = cumulative[..., -1:].clamp(min=1e-30)
    return cumulative / total


# ---------------------------------------------------------------------------
# The correspondence problem (sections 4-5)
# ---------------------------------------------------------------------------

def subset_select_dp(cost):
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


def subset_select_logsumexp(cost):
    """Equation (13) - the marginalised counterpart of the DP, log Z(theta, x).

    Same recursion with min replaced by logsumexp on the negated cost, giving the
    total probability mass over every order-preserving injection instead of the
    single cheapest one (the forward algorithm to Algorithm 1's Viterbi). Provided
    for the section 7.2 training mode; not used by the default hard path. Operates
    on torch tensors because, unlike Algorithm 1, this one is differentiated through.
    """
    M, N = cost.shape
    device, dtype = cost.device, cost.dtype
    if M == 0:
        return torch.zeros((), device=device, dtype=dtype)
    if M > N:
        raise ValueError(f"infeasible: M={M} > N={N}")

    neg_inf = torch.finfo(dtype).min
    previous = torch.zeros(N + 1, device=device, dtype=dtype)
    for i in range(1, M + 1):
        a = previous[0:N] - cost[i - 1]
        # running logsumexp along j, i.e. logcumsumexp
        accumulated = torch.logcumsumexp(a, dim=0)
        current = torch.full((N + 1,), neg_inf, device=device, dtype=dtype)
        current[1:] = accumulated
        previous = current
    return previous[N]


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
                 diagnostic_every=200, beat_only_warmup=2000, beat_only_confidence=0.7):
        super(SubsetCriterion, self).__init__()
        self.gamma = gamma
        # Printed from inside the criterion rather than returned to train.py because
        # under DataParallel the forward runs on a replica, so an attribute set here
        # never reaches the parent module.
        self.diagnostic_every = diagnostic_every
        self._call_count = 0
        # Section 7.1 confirmation-bias mitigations: train beat-only events on the
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
        # buffer so it round-trips through checkpoints with the model
        self.register_buffer('b', torch.tensor(float(b_scale)))

    @property
    def lambda_l1(self):
        return 1.0 / float(self.b.clamp(min=self.b_min))

    def build_cost(self, log_probabilities, t_hat, event_classes, event_times):
        """Per-pair cost (3) plus the section 7.2 background correction.

        (3) is L_match = -log p_j(c_i) + lambda_l1 * |t_i - t_hat_j|, term for term the
        negative log-likelihood of the observation model (2). The paper is explicit
        that this is -log p_j, not the -p_j inherited from the DETR matching-cost
        literature.

        The corrected cost L'_match = L_match - gamma * g_j subtracts the background
        term a candidate would otherwise have paid. Section 7.2 shows the full loss
        (8) contains gamma * sum_{j not in im(sigma)} g_j, and since |im(sigma)| = M is
        fixed, minimising (8) over sigma is minimising sum_i [L_match - gamma *
        g_sigma(i)]. Without this subtraction the DP minimises only part of the loss it
        is supposed to be selecting for - a real gap in the hard formulation, not only
        in the marginalised one.

        Returns the corrected cost (M, N) plus the two raw terms for diagnostics.
        """
        class_cost = self.class_nll(log_probabilities, event_classes)          # (M, N)
        time_cost = self.lambda_l1 * (event_times[:, None] - t_hat[None, :]).abs()
        background = -log_probabilities[:, BACKGROUND]                          # (N,)
        corrected = class_cost + time_cost - self.gamma * background[None, :]
        return corrected, class_cost, time_cost

    @staticmethod
    def class_nll(log_probabilities, event_classes):
        """-log p_j(c_i) for every (event, candidate) pair, handling unlabelled classes.

        For a normally annotated event this is just the log-probability of its class.
        For a beat-only event (CLASS_UNKNOWN, section 7.1) the class is known to lie in
        {B, DB} but which one was never observed, so the likelihood of what is ACTUALLY
        known is the marginal over that superset - equation (9):

            l_i = -log( p_j(B) + p_j(DB) ) = -log( 1 - p_j(empty) )

        This is the exact likelihood of the observation "this candidate is some kind of
        beat", not an approximation. Note what it cannot do (the paper is explicit):
        its gradient with respect to any redistribution of mass between B and DB that
        holds their sum fixed is exactly zero, so on its own it teaches nothing about
        telling downbeats from beats - it only pushes the candidate away from
        background. Section 7.1's EM pseudo-label (equations 10-11) is what recovers
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

    def forward(self, class_logits, t_hat, targets):
        """class_logits (B, N, 3), t_hat (B, N), targets: list of B dicts with
        'classes' (M_b,) long and 'times' (M_b,) float in [0, 1].

        Returns (losses, stats) where losses is a dict of differentiable tensors
        ('class', 'time', 'background', 'total') and stats is floats for logging.
        The components are kept separate because train.py unpacks a five-tuple of
        losses per iteration and logs each one.
        """
        batch_size, num_candidates, _ = class_logits.shape
        log_probabilities = F.log_softmax(class_logits, dim=-1).clamp(min=LOG_PROB_FLOOR)

        # Per-fragment normalized terms. Normalization is PER FRAGMENT, not by the
        # batch-wide event count: with a shared batch denominator an M=0 fragment's
        # background term was divided by the OTHER fragments' event counts, so the
        # identical fragment received a 10x larger gradient in an all-empty batch than
        # when batched with a dense one (adversarial audit finding, reproduced). Here
        # each fragment is scaled by its own denominator and the batch is averaged, so
        # a fragment's effective weight never depends on what it is batched with.
        class_terms, time_terms, background_terms = [], [], []
        matched_residuals = []
        total_events = 0
        contributing_fragments = 0
        cost_class_sum, cost_cells = 0.0, 0
        class_spread_sum, class_spread_count = 0.0, 0
        infeasible = 0
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
                cost_class_sum += float(class_cost.sum())
                cost_cells += class_cost.numel()
                # Spread of the class term between neighbouring candidates: the amount
                # of class evidence available to overcome one slot of time cost.
                class_spread_sum += float((class_cost[:, 1:] - class_cost[:, :-1]).abs().mean())
                class_spread_count += 1
                # Selection is a non-differentiable combinatorial step evaluated at the
                # current parameters; sigma enters the loss as data (Alg. 2 line 17).
                sigma_np = subset_select_dp(corrected.detach().cpu().numpy())
            sigma = torch.from_numpy(sigma_np).to(class_logits.device)

            # --- loss (8), matched terms -------------------------------------
            denominator = float(M) if self.normalize_by_events else 1.0

            matched_log = log_probabilities_b[sigma]                      # (M, 3)
            unknown = event_classes == CLASS_UNKNOWN
            safe_classes = torch.where(unknown, torch.zeros_like(event_classes), event_classes)
            per_event = -matched_log.gather(1, safe_classes[:, None]).squeeze(1)

            if bool(unknown.any()):
                per_event = torch.where(
                    unknown, self._beat_only_term(matched_log), per_event)
                unlabelled_events += int(unknown.sum())

            omega = torch.where(
                event_classes == DOWNBEAT,
                torch.full_like(event_times, self.omega_downbeat),
                torch.full_like(event_times, self.omega_beat))
            class_terms.append((omega * per_event).sum() / denominator)

            residual = (event_times - t_hat[b][sigma]).abs()
            time_terms.append(self.lambda_l1 * residual.sum() / denominator)
            matched_residuals.append(residual.detach())

            # --- loss (8), background term -----------------------------------
            unmatched = torch.ones(num_candidates, dtype=torch.bool, device=class_logits.device)
            unmatched[sigma] = False
            background_terms.append(background[unmatched].sum() / denominator)

            total_events += M
            contributing_fragments += 1

        device = class_logits.device
        zero = torch.zeros((), device=device)
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
        losses['total'] = losses['class'] + losses['time'] + losses['background']

        if self.learn_b and matched_residuals:
            self._update_b(torch.cat(matched_residuals))

        stats = {
            'cls': float(losses['class']),
            'time': float(losses['time']),
            'bg': float(losses['background']),
            'total': float(losses['total']),
            'num_events': total_events,
            'infeasible': infeasible,
            'unlabelled_events': unlabelled_events,
            'b': float(self.b),
            'lambda_l1': self.lambda_l1,
            # Diagnostics for the scale warning above.
            'cost_class_mean': cost_class_sum / max(cost_cells, 1),
            'class_spread': class_spread_sum / max(class_spread_count, 1),
            'slot_time_cost': self.lambda_l1 / num_candidates,
        }
        if matched_residuals:
            stats['residual_mean'] = float(torch.cat(matched_residuals).mean())
        with torch.no_grad():
            # Smallest spacing between consecutive candidate times. Equation (1)
            # guarantees this is >= 0 always; if it reaches exactly 0 the sequence has
            # stopped being strictly increasing in float32 (see monotonic_times) and
            # duplicate detection times become possible. (Guarded: at N=1 there are no
            # gaps and .min() of an empty tensor would crash.)
            gaps = t_hat[:, 1:] - t_hat[:, :-1]
            stats['min_gap'] = float(gaps.min()) if gaps.numel() else float('inf')

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
                  f"({stats.get('residual_mean', 0.0) * 1000:.0f}ms/window) "
                  f"min_gap={stats['min_gap']:.2e} events={stats['num_events']} "
                  f"infeasible={stats['infeasible']}", flush=True)
        return losses, stats

    def _beat_only_term(self, matched_log):
        """Section 7.1 class term for events whose B/DB label was never observed.

        Two regimes, switched by warm-up:

        Warm-up (or gating declines): equation (9), the marginal likelihood
            l_i = -log( p(B) + p(DB) )
        which is exactly correct but, as its own derivation notes, carries no gradient
        for distinguishing B from DB.

        After warm-up: equations (10)-(11). The shared classification head has been
        trained on data where the distinction IS observed, so its own current estimate
        of DB-versus-B is a soft pseudo-label for the withheld one:
            q(DB) = p(DB) / (p(DB) + p(B)),   q(B) = 1 - q(DB)
        detached exactly as sigma_hat is detached, and used as a soft cross-entropy
        target. This is a genuine E-step posterior over a real latent variable (the
        withheld label), so it is EM-shaped rather than EM by analogy.

        Confidence gating guards against bootstrapping early mistakes: if the head is
        not yet decided (max q below the threshold), fall back to (9) rather than
        force-fitting an unreliable pseudo-label.
        """
        active = torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)     # log(1-p(empty))
        marginal = -active
        if self._call_count < self.beat_only_warmup:
            return marginal
        with torch.no_grad():
            q = torch.softmax(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)        # (M, 2), normalised
            confident = q.max(dim=-1).values >= self.beat_only_confidence
        pseudo = -(q[:, 0] * matched_log[:, DOWNBEAT] + q[:, 1] * matched_log[:, BEAT])
        # Keep the marginal alongside the pseudo-label term: (11) supervises WHICH of
        # B/DB, while (9) is what keeps the candidate off the background class at all.
        return torch.where(confident, pseudo + marginal, marginal)

    @torch.no_grad()
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
    target precision; the FCOS path has always had the same quantisation, so this
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
    such a crop contributes no targets at all rather than a partial set. The FCOS path
    has always trained under exactly this rule, so reproducing it keeps the comparison
    against fcos_lite honest. It does mean very slow or near-silent crops are silently
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
# Inference (section 8.2, Algorithm 3)
# ---------------------------------------------------------------------------

def decode_events(class_logits, t_hat, threshold_beat=0.2, threshold_downbeat=0.2):
    """Algorithm 3 for one fragment: per-candidate argmax, then a class threshold.

    No NMS and no de-duplication: exactly one classification decision is made per
    candidate and t_hat is strictly increasing by equation (1), so two reported
    detections can never coincide or cross. The predicted times are points, and the
    property that matters for a point sequence is strict ordering, not disjointness.

    class_logits (N, 3), t_hat (N,). Returns (classes, times, scores), ascending in
    time, with times still normalised to (0, 1).
    """
    probabilities = F.softmax(class_logits, dim=-1)
    scores, predicted = probabilities.max(dim=-1)

    thresholds = torch.where(
        predicted == DOWNBEAT,
        torch.full_like(scores, threshold_downbeat),
        torch.full_like(scores, threshold_beat))
    keep = (predicted != BACKGROUND) & (scores >= thresholds)

    return predicted[keep], t_hat[keep], scores[keep]
