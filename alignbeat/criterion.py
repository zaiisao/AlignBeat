"""The training loss (equation 8) and its EM dispatch (Algorithms 3-9)."""
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import (BACKGROUND, BEAT, CLASS_UNKNOWN, DOWNBEAT,
                               F_MEASURE_TOLERANCE, METER_PRIOR)
from alignbeat.dp import subset_select_dp


# Nothing -- no CLI flag, no test -- ever sets these, so they are constants.
OMEGA_BEAT = 1.0              # eq. (8); only omega_DB is swept
FRAGMENT_SECONDS = 30.0       # 1500 frames at 50 fps; diagnostic display only.
                              # Also the unit t_hat lives in: eps below is this many
                              # seconds' worth of the (0, 1] window.

# Defaults for the arguments below, which only the tests vary.
B_MIN = 1e-4
NORMALIZE_BY_EVENTS = True
DIAGNOSTIC_EVERY = 200
# The tolerance in the units t_hat lives in: a fraction of the window.
EPS = F_MEASURE_TOLERANCE / FRAGMENT_SECONDS

# E-step time scale, eq. (3)'s 1/lambda_L1: fixed from the tolerance rather than
# estimated by eq. (5). At b = eps/2 the Laplace puts 86% of its mass inside the window
# and charges a linear tail beyond it (rubato, late annotations). Fixed so the
# class/time exchange rate does not drift as the regression sharpens: eq. (5)'s EMA ran
# 73 -> 28 ms over one run and steepened the cost into a gate by itself (E_estep).
ESTEP_B = EPS / 2.0

PRECISION_PRIOR_ALPHA = 2.0
PRECISION_PRIOR_BETA = None


class Match(NamedTuple):
    """What the E-step decided for one fragment."""
    sigma: object                   # (M,) numpy int array, the chosen candidates
    meter: object                   # L in force for this fragment, 0 if none
    r: object = None                # beat-only: P(event i is a downbeat), eq. (34)
    meter_posterior: object = None  # beat-only: P(L | x) as a dict keyed by L, eq. (33)
    prior_term: float = 0.0         # beat-only: -sum_h pi_h (log pi_M + log pi_psi)


class SubsetCriterion(nn.Module):
    """Per-pair cost (3), the selection DP, and the training loss (8)."""

    def __init__(self,
                 omega_downbeat=2.0, gamma=0.5,
                 normalize_by_events=NORMALIZE_BY_EVENTS,
                 background_by_unmatched=False,
                 b_min=B_MIN, residual_ema_decay=0.99,
                 precision_prior_alpha=PRECISION_PRIOR_ALPHA,
                 precision_prior_beta=PRECISION_PRIOR_BETA,
                 meter_candidates=()):
        super(SubsetCriterion, self).__init__()

        self.omega_downbeat = omega_downbeat
        self.gamma = gamma
        self.normalize_by_events = normalize_by_events
        self.background_by_unmatched = background_by_unmatched

        self.b_min = b_min
        self.estep_b = ESTEP_B
        # Running mean matched residual, eq. (5): used only as the Gamma prior's mode
        # (section 4.1.3's data-informed default). The E-step cost runs on ESTEP_B.
        self.register_buffer("residual_ema", torch.tensor(float(EPS)))
        self.residual_ema_decay = residual_ema_decay
        self.precision_prior_alpha = precision_prior_alpha
        self.precision_prior_beta = precision_prior_beta

        self.meter_candidates = tuple(meter_candidates)
        # log P(L, phi_0) = log P(L) - log L: the hypotheses are (L, phi_0) pairs and
        # phase is uniform within a meter, so without the -log L a meter of L gets L
        # times its share and the posterior drifts toward large meters.

        meter_prior_sum = sum(METER_PRIOR.get(L, 0.0) for L in tuple(meter_candidates))
        self.meter_prior = {int(L): math.log(METER_PRIOR.get(L, 0.0) / meter_prior_sum) - math.log(L)
                            for L in tuple(meter_candidates) if METER_PRIOR.get(L, 0.0) > 0.0}

        # pi_C(c) over {DB, B}: the base rate of downbeats among matched beats. No
        # background entry -- matching already established the event is a beat, so
        # "no beat" is not a live hypothesis for it. It is never configured, because
        # the meter prior already fixes it: exp(meter_prior[L]) is P(L)/L, so summing
        # over L is E[1/L], the chance an event drawn under that prior is a downbeat.
        # Deriving it keeps the two priors consistent whatever meter_candidates is.
        downbeat_share = sum(math.exp(v) for v in self.meter_prior.values())
        if not 0.0 < downbeat_share < 1.0:
            # No meter candidates, so no hypothesis is ever scored and pi_C is unused.
            downbeat_share = 0.5
        prior = torch.tensor([downbeat_share, 1.0 - downbeat_share], dtype=torch.float32)

        self._call_count = 0

        self.register_buffer("log_class_prior", torch.log(prior / prior.sum()), persistent=False)

        print(f"[subset-criterion] omega_db={self.omega_downbeat} gamma={self.gamma} "
              f"meter_candidates={self.meter_candidates or 'off'} "
              f"normalize_by_events={self.normalize_by_events}", flush=True)


    def l1(self, t_hat, t_target):
        """Eq. (3)'s time channel: plain L1 over the fixed E-step scale."""
        return (t_hat - t_target).abs() / self.estep_b

    def build_cost(self, log_probabilities, t_hat, gt_class, gt_time):
        """Per-pair cost (3) plus the section 8.4 background correction."""
        class_cost = self.class_nll(log_probabilities, gt_class)
        time_cost = self.l1(t_hat[None, :], gt_time[:, None])

        background_nll = -log_probabilities[:, BACKGROUND]                      # (N,)

        l_match = class_cost + time_cost                                        # eq. (3)
        return l_match - self.gamma * background_nll[None, :]                   # section 8.4

    def _fragment_meter(self, gt_class):
        """The meter L in force for this fragment, or 0 if none is available."""
        positions = (gt_class == DOWNBEAT).nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            return 0
        return int(np.median(np.diff(positions.cpu().numpy())))

    def class_nll(self, log_probabilities, gt_class):
        """-log p_j(c_i) for every (event, candidate) pair, handling unlabelled classes.

        A labelled event costs the NLL of its own class. An unlabelled one (beat-only
        data's B* label) is known to be an event but not which kind, so it costs the
        marginal -log(p_DB + p_B) instead, which steers the model toward neither.
        """
        # JA: CLASS_UNKNOWN is assigned to all events in the beat-only dataset
        labelled = gt_class != CLASS_UNKNOWN
        
        # JA: Algorithm 1 line 10. C_i is genuinely unknown here, so its own prior is
        # combined in: the cost is -log sum_c pi_C(c) p_j(c), the pi_C-weighted mixture
        # over {DB, B}. It stays phase- and meter-blind, since the DP needs Proposition
        # 5.1's additive separability and sigma is resolved before L is ever considered.
        mixture = torch.logsumexp(
            log_probabilities[:, [DOWNBEAT, BEAT]] + self.log_class_prior[None, :],
            dim=-1, keepdim=True)
        cost = -mixture.repeat(1, len(gt_class))                           # (N, M)

        cost[:, labelled] = -log_probabilities[:, gt_class[labelled]]

        return cost.transpose(0, 1)                                             # (M, N)

    def _e_step(self, log_probabilities, t_hat, gt_class, gt_time):
        """Algorithm 3 lines 1-9: the MAP estimate of sigma under the current theta."""
        with torch.no_grad():
            corrected = self.build_cost(log_probabilities, t_hat, gt_class, gt_time)

            fragment_meter = 0
            if (gt_class != CLASS_UNKNOWN).all():
                # JA: Calculate the meter from the downbeat annotation
                fragment_meter = self._fragment_meter(gt_class)

            if not bool(torch.isfinite(corrected).all()):
                # log p is floored at LOG_PROB_FLOOR and the time term is bounded, so a
                # non-finite cost means the model emitted NaN/Inf. Raise rather than
                # skip: a surviving batch would mask its own cause.
                raise FloatingPointError(
                    f"non-finite matching cost, M={gt_class.numel()}: the model "
                    f"produced NaN/Inf class logits or t_hat")

            # JA: Determine the value of latent variable sigma
            sigma_np = subset_select_dp(corrected.cpu().numpy())
            sigma = torch.from_numpy(sigma_np).to(log_probabilities.device)

            if bool((gt_class == CLASS_UNKNOWN).all()):
                match_likelihood = log_probabilities[sigma] # JA: This is pi_p_i

                # JA: Determine the value of latent variable L
                posterior = self._compute_latent_posterior(match_likelihood)
                if posterior is not None:
                    r, meter_posterior, prior_term = posterior
                    return Match(sigma_np, fragment_meter, r, meter_posterior,
                                 prior_term)

            return Match(sigma_np, fragment_meter)


    def _class_term(self, matched_class_probs, gt_class, target, match):
        """Algorithm 1 line 41's cls(theta; ind), branched on the supervision indicator.

        ind=0 (line 43) is an ordinary cross-entropy against the raw class head: the
        label is observed, so pi_C has no uncertainty left to resolve. ind=1 (line 45)
        is the EM surrogate under the frozen pi_{psi,L}.
        """

        beat_only = bool((gt_class == CLASS_UNKNOWN).all())
        num_unlabeled = 0
        meter_posterior = None

        if beat_only:
            meter_posterior = match.meter_posterior
            nll = self._beat_only_term(matched_class_probs, match.r)
            num_unlabeled = len(gt_class)
        else:
            # JA: The input music has both downbeat and beat labels
            event_indices = torch.arange(len(gt_class), device=gt_class.device)
            nll = -matched_class_probs[event_indices, gt_class]

        return nll, num_unlabeled, meter_posterior

    def _time_term(self, residual, laplace_scale, sigma, denominator):
        """Loss (8)'s second bracket, with section 4.1.2's per-candidate b_j if enabled.

        Unweighted: eq. (8) closes the omega_{c_i} bracket around the class term alone,
        so the timing channel -- and, under 4.1.2, the log 2 b_j precision channel with
        it -- is not scaled by the downbeat weight.
        """
        if laplace_scale is None:
            raise ValueError("laplace_scale is required: the head's per-candidate b_j is "
                             "the only scale there is")
        b_j = laplace_scale[sigma]
        return (self._per_candidate_time_term(residual, b_j).sum() / denominator,
                self._precision_prior(b_j) / denominator)

    def _m_step(self, match, log_probabilities, t_hat, target, laplace_scale):
        """Algorithm 3 lines 10-15: sigma held fixed, loss (8) built from p and t."""
        gt_class, gt_time = target['classes'], target['times']

        device = log_probabilities.device
        num_candidates = log_probabilities.shape[0]
        background_nll = -log_probabilities[:, BACKGROUND]

        sigma = torch.from_numpy(match.sigma).to(device)

        M = int(gt_class.numel())
        denominator = float(M) if self.normalize_by_events else 1.0

        omega = torch.where(
            gt_class == DOWNBEAT,
            torch.full_like(gt_time, self.omega_downbeat),
            torch.full_like(gt_time, OMEGA_BEAT))

        # line 13, first bracket
        per_event, unlabelled, meter_posterior = self._class_term(
            log_probabilities[sigma], gt_class, target, match)

        # line 13, second bracket. The loss sees the true error: a pointed Laplace, no
        # flat core, so t_hat is pulled onto the onset rather than merely inside the
        # tolerance. (Under the flat core the clock stopped at the window edge: slope
        # 0.63 of the needed shift, 24 ms mean error, b_hat parked at its 73 ms init.)
        matched_residual = (gt_time - t_hat[sigma]).abs()            # (M,), raw
        residual = matched_residual
        time_term, precision_term = self._time_term(
            residual, laplace_scale, sigma, denominator)

        # line 13, third sum: the candidates sigma did not match
        unmatched = torch.ones(num_candidates, dtype=torch.bool, device=device)
        unmatched[sigma] = False

        # The background sum runs over N-M candidates but is divided by M, so it carries
        # a weight of gamma*(N-M)/M relative to the class term -- 3.3 at 25 events, 0.5
        # at 92. A slow fragment's loss is then ~6x a fast one's and dominates the batch
        # gradient. Dividing it by N-M instead makes it a mean like the others, removing
        # the tempo dependence without changing the loss's overall scale.
        background_denominator = denominator
        if self.background_by_unmatched and self.normalize_by_events:
            background_denominator = float(max(num_candidates - M, 1))

        # Line 45's bracket also carries log pi_M + log pi_psi. Those are constant in
        # theta, so they change no gradient, but they belong to the reported cls(theta;
        # 1) and are outside omega, which eq. (8) closes around the class term alone.
        class_term = (omega * per_event).sum() + match.prior_term

        return {
            'class': class_term / denominator,
            'time': time_term,
            'background': background_nll[unmatched].sum() / background_denominator,
            'precision': precision_term,
            'residual': matched_residual.detach(),
            'unlabelled': unlabelled,
            'meter_posterior': meter_posterior,
        }

    def forward(self, class_logits, t_hat, b_hat, targets, train_precision=True):
        """One EM step per fragment; returns (losses, stats)."""
        batch_size, num_candidates, _ = class_logits.shape

        # JA: log_probabilities is the log probabilities of all N candidates
        log_probabilities = F.log_softmax(class_logits, dim=-1)

        # JA: b_hat is the output of the precision head which is learnable
        laplace_scale = self.b_min + b_hat # y = b
        precision_terms = []

        class_terms, time_terms, background_terms = [], [], []
        matched_residuals = []

        num_events, num_contributing, num_infeasible, num_unlabelled = 0, 0, 0, 0

        for b in range(batch_size):
            gt_class = targets[b]['classes']
            gt_time = targets[b]['times']
            M = int(gt_class.numel())

            background_nll = -log_probabilities[b, :, BACKGROUND]

            if M == 0:
                denominator = float(num_candidates) if self.normalize_by_events else 1.0
                background_terms.append(background_nll.sum() / denominator)
                num_contributing += 1
                continue

            if M > num_candidates:
                num_infeasible += 1
                print(f"[subset] WARNING: fragment with M={M} events > N={num_candidates} "
                      f"candidates skipped entirely (loss undefined; raise --num_candidates)",
                      flush=True)
                continue

            # E-step: MAP estimate of sigma under the current theta (Alg. 3, 1-9)
            match = self._e_step(log_probabilities[b], t_hat[b],
                                 gt_class, gt_time)

            # M-step: sigma fixed and the loss evaluated at it (Alg. 3, 10-15).
            terms = self._m_step(match, log_probabilities[b], t_hat[b], targets[b], laplace_scale[b])

            for key, bucket in (('class', class_terms), ('time', time_terms),
                                ('background', background_terms),
                                ('precision', precision_terms)):
                if terms[key] is not None:
                    bucket.append(terms[key])

            if terms['residual'] is not None:
                matched_residuals.append(terms['residual'])

            num_unlabelled += terms['unlabelled']
            num_events += M
            num_contributing += 1

        losses = self._aggregate(
            class_logits, class_terms, time_terms, background_terms,
            precision_terms, num_contributing)

        if self.training and matched_residuals:
            # Eq. (5) between steps, for the prior only.
            with torch.no_grad():
                batch_b = torch.cat(matched_residuals).mean().clamp(min=self.b_min)
                self.residual_ema.mul_(self.residual_ema_decay).add_(
                    (1.0 - self.residual_ema_decay) * batch_b)

        with torch.no_grad():
            stats = self._make_stats(
                losses, t_hat, num_candidates, matched_residuals,
                counts=dict(num_events=num_events, infeasible=num_infeasible,
                            unlabelled_events=num_unlabelled),
                b_hat_mean=float(laplace_scale.mean()),
                b_hat_min=float(laplace_scale.min()),
                b_hat_max=float(laplace_scale.max()))

        if self.training:
            self._call_count += 1
        if self._call_count % DIAGNOSTIC_EVERY == 1:
            self._log_diagnostic(stats, num_candidates)
        return losses, stats

    def _aggregate(self, class_logits, class_terms, time_terms, background_terms,
                   precision_terms, num_contributing):
        """Per-fragment terms -> the loss dict train.py unpacks."""
        zero = torch.nan_to_num(class_logits).sum() * 0.0
        total = lambda terms: torch.stack(terms).sum() if terms else zero
        n = max(num_contributing, 1) if self.normalize_by_events else 1

        losses = {
            'class': total(class_terms) / n,
            'time': total(time_terms) / n,
            'background': self.gamma * total(background_terms) / n,
        }
        losses['time'] = losses['time'] + total(precision_terms) / n
        losses['total'] = sum(losses[k] for k in ('class', 'time', 'background'))
        return losses

    def _make_stats(self, losses, t_hat, num_candidates, matched_residuals, counts,
                    b_hat_mean=float('nan'), b_hat_min=float('nan'),
                    b_hat_max=float('nan')):
        """Logging floats, refreshed on a diagnostic step and cached otherwise."""
        stats = {
            'cls': float(losses['class']), 'time': float(losses['time']),
            'bg': float(losses['background']), 'total': float(losses['total']),
            'b_hat_mean': b_hat_mean, 'b_hat_min': b_hat_min, 'b_hat_max': b_hat_max,
            **counts,
        }
        if matched_residuals:
            stats['residual_mean'] = float(torch.cat(matched_residuals).mean())
        with torch.no_grad():
            gaps = t_hat[:, 1:] - t_hat[:, :-1]
            stats['min_gap'] = float(gaps.min()) if gaps.numel() else float('inf')

        return stats

    def _log_diagnostic(self, stats, num_candidates):
        print(f"[subset] residual_ema={float(self.residual_ema) * FRAGMENT_SECONDS * 1000:.0f}ms "
              f"b_hat={stats['b_hat_mean']:.5f} "
              f"[{stats['b_hat_min']:.5f}, {stats['b_hat_max']:.5f}] "
              f"({stats['b_hat_mean'] * FRAGMENT_SECONDS * 1000:.0f}ms) | "
              f"residual={stats.get('residual_mean', float('nan')):.5f} "
              f"({stats.get('residual_mean', 0.0) * FRAGMENT_SECONDS * 1000:.0f}ms) "
              f"min_gap={stats['min_gap']:.2e} events={stats['num_events']} "
              f"infeasible={stats['infeasible']}", flush=True)

    def _beat_only_term(self, matched_log, r):
        """Algorithm 1 line 45's cls(theta; 1), per event, before the outer sum.

        The EM surrogate, not the direct marginal: pi_{psi,L} is frozen at theta_old
        and line 45 takes its expectation of the complete-data log-likelihood. That
        expectation factorises per event -- for each i the hypotheses split into those
        claiming DB and those claiming B, and pi summed over the first group is exactly
        r_i -- so the two-term form below is the surrogate itself, not an approximation.

        The factors are P_hat(C_i = c | x; theta), prior-combined with pi_C and
        normalised over {DB, B}, which is the quantity line 45 names. The raw head
        appears only in the ind=0 branch, line 43.
        """
        if r is None:
            # No viable meter hypothesis, so there is no pi_{psi,L} to take an
            # expectation under. Fall back to what the label alone asserts: the event
            # is a beat of some kind.
            return -torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)

        log_db = matched_log[:, DOWNBEAT] + self.log_class_prior[DOWNBEAT]
        log_b = matched_log[:, BEAT] + self.log_class_prior[BEAT]
        log_norm = torch.logaddexp(log_db, log_b)
        return -(r * (log_db - log_norm) + (1.0 - r) * (log_b - log_norm))

    def _log_hypothesis_prior(self, meter):
        """log pi_M(L) + log pi_psi(psi), Algorithm 1 line 45's first two bracket terms.

        pi_psi is uniform over the L phases, so log pi_psi = -log L and the pair is the
        same under every psi of a given meter. meter_prior stores the two combined.
        """
        if self.meter_prior is None:
            return -math.log(meter)
        return self.meter_prior.get(meter, -float('inf'))

    def _hypothesis_log_scores(self, match_likelihood):
        """The log score of every (L, phi_0 = p) hypothesis: eq. (12)'s numerator in log space.

        The window crop is arbitrary, so the span may begin anywhere in the bar and
        every phi_0 in range(L) is open.

        Vectorised over hypotheses: one (H, M) downbeat mask and one matmul, rather than
        a where+sum per hypothesis. With gradient on, the per-hypothesis loop launched
        ~56 kernels per labelled fragment and cost 45 ms per batch; this costs two.
        """
        M = match_likelihood.shape[0]
        device = match_likelihood.device

        i0 = torch.arange(M, device=device)
        blocks = {}

        log_db = match_likelihood[:, DOWNBEAT] + self.log_class_prior[DOWNBEAT]
        log_b = match_likelihood[:, BEAT] + self.log_class_prior[BEAT]
        # P_hat normalises over {DB, B}: one log Z_i per event, identical under every
        # hypothesis, so it cancels in the posterior but not in the loss's own value.
        log_norm = torch.logaddexp(log_db, log_b).sum()

        for meter in self.meter_candidates:
            meter = int(meter)
            # A meter of 1 makes every event a downbeat, so it carries no phase to
            # infer. Requiring M >= L is a choice about short fragments rather than a
            # well-definedness guard: c_i(omega, L) is defined for any L, but meters
            # larger than the event count generate indistinguishable patterns, so they
            # would only spread the prior's mass. It fires only on degenerate crops.
            if meter <= 1 or M < meter:
                continue

            log_prior = self._log_hypothesis_prior(meter)

            phases = torch.arange(meter, device=device)
            # is_db[p, i]: event i is a downbeat under phi_0 = p, i.e. (p + i) % L == 0.
            is_db = (((phases[:, None] + i0[None, :]) % meter) == 0).to(match_likelihood.dtype)

            blocks[meter] = (is_db @ log_db + (1.0 - is_db) @ log_b + log_prior - log_norm)

        if not blocks:
            return None

        return blocks


    def _meter_log_posterior(self, match_likelihood):
        """Eq. (33), log P(L | x), for every viable L, differentiable in span."""
        blocks = self._hypothesis_log_scores(match_likelihood)
        if blocks is None:
            return None
        meters = list(blocks)
        log_joint = torch.log_softmax(torch.cat([blocks[L] for L in meters]), dim=0)
        out, start = {}, 0
        for meter in meters:
            out[meter] = log_joint[start:start + meter].logsumexp(dim=0)
            start += meter
        return out


    def _compute_latent_posterior(self, match_likelihood):
        """Algorithm 1 lines 22-31: pi_{psi,L}, the E-step's joint responsibility.

        Returns (r, meter_posterior, prior_term). r_i is pi_{psi,L} summed over the
        hypotheses that call event i a downbeat, which is all the M-step needs: line
        45's expectation factorises per event, since for each i the hypotheses split
        into those claiming DB and those claiming B. prior_term is the bracket's own
        first two terms, -sum_h pi_h (log pi_M + log pi_psi), constant in theta but
        part of the reported loss.
        """
        blocks = self._hypothesis_log_scores(match_likelihood)
        if blocks is None:
            return None
        M = match_likelihood.shape[0]
        device = match_likelihood.device

        meters = list(blocks)

        # JA: This is pi_{psi,L} in log space
        log_joint = torch.log_softmax(torch.cat([blocks[L] for L in meters]), dim=0)

        i0 = torch.arange(M, device=device)
        r = torch.zeros(M, device=device, dtype=match_likelihood.dtype)
        meter_posterior, prior_term, start = {}, 0.0, 0
        for meter in meters:
            block = log_joint[start:start + meter]
            start += meter
            # Line 45's own singleton: within a meter exactly one psi calls event i a
            # downbeat, namely psi = (-i) mod L, so no sum over psi is needed here.
            r = r + block[(-i0) % meter].exp()
            mass = float(block.logsumexp(dim=0).exp())
            meter_posterior[meter] = mass
            prior_term -= mass * self._log_hypothesis_prior(meter)

        return r, meter_posterior, prior_term

    def _per_candidate_time_term(self, residual, b_j):
        """-log p(r | b_j) for the Laplace density exp(-r/b)/(2b), split per 4.1.3.

        log(2 b_j) is the normaliser that makes this a likelihood in b_j: without it
        b_j -> inf minimises the loss and the timing channel switches itself off. Its
        stationary point is b_j = mean residual, the actual timing error.
        """
        localisation = residual / b_j.detach()                       # gradient to t_hat
        precision = residual.detach() / b_j + torch.log(2.0 * b_j)   # to b_j
        return localisation + precision

    def _precision_prior(self, b_j):
        """Mitigation two: a Gamma prior on the precision 1/b_j, as a MAP term.

        Its mode is eq. (5)'s running mean residual, section 4.1.3's data-informed
        default, so each b_j is shrunk toward the data rather than toward the tolerance.
        """
        alpha = self.precision_prior_alpha
        beta = (self.precision_prior_beta if self.precision_prior_beta is not None
                else float(self.residual_ema) * max(alpha - 1.0, 1e-6))
        return ((alpha - 1.0) * torch.log(b_j) + beta / b_j).sum()

