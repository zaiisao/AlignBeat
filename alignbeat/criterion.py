"""The training loss (equation 8) and its EM dispatch (Algorithms 3-9)."""
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import (BACKGROUND, BEAT, CLASS_UNKNOWN, DOWNBEAT,
                               F_MEASURE_TOLERANCE)
from alignbeat.dp import subset_select_dp


# Nothing -- no CLI flag, no test -- ever sets these, so they are constants.
FRAGMENT_SECONDS = 30.0       # 1500 frames at 50 fps; diagnostic display only.
                              # Also the unit t_hat lives in: eps below is this many
                              # seconds' worth of the (0, 1] window.

# Defaults for the arguments below, which only the tests vary.
DIAGNOSTIC_EVERY = 200
# The tolerance in the units t_hat lives in: a fraction of the window.
EPS = F_MEASURE_TOLERANCE / FRAGMENT_SECONDS

# E-step time scale, eq. (3)'s 1/lambda_L1: fixed from the tolerance rather than
# estimated by eq. (5). At b = eps/2 the Laplace puts 86% of its mass inside the window
# and charges a linear tail beyond it (rubato, late annotations). Fixed so the
# class/time exchange rate does not drift as the regression sharpens: eq. (5)'s EMA ran
# 73 -> 28 ms over one run and steepened the cost into a gate by itself (E_estep).
# lambda_L1, Algorithm 1's own weight on the timing term. Fixed from the tolerance:
# at b = eps/2 a Laplace puts 86% of its mass inside the F-measure window, so
# lambda_L1 = 1/b = 2/eps trades one nat of class cost for eps/2 of timing error.
LAMBDA_L1 = 2.0 / EPS



class Match(NamedTuple):
    """What the E-step decided for one fragment."""
    sigma: object                   # (M,) numpy int array, the chosen candidates
    pi: object = None               # beat-only: pi_{omega,L}, line 40, frozen (line 46)


class SubsetCriterion(nn.Module):
    """Per-pair cost (3), the selection DP, and the training loss (8)."""

    def __init__(self,
                 data_prior, lambda_l1=LAMBDA_L1, gamma=0.5,
                 meter_prior=None):
        super(SubsetCriterion, self).__init__()

        self.lambda_l1 = lambda_l1
        self.gamma = gamma
        self.meter_candidates = tuple(sorted(int(L) for L in meter_prior)) if meter_prior else ()
        self.meter_prior = ({int(L): math.log(p) - math.log(int(L))
                             for L, p in meter_prior.items() if p > 0.0}
                            if meter_prior else None)

        downbeat_share = (sum(math.exp(v) for v in self.meter_prior.values())
                          if self.meter_prior else 0.0)

        if not 0.0 < downbeat_share < 1.0:
            # No meter candidates, so no hypothesis is ever scored and pi_C is unused.
            downbeat_share = 0.5
        prior = torch.tensor([downbeat_share, 1.0 - downbeat_share], dtype=torch.float32)

        self._call_count = 0

        # Persistent: Algorithm 3's Require lists pi_C and pi_data as inference inputs,
        # and a deployed model has no training split to recount them from. They ride
        # in the checkpoint.
        self.register_buffer("log_class_prior", torch.log(prior / prior.sum()))

        # pi_data(c): the DB:B balance the head was trained on, measured per fold by
        # BeatDataModule.get_train_class_prior. Section 1.3 divides it out of the raw
        # head before pi_C is applied, so the desired prior REPLACES the training set's
        # own rather than layering on top of it.
        data_priors = torch.tensor([data_prior["downbeat"], data_prior["beat"]],
                                    dtype=torch.float32)
        log_data_prior = torch.log(data_priors / data_priors.sum())

        self.register_buffer("log_data_prior", log_data_prior)

        print(f"[subset-criterion] lambda_L1={self.lambda_l1:g} gamma={self.gamma} "
              f"meter_candidates={self.meter_candidates or 'off'}", flush=True)


    def l1(self, t_hat, t_target):
        """Eq. (3)'s time channel: plain L1 over the fixed E-step scale."""
        return self.lambda_l1 * (t_hat - t_target).abs()

    def build_cost(self, log_probabilities, t_hat, gt_class, gt_time):
        """Algorithm 1 lines 18 and 20. Two terms: the class cost of the event this
        candidate would explain, and lambda_L1 times the timing error. No background
        correction, no prior -- class_nll branches on ind for us."""
        class_cost = self.class_nll(log_probabilities, gt_class)
        time_cost = self.l1(t_hat[None, :], gt_time[:, None])

        return class_cost + time_cost

    def class_nll(self, log_probabilities, gt_class):
        """-log p_j(c_i) for every (event, candidate) pair, handling unlabelled classes.

        A labelled event costs the NLL of its own class. An unlabelled one (beat-only
        data's B* label) is known to be an event but not which kind, so it costs the
        marginal -log(p_DB + p_B) instead, which steers the model toward neither.
        """
        # JA: CLASS_UNKNOWN is assigned to all events in the beat-only dataset
        labelled = gt_class != CLASS_UNKNOWN
        
        # algorithm5_hard1 Algorithm 1 line 20: -log(1 - p_j(empty)), which is
        # -log(p_DB + p_B) since the three classes sum to 1. The raw head, no prior:
        # line 20 is the definition for ind=1, not a configurable alternative.
        mixture = torch.logsumexp(
            log_probabilities[:, [DOWNBEAT, BEAT]], dim=-1, keepdim=True)
        cost = -mixture.repeat(1, len(gt_class))                           # (N, M)

        cost[:, labelled] = -log_probabilities[:, gt_class[labelled]]

        return cost.transpose(0, 1)                                             # (M, N)

    def _e_step(self, log_probabilities, t_hat, gt_class, gt_time):
        """Algorithm 1 lines 4-40, on one fragment, at theta_old."""
        with torch.no_grad():
            beat_only = bool((gt_class == CLASS_UNKNOWN).all())

            # Lines 5-12: C's own posterior at EVERY candidate j, via Bayes' rule, only
            # when ind = 1. Per-candidate independent, so this is the same as evaluating
            # it later at sigma_hat(i) -- the algorithm computes it here, and line 36
            # reads it back at the matched candidates.
            class_posterior = (self._class_log_posterior(log_probabilities)
                               if beat_only else None)

            # Lines 15-23
            corrected = self.build_cost(log_probabilities, t_hat, gt_class, gt_time)

            # Line 26: resolve sigma_hat
            sigma_np = subset_select_dp(corrected.cpu().numpy())
            sigma = torch.from_numpy(sigma_np).to(log_probabilities.device)

            if beat_only:
                # Line 36: q_i(c) <- P_hat(C=c | x, sigma_hat(i)), read back at the
                # matched candidates rather than re-derived.
                q = tuple(channel[sigma] for channel in class_posterior)
                # Lines 39-41: the joint responsibility itself. This is the E-step's
                # whole output -- pi over the (omega, L) hypotheses. Everything the
                # M-step needs is a function of pi and of theta, so nothing else
                # crosses the boundary.
                scores = self._hypothesis_log_scores(q)
                if scores is not None:
                    flat = torch.cat([scores[L] for L in scores])
                    return Match(sigma_np, torch.softmax(flat, dim=0))

            return Match(sigma_np)


    def _class_term(self, matched_class_probs, gt_class, target, match):
        """Algorithm 2 lines 49-53's cls(theta; ind), branched on the indicator.

        ind=0 (line 50) is an ordinary cross-entropy against the raw class head: the
        label is observed, so no prior has uncertainty left to resolve. ind=1 (line 52)
        is the EM surrogate under the frozen pi_{omega,L}.
        """

        beat_only = bool((gt_class == CLASS_UNKNOWN).all())
        num_unlabeled = 0

        if beat_only:
            term = self._beat_only_term(matched_class_probs, match.pi)
            num_unlabeled = len(gt_class)
        else:
            # JA: The input music has both downbeat and beat labels
            event_indices = torch.arange(len(gt_class), device=gt_class.device)
            term = -matched_class_probs[event_indices, gt_class].sum()

        return term, num_unlabeled

    def _time_term(self, residual):
        """Line 56's timing term: lambda_L1 |t_i - t_hat_sigma(i)|, summed over events."""
        return self.lambda_l1 * residual.sum()

    def _m_step(self, match, log_probabilities, t_hat, target):
        """Algorithm 2 lines 48-56: sigma_hat held fixed, the loss built at theta."""
        gt_class, gt_time = target['classes'], target['times']

        device = log_probabilities.device
        num_candidates = log_probabilities.shape[0]
        background_nll = -log_probabilities[:, BACKGROUND]

        sigma = torch.from_numpy(match.sigma).to(device)

        # line 56, first term: cls(theta; ind)
        class_term, unlabelled = self._class_term(
            log_probabilities[sigma], gt_class, target, match)

        residual = (gt_time - t_hat[sigma]).abs()            # (M,), raw
        time_term = self._time_term(residual)

        # line 56, third term: the candidates sigma_hat did not match
        unmatched = torch.ones(num_candidates, dtype=torch.bool, device=device)
        unmatched[sigma] = False

        return {
            'class': class_term,
            'time': time_term,
            'background': background_nll[unmatched].sum(),
            'residual': residual.detach(),
            'unlabelled': unlabelled,
        }

    def forward(self, class_logits, t_hat, targets):
        """One EM step per fragment; returns (losses, stats)."""
        batch_size, num_candidates, _ = class_logits.shape

        # JA: log_probabilities is the log probabilities of all N candidates
        log_probabilities = F.log_softmax(class_logits, dim=-1)

        class_terms, time_terms, background_terms = [], [], []
        matched_residuals = []

        num_events, num_contributing, num_infeasible, num_unlabelled = 0, 0, 0, 0

        for b in range(batch_size):
            gt_class = targets[b]['classes']
            gt_time = targets[b]['times']
            M = int(gt_class.numel())

            background_nll = -log_probabilities[b, :, BACKGROUND]

            if M == 0:
                background_terms.append(background_nll.sum())
                num_contributing += 1
                continue

            if M > num_candidates:
                num_infeasible += 1
                print(f"[subset] WARNING: fragment with M={M} events > N={num_candidates} "
                      f"candidates skipped entirely (loss undefined; raise --num_candidates)",
                      flush=True)
                continue

            # E-step: MAP estimate of sigma under the current theta (Alg. 3, 1-9)
            match = self._e_step(log_probabilities[b], t_hat[b], gt_class, gt_time)

            # M-step: sigma fixed and the loss evaluated at it (Alg. 3, 10-15).
            terms = self._m_step(match, log_probabilities[b], t_hat[b], targets[b])

            for key, bucket in (('class', class_terms), ('time', time_terms),
                                ('background', background_terms)):
                if terms[key] is not None:
                    bucket.append(terms[key])

            if terms['residual'] is not None:
                matched_residuals.append(terms['residual'])

            num_unlabelled += terms['unlabelled']
            num_events += M
            num_contributing += 1

        losses = self._aggregate(
            class_logits, class_terms, time_terms, background_terms,
            num_contributing)

        with torch.no_grad():
            stats = self._make_stats(
                losses, t_hat, num_candidates, matched_residuals,
                counts=dict(num_events=num_events, infeasible=num_infeasible,
                            unlabelled_events=num_unlabelled))

        if self.training:
            self._call_count += 1
        if self._call_count % DIAGNOSTIC_EVERY == 1:
            self._log_diagnostic(stats, num_candidates)
        return losses, stats

    def _aggregate(self, class_logits, class_terms, time_terms, background_terms,
                   num_contributing):
        """Per-fragment terms -> the loss dict train.py unpacks."""
        zero = torch.nan_to_num(class_logits).sum() * 0.0
        total = lambda terms: torch.stack(terms).sum() if terms else zero

        losses = {
            'class': total(class_terms),
            'time': total(time_terms),
            'background': self.gamma * total(background_terms),
        }
        losses['total'] = sum(losses[k] for k in ('class', 'time', 'background'))
        return losses

    def _make_stats(self, losses, t_hat, num_candidates, matched_residuals, counts,
                    ):
        """Logging floats, refreshed on a diagnostic step and cached otherwise."""
        stats = {
            'cls': float(losses['class']), 'time': float(losses['time']),
            'bg': float(losses['background']), 'total': float(losses['total']),
            **counts,
        }
        if matched_residuals:
            stats['residual_mean'] = float(torch.cat(matched_residuals).mean())
        with torch.no_grad():
            gaps = t_hat[:, 1:] - t_hat[:, :-1]
            stats['min_gap'] = float(gaps.min()) if gaps.numel() else float('inf')

        return stats

    def _log_diagnostic(self, stats, num_candidates):
        print(f"[subset] residual={stats.get('residual_mean', float('nan')):.5f} "
              f"({stats.get('residual_mean', 0.0) * FRAGMENT_SECONDS * 1000:.0f}ms) "
              f"min_gap={stats['min_gap']:.2e} events={stats['num_events']} "
              f"infeasible={stats['infeasible']}", flush=True)

    def _log_likelihood(self, log_p):
        """Algorithm 1 line 7: l_hat_j(x | c) := p_hat_j(c | x) / pi_data(c).

        The raw head is a posterior -- cross-entropy drives it to the training data's
        own P(c | x), which carries pi_data uninvited. Dividing that out leaves a ratio
        proportional to P_data(x | c): likelihood-shaped in what it represents, even
        though the network never computes anything x-targeted. Named rather than
        inlined because it is the piece that plays the likelihood's role in line 10,
        and because applying pi_C without this division would double-count a prior.

        Returns (log l_hat(DB), log l_hat(B)); unnormalised, as a likelihood is.
        """
        return (log_p[:, DOWNBEAT] - self.log_data_prior[DOWNBEAT],
                log_p[:, BEAT] - self.log_data_prior[BEAT])

    def _class_log_posterior(self, log_p):
        """Algorithm 1 line 10: P_hat(C = c | x, j) = pi_C(c) l_hat_j / sum_c' ...

        Bayes' rule over {DB, B}, whose support excludes empty because matching has
        already established the event is a real beat.

        Both consumers must use this same quantity. pi_{omega,L} is frozen from the
        hypothesis scores and the M-step surrogate is its expectation, so Fisher's
        identity holds only if the two are built from one complete-data log-likelihood
        (tests/test_phase.py catches a divergence). Keeping it in one place is what
        makes that structural rather than a comment.
        """
        log_l_db, log_l_b = self._log_likelihood(log_p)
        log_db = log_l_db + self.log_class_prior[DOWNBEAT]
        log_b = log_l_b + self.log_class_prior[BEAT]
        log_norm = torch.logaddexp(log_db, log_b)
        return log_db - log_norm, log_b - log_norm

    def _beat_only_term(self, matched_log, pi):
        """Algorithm 2 line 52's cls(theta; 1).

            cls(theta; 1) = - sum_{L, omega} pi_{omega,L} [ log pi_M(L) + log pi_omega
                                              + sum_i log P_hat(C_i = c_i(omega, L)) ]

        The bracket is exactly the hypothesis score _hypothesis_log_scores builds, so
        the surrogate is one dot product: pi frozen at theta_old against the same
        scores recomputed at theta. The EM structure is the two arguments -- pi carries
        no gradient, the scores carry all of it.
        """
        if pi is None:
            # No viable meter hypothesis, so there is no pi_{omega,L} to take an
            # expectation under. Fall back to what the label alone asserts: the event
            # is a beat of some kind.
            return -torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1).sum()

        scores = self._hypothesis_log_scores(self._class_log_posterior(matched_log))
        flat = torch.cat([scores[L] for L in scores])
        return -(pi * flat).sum()

    def _log_hypothesis_prior(self, meter):
        """log pi_M(L) + log pi_omega(omega), Algorithm 2 line 52's first two brackets.

        pi_omega is uniform over the L phases, so log pi_omega = -log L and the pair is
        the same under every omega of a given meter. meter_prior stores them combined.
        """
        if self.meter_prior is None:
            return -math.log(meter)
        return self.meter_prior.get(meter, -float('inf'))

    def _hypothesis_log_scores(self, q):
        """Algorithm 1 line 40's numerator: the log score of every (omega, L) hypothesis.

        q is (log P_hat(DB), log P_hat(B)) per event, as lines 5-12 produced it and
        line 36 read it back at the matched candidates. Taking it rather than raw
        logits is what keeps the algorithm's own order: the posterior is built once,
        before sigma, and this only scores patterns against it.

        The window crop is arbitrary, so the span may begin anywhere in the bar and
        every phi_0 in range(L) is open.

        Vectorised over hypotheses: one (H, M) downbeat mask and one matmul, rather than
        a where+sum per hypothesis. With gradient on, the per-hypothesis loop launched
        ~56 kernels per labelled fragment and cost 45 ms per batch; this costs two.
        """
        log_db, log_b = q
        M = log_db.shape[0]
        device = log_db.device

        i0 = torch.arange(M, device=device)
        blocks = {}

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
            is_db = (((phases[:, None] + i0[None, :]) % meter) == 0).to(log_db.dtype)

            blocks[meter] = is_db @ log_db + (1.0 - is_db) @ log_b + log_prior

        if not blocks:
            return None

        return blocks


    def _meter_log_posterior(self, match_likelihood):
        """log P(L | x) for every viable L, from raw matched class log-probabilities.

        Convenience for the diagnostics, which start from a checkpoint's logits rather
        than from an E-step that already built the posterior.
        """
        blocks = self._hypothesis_log_scores(self._class_log_posterior(match_likelihood))
        if blocks is None:
            return None
        meters = list(blocks)
        log_joint = torch.log_softmax(torch.cat([blocks[L] for L in meters]), dim=0)
        out, start = {}, 0
        for meter in meters:
            out[meter] = log_joint[start:start + meter].logsumexp(dim=0)
            start += meter
        return out
