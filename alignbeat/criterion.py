"""The training loss (equation 8) and its EM dispatch (Algorithms 3-9)."""
import math
from typing import NamedTuple

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
                 meter_prior=None, match_event_cost=False, loss_event_term=False):
        super(SubsetCriterion, self).__init__()

        self.lambda_l1 = lambda_l1
        self.gamma = gamma
        # The two deviations from algorithm5_hard-1, each independently switchable so
        # an arm can price them apart. False on both is the algorithm as written.
        self.match_event_cost = match_event_cost      # line 20's -log(1 - p(empty))
        self.loss_event_term = loss_event_term        # the same quantity in line 52
        self.meter_candidates = tuple(sorted(int(L) for L in meter_prior)) if meter_prior else ()
        # pi_M(L) pi_omega(omega), combined: pi_omega is uniform over the L phases, so
        # the pair is the same under every phase of a given meter.
        self.meter_prior = ({int(L): p / int(L)
                             for L, p in meter_prior.items() if p > 0.0}
                            if meter_prior else None)

        downbeat_share = (sum(self.meter_prior.values())
                          if self.meter_prior else 0.0)

        if not 0.0 < downbeat_share < 1.0:
            # No meter candidates, so no hypothesis is ever scored and pi_C is unused.
            downbeat_share = 0.5
        prior = torch.tensor([downbeat_share, 1.0 - downbeat_share], dtype=torch.float32)

        self._call_count = 0

        # Persistent: Algorithm 3's Require lists pi_C and pi_data as inference inputs,
        # and a deployed model has no training split to recount them from. They ride
        # in the checkpoint.
        self.register_buffer("class_prior", prior / prior.sum())

        # pi_data(c): the DB:B balance the head was trained on, measured per fold by
        # BeatDataModule.get_train_class_prior. Section 1.3 divides it out of the raw
        # head before pi_C is applied, so the desired prior REPLACES the training set's
        # own rather than layering on top of it.
        data_priors = torch.tensor([data_prior["downbeat"], data_prior["beat"]],
                                    dtype=torch.float32)

        self.register_buffer("data_prior", data_priors / data_priors.sum())

        print(f"[subset-criterion] lambda_L1={self.lambda_l1:g} gamma={self.gamma} "
              f"meter_candidates={self.meter_candidates or 'off'} "
              f"match_event_cost={self.match_event_cost} "
              f"loss_event_term={self.loss_event_term}", flush=True)


    def l1(self, t_hat, t_target):
        """Eq. (3)'s time channel: plain L1 over the fixed E-step scale."""
        return self.lambda_l1 * (t_hat - t_target).abs()

    def build_l_match(self, log_probabilities, t_hat, gt_class, gt_time):
        """Algorithm 1 lines 18 and 20: L_match(i, j), returned (M, N) for the DP.
        Line 18 charges the observed class's NLL; line 20 has no class to charge, so an
        unlabelled event pays the timing error alone."""
        labelled = gt_class != CLASS_UNKNOWN

        # Line 19's L1, already scaled by lambda_L1.
        time_cost = self.l1(t_hat[None, :], gt_time[:, None])                   # (M, N)

        # gt_class is CLASS_UNKNOWN at the unlabelled events, which would index the
        # background column. Clamp so the gather stays on a real class; where() then
        # discards those entries. Per event rather than per fragment, so a fragment
        # carrying both kinds of annotation gets line 18 on the events that have a
        # label and line 20 on the events that do not.
        class_cost = -log_probabilities[:, gt_class.clamp(min=0)].transpose(0, 1)

        # DEVIATION, off by default: line 20's own -log(1 - p_j(empty)), the mass the
        # head puts on the event classes, charged to unlabelled events instead of zero.
        unlabelled_cost = 0.0
        if self.match_event_cost:
            unlabelled_cost = -torch.logsumexp(
                log_probabilities[:, [DOWNBEAT, BEAT]], dim=-1)[None, :]

        return torch.where(labelled[:, None], class_cost, unlabelled_cost) + time_cost

    def _e_step(self, class_logits, t_hat, gt_class, gt_time):
        """Algorithm 1 lines 4-40, on one fragment, at theta_old. Takes the logits
        because the algorithm wants the head both ways: lines 18 and 20 write log p_hat,
        lines 7 and 10 write p_hat, and both come straight off them."""
        with torch.no_grad():
            has_class_labels = bool((gt_class != CLASS_UNKNOWN).any())

            # Lines 15-23: L_match(i, j).
            l_match = self.build_l_match(F.log_softmax(class_logits, dim=-1),
                                         t_hat, gt_class, gt_time)

            # Line 26: sigma_hat <- SubsetSelectDP(L_match).
            sigma = subset_select_dp(l_match.cpu().numpy())

            if has_class_labels:
                # ind = 0: sigma_hat is the whole E-step. Lines 5-12 and 36-40 below
                # are the ind = 1 branch and have nothing to contribute here.
                return Match(sigma)

            # Lines 5-12: P_hat(C = c | x, j) at EVERY candidate. The algorithm builds
            # it before sigma_hat; it is per-candidate independent, so building it here
            # and reading it back at sigma_hat(i) gives the same numbers.
            class_posterior = self._class_posterior(F.softmax(class_logits, dim=-1))

            # Line 36: q_i(c) <- P_hat(C = c | x, sigma_hat(i)).
            matched_candidates = torch.from_numpy(sigma).to(class_logits.device)
            matched_class_posterior = tuple(channel[matched_candidates]
                                            for channel in class_posterior)

            # Line 39: the numerator, pi_M(L) pi_omega(omega) prod_i q_i(c_i(omega, L)).
            scores_by_meter = self._meter_phase_scores(matched_class_posterior)
            if scores_by_meter is None:
                return Match(sigma)

            # Line 40: pi_{omega,L} <- that numerator over its own sum across every
            # (omega, L) pair. This is the E-step's whole output: everything the M-step
            # needs is a function of pi and theta, so nothing else crosses the boundary.
            meter_phase_scores = torch.cat(
                [scores_by_meter[meter] for meter in scores_by_meter])
            return Match(sigma, meter_phase_scores / meter_phase_scores.sum())


    def _class_term(self, matched_class_probs, gt_class, match):
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

        # lines 49-56, first term: cls(theta; ind)
        class_term, unlabelled = self._class_term(
            log_probabilities[sigma], gt_class, match)

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

        num_events, num_infeasible, num_unlabelled = 0, 0, 0

        for b in range(batch_size):
            gt_class = targets[b]['classes']
            gt_time = targets[b]['times']
            M = int(gt_class.numel())

            background_nll = -log_probabilities[b, :, BACKGROUND]

            if M == 0:
                background_terms.append(background_nll.sum())
                continue

            if M > num_candidates:
                num_infeasible += 1
                print(f"[subset] WARNING: fragment with M={M} events > N={num_candidates} "
                      f"candidates skipped entirely (loss undefined; raise --num_candidates)",
                      flush=True)
                continue

            # E-step: sigma_hat under the current theta (Algorithm 1, lines 4-40)
            match = self._e_step(class_logits[b], t_hat[b], gt_class, gt_time)

            # M-step: sigma_hat fixed, the loss built at theta (Algorithm 2, lines 48-56)
            terms = self._m_step(match, log_probabilities[b], t_hat[b], targets[b])

            for key, bucket in (('class', class_terms), ('time', time_terms),
                                ('background', background_terms)):
                if terms[key] is not None:
                    bucket.append(terms[key])

            if terms['residual'] is not None:
                matched_residuals.append(terms['residual'])

            num_unlabelled += terms['unlabelled']
            num_events += M

        losses = self._aggregate(
            class_logits, class_terms, time_terms, background_terms)

        with torch.no_grad():
            stats = self._make_stats(
                losses, t_hat, matched_residuals,
                counts=dict(num_events=num_events, infeasible=num_infeasible,
                            unlabelled_events=num_unlabelled))

        if self.training:
            self._call_count += 1
        if self._call_count % DIAGNOSTIC_EVERY == 1:
            self._log_diagnostic(stats)
        return losses, stats

    def _aggregate(self, class_logits, class_terms, time_terms, background_terms):
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

    def _make_stats(self, losses, t_hat, matched_residuals, counts):
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

    def _log_diagnostic(self, stats):
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
        return (log_p[:, DOWNBEAT] - self.data_prior[DOWNBEAT].log(),
                log_p[:, BEAT] - self.data_prior[BEAT].log())

    def _class_posterior(self, p):
        """Algorithm 1 lines 7 and 10: Bayes over {DB, B}, pi_data divided out first.
        Neither line carries a logarithm, so neither does this; _class_log_posterior is
        the same quantity for line 52. float64 because a rejected class underflows f32."""
        p = p.double()
        prior = self.class_prior.double()
        data_prior = self.data_prior.double()

        likelihood_db = p[:, DOWNBEAT] / data_prior[DOWNBEAT]
        likelihood_b = p[:, BEAT] / data_prior[BEAT]

        joint_db = prior[DOWNBEAT] * likelihood_db
        joint_b = prior[BEAT] * likelihood_b
        total = joint_db + joint_b

        return joint_db / total, joint_b / total

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
        log_db = log_l_db + self.class_prior[DOWNBEAT].log()
        log_b = log_l_b + self.class_prior[BEAT].log()
        log_norm = torch.logaddexp(log_db, log_b)
        return log_db - log_norm, log_b - log_norm

    def _beat_only_term(self, matched_log, pi):
        """Algorithm 2 line 52's cls(theta; 1): pi frozen at theta_old, dotted into the
        same bracket recomputed at theta. P_hat is normalised over {DB, B}, so this says
        nothing about whether a matched candidate is an event at all.
        """
        # DEVIATION, off by default: line 50 decomposes into which class the event is
        # and whether it is an event at all. Line 52 supplies only the first, so nothing
        # constrains p(empty) at a matched event on beat-only data. This restores it.
        event = (-torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1).sum()
                 if self.loss_event_term
                 else torch.zeros((), dtype=matched_log.dtype,
                                  device=matched_log.device))

        if pi is None:
            # No viable meter hypothesis, so there is no pi_{omega,L} to take an
            # expectation under and line 52 has no value.
            return event

        scores_by_meter = self._log_meter_phase_scores(
            self._class_log_posterior(matched_log))
        meter_phase_scores = torch.cat(
            [scores_by_meter[meter] for meter in scores_by_meter])
        return event - (pi * meter_phase_scores).sum()

    def infer_pattern(self, p):
        """Algorithm 3 lines 10-24: resolve one (omega, L) for these events and label
        every one of them from it.

        p is the head's probabilities at the DETECTED candidates only, already ordered
        by increasing t_hat (line 7). Returns (classes, omega_hat, L_hat), or
        None when no meter candidate is viable -- which on line 17's product means
        there is no hypothesis to take an argmax over, not that the answer is beats.

        Lines 11-14 are _class_posterior and line 17's numerator is
        _meter_phase_scores: the same two functions the E-step uses, so inference and
        training score a hypothesis identically by construction. Line 17's denominator
        is constant in (omega, L), so line 20's argmax needs only the numerator.
        """
        scores_by_meter = self._meter_phase_scores(self._class_posterior(p))
        if not scores_by_meter:
            return None

        meters = list(scores_by_meter)
        meter_phase_scores = torch.cat([scores_by_meter[meter] for meter in meters])
        best = int(torch.argmax(meter_phase_scores))         # line 20

        # Invert the concatenation: each meter holds one entry per phase, in phase
        # order, so the flat index decomposes into (L_hat, omega_hat) by walking it.
        start = 0
        for meter in meters:
            width = scores_by_meter[meter].shape[0]
            if best < start + width:
                omega_hat, meter_hat = best - start, meter
                break
            start += width

        # Line 23: c_i(omega_hat, L_hat), the same pattern _meter_phase_scores
        # scored, so the emitted labels are exactly what won the argmax.
        i0 = torch.arange(p.shape[0], device=p.device)
        is_downbeat = ((omega_hat + i0) % meter_hat) == 0
        classes = torch.where(is_downbeat,
                              torch.full_like(i0, DOWNBEAT),
                              torch.full_like(i0, BEAT))
        return classes, int(omega_hat), int(meter_hat)


    def _meter_phase_prior(self, meter):
        """pi_M(L) pi_omega(omega), line 39's first two factors, stored combined.
        pi_omega is uniform over the L phases, so the pair is the same under every omega."""
        if self.meter_prior is None:
            return 1.0 / meter
        return self.meter_prior.get(meter, 0.0)

    def _log_meter_phase_prior(self, meter):
        """The same pair in line 52's own logs. A meter the corpus never shows scores
        -inf, which is what it deserves.
        """
        prior = self._meter_phase_prior(meter)
        return math.log(prior) if prior > 0.0 else -float('inf')

    def _meter_phase_scores(self, matched_class_posterior):
        """Algorithm 1 line 39: pi_M(L) pi_omega(omega) prod_i q_i(c_i(omega, L)).
        One entry per (omega, L), grouped by meter, in the order line 40 sums over.
        Rejected pairs reach 1e-360, so the product stays in the posterior's float64."""
        q_db, q_b = matched_class_posterior
        M = q_db.shape[0]
        events = torch.arange(M, device=q_db.device)
        scores_by_meter = {}

        for meter in self.meter_candidates:
            meter = int(meter)
            # A meter of 1 makes every event a downbeat, so it carries no phase to
            # infer. Requiring M >= L is a choice about short fragments rather than a
            # well-definedness guard: c_i(omega, L) is defined for any L, but meters
            # larger than the event count generate indistinguishable patterns, so they
            # would only spread the prior's mass. It fires only on degenerate crops.
            if meter <= 1 or M < meter:
                continue

            # pi_M(L) pi_omega(omega), the same under every phase of a given meter.
            prior = self._meter_phase_prior(meter)

            phases = torch.arange(meter, device=q_db.device)
            # is_db[p, i]: event i is a downbeat under phi_0 = p, i.e. (p + i) % L == 0.
            is_db = ((phases[:, None] + events[None, :]) % meter) == 0

            # q_i(c_i(omega, L)) for every event, then the product over events.
            factors = torch.where(is_db, q_db[None, :], q_b[None, :])
            scores_by_meter[meter] = prior * factors.prod(dim=1)

        return scores_by_meter or None

    def _log_meter_phase_scores(self, matched_class_posterior):
        """Algorithm 2 line 52's bracket, in the logs that line writes itself:
        log pi_M(L) + log pi_omega + sum_i log P_hat(C_i = c_i(omega, L)). Agreeing with
        _meter_phase_scores up to exp is what makes Fisher's identity hold across E/M."""
        log_db, log_b = matched_class_posterior
        M = log_db.shape[0]
        device = log_db.device

        i0 = torch.arange(M, device=device)
        scores_by_meter = {}

        for meter in self.meter_candidates:
            meter = int(meter)
            # A meter of 1 makes every event a downbeat, so it carries no phase to
            # infer. Requiring M >= L is a choice about short fragments rather than a
            # well-definedness guard: c_i(omega, L) is defined for any L, but meters
            # larger than the event count generate indistinguishable patterns, so they
            # would only spread the prior's mass. It fires only on degenerate crops.
            if meter <= 1 or M < meter:
                continue

            log_prior = self._log_meter_phase_prior(meter)

            phases = torch.arange(meter, device=device)
            # is_db[p, i]: event i is a downbeat under phi_0 = p, i.e. (p + i) % L == 0.
            is_db = (((phases[:, None] + i0[None, :]) % meter) == 0).to(log_db.dtype)

            scores_by_meter[meter] = (is_db @ log_db + (1.0 - is_db) @ log_b
                                      + log_prior)

        if not scores_by_meter:
            return None

        return scores_by_meter


    def _meter_log_posterior(self, match_likelihood):
        """log P(L | x) for every viable L, from raw matched class log-probabilities.

        Convenience for the diagnostics, which start from a checkpoint's logits rather
        than from an E-step that already built the posterior.
        """
        scores_by_meter = self._log_meter_phase_scores(
            self._class_log_posterior(match_likelihood))
        if scores_by_meter is None:
            return None
        meters = list(scores_by_meter)
        log_joint = torch.log_softmax(
            torch.cat([scores_by_meter[meter] for meter in meters]), dim=0)
        out, start = {}, 0
        for meter in meters:
            out[meter] = log_joint[start:start + meter].logsumexp(dim=0)
            start += meter
        return out
