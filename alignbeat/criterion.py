"""The training loss (equation 8) and its EM dispatch (Algorithms 3-9)."""
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.classes import (BACKGROUND, BEAT, CLASS_UNKNOWN, DOWNBEAT,
                               F_MEASURE_TOLERANCE, METER_PRIOR)
from alignbeat.dp import (event_is_downbeat_under, phase_class_nll, phase_star,
                          subset_select_dp, subset_select_dp_joint_phase,
                          subset_select_dp_meter)


# Nothing -- no CLI flag, no test -- ever sets these, so they are constants.
OMEGA_BEAT = 1.0              # eq. (8); only omega_DB is swept
FRAGMENT_SECONDS = 30.0       # 1500 frames at 50 fps; diagnostic display only.
                              # Also the unit t_hat lives in: eps below is this many
                              # seconds' worth of the (0, 1] window.

# Defaults for the arguments below, which only the tests vary.
B_MIN = 1e-4
NORMALIZE_BY_EVENTS = True
DIAGNOSTIC_EVERY = 200
BEAT_ONLY_WARMUP = 2000
BEAT_ONLY_CONFIDENCE = 0.7
# The tolerance in the units t_hat lives in: a fraction of the window.
EPS = F_MEASURE_TOLERANCE / FRAGMENT_SECONDS

PRECISION_PRIOR_ALPHA = 2.0
PRECISION_PRIOR_BETA = None


class Match(NamedTuple):
    """What the E-step decided for one fragment."""
    sigma: object          # (M,) numpy int array, the chosen candidates
    phi: object            # phase of the first matched event, or None
    meter: int             # L in force for this fragment, 0 if none


class SubsetCriterion(nn.Module):
    """Per-pair cost (3), the selection DP, and the training loss (8)."""

    def __init__(self,
                 omega_downbeat=2.0, gamma=0.5,
                 normalize_by_events=NORMALIZE_BY_EVENTS,
                 background_by_unmatched=False,
                 b_min=B_MIN,
                 precision_prior_alpha=PRECISION_PRIOR_ALPHA,
                 precision_prior_beta=PRECISION_PRIOR_BETA,
                 meter_length=0, meter_candidates=(), meter_prior=None,
                 joint_phase=False, mu_meter=0.0,
                 beat_only_warmup=BEAT_ONLY_WARMUP,
                 beat_only_confidence=BEAT_ONLY_CONFIDENCE):
        super(SubsetCriterion, self).__init__()
        self.omega_downbeat = omega_downbeat
        self.gamma = gamma
        self.normalize_by_events = normalize_by_events
        self.background_by_unmatched = background_by_unmatched

        self.b_min = b_min
        self.precision_prior_alpha = precision_prior_alpha
        self.precision_prior_beta = precision_prior_beta

        self.meter_length = meter_length
        self.meter_candidates = tuple(meter_candidates)
        # log P(L, phi_0) = log P(L) - log L: the hypotheses are (L, phi_0) pairs and
        # phase is uniform within a meter, so without the -log L a meter of L gets L
        # times its share and the posterior drifts toward large meters.
        self.meter_prior = None
        if meter_prior is not None:
            table = METER_PRIOR if meter_prior == "corpus" else dict(meter_prior)
            keys = tuple(meter_candidates) or tuple(table)
            total = sum(table.get(L, 0.0) for L in keys)
            if total <= 0.0:
                raise ValueError(f"meter_prior has no mass on candidates {keys}")
            self.meter_prior = {int(L): math.log(table.get(L, 0.0) / total) - math.log(L)
                                for L in keys if table.get(L, 0.0) > 0.0}
        self.joint_phase = joint_phase
        self.mu_meter = mu_meter

        self.beat_only_warmup = beat_only_warmup
        self.beat_only_confidence = beat_only_confidence
        self._call_count = 0

        print(f"[subset-criterion] omega_db={self.omega_downbeat} gamma={self.gamma} "
              f"meter_L={self.meter_length} "
              f"meter_candidates={self.meter_candidates or 'off'} "
              f"joint_phase={self.joint_phase} mu_meter={self.mu_meter} "
              f"beat_only_warmup={self.beat_only_warmup} "
              f"normalize_by_events={self.normalize_by_events}", flush=True)


    def eps_l1(self, t_hat, t_target, laplace_scale, eps=None):
        """eps-insensitive L1: exactly zero within eps of the annotation."""
        # eps is a tolerance in seconds; t_hat is a fraction of the window, so it has
        # to cross into the same unit as the residual it is subtracted from.
        if eps is None:
            eps = EPS
        return (t_hat - t_target).abs().sub(eps).clamp(min=0.0) / laplace_scale

    def build_cost(self, log_probabilities, t_hat, laplace_scale, event_classes,
                   event_times):
        """Per-pair cost (3) plus the section 8.4 background correction."""
        class_cost = self.class_nll(log_probabilities, event_classes) # (M, N)
        time_cost = self.eps_l1(t_hat[None, :], event_times[:, None], laplace_scale)

        background_nll = -log_probabilities[:, BACKGROUND]                                # (N,)

        l_match = class_cost + time_cost                                        # eq. (3)
        return l_match - self.gamma * background_nll[None, :]                   # section 8.4

    def build_phase_cost(self, log_probabilities, t_hat, laplace_scale, event_classes,
                         event_times, meter, p):
        """Equation (18), and its section 8.5 background-corrected form L'^p_match."""
        M = event_times.shape[0]
        phase_cost = phase_class_nll(log_probabilities, M, meter, p)          # (M, N)
        observed_cost = self.class_nll(log_probabilities, event_classes)      # (M, N)
        empty = (event_classes == CLASS_UNKNOWN)[:, None]
        class_cost = torch.where(empty, phase_cost, observed_cost)

        # Same time channel as build_cost, so the phase hypotheses are ranked on the
        # same quantity the phase-blind path uses.
        time_cost = self.eps_l1(t_hat[None, :], event_times[:, None], laplace_scale)
        background_nll = -log_probabilities[:, BACKGROUND]                    # (N,)

        l_match_p = class_cost + time_cost                                    # eq. (18)
        return l_match_p - self.gamma * background_nll[None, :]               # section 8.5

    def _fragment_meter(self, event_classes):
        """The meter L in force for this fragment, or 0 if none is available."""
        if self.meter_length > 0:
            return int(self.meter_length)
        positions = (event_classes == DOWNBEAT).nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            return 0
        return int(np.median(np.diff(positions.cpu().numpy())))

    @staticmethod
    def class_nll(log_probabilities, event_classes):
        """-log p_j(c_i) for every (event, candidate) pair, handling unlabelled classes."""
        empty = event_classes == CLASS_UNKNOWN
        safe = torch.where(empty, torch.zeros_like(event_classes), event_classes)
        cost = -log_probabilities[:, safe].transpose(0, 1)                      # (M, N)
        if bool(empty.any()):
            active = torch.logsumexp(
                log_probabilities[:, [DOWNBEAT, BEAT]], dim=-1)                 # (N,)
            cost = torch.where(empty[:, None], (-active)[None, :], cost)
        return cost

    def _e_step(self, log_probabilities_b, t_hat_b, laplace_scale, event_classes, event_times):
        """Algorithm 3 lines 1-9: the MAP estimate of sigma under the current theta."""
        with torch.no_grad():
            corrected = self.build_cost(log_probabilities_b, t_hat_b,
                                        laplace_scale, event_classes, event_times)
            fragment_meter = self._fragment_meter(event_classes)

            if not bool(torch.isfinite(corrected).all()):
                # log p is floored at LOG_PROB_FLOOR and the time term is bounded, so a
                # non-finite cost means the model emitted NaN/Inf. Raise rather than
                # skip: a surviving batch would mask its own cause.
                raise FloatingPointError(
                    f"non-finite matching cost, M={event_classes.numel()}: the model "
                    f"produced NaN/Inf class logits or t_hat")

            if self.joint_phase and fragment_meter > 1:
                # JA: Section 8.3 eq. (19): rerun the DP once per phase hypothesis, keep the
                # cheapest, so sigma is chosen WITH phase evidence rather than before it.
                phase_costs = torch.stack([
                    self.build_phase_cost(log_probabilities_b, t_hat_b, laplace_scale,
                                          event_classes, event_times, fragment_meter, p)
                    for p in range(fragment_meter)])

                sigma_np, phi_hat = subset_select_dp_joint_phase(phase_costs.cpu().numpy())

                return Match(sigma_np, phi_hat, fragment_meter)

            cost_np = corrected.cpu().numpy()
            if self.mu_meter > 0.0 and float(fragment_meter) >= 1.0:
                downbeats = (event_classes == DOWNBEAT).nonzero(
                    as_tuple=False).flatten().cpu().numpy()

                sigma_np = subset_select_dp_meter(
                    cost_np, downbeats, t_hat_b.cpu().numpy(), float(fragment_meter),
                    self.mu_meter)
            else:
                sigma_np = subset_select_dp(cost_np)

            return Match(sigma_np, None, fragment_meter)

    def _class_term(self, matched_log, event_classes, target, match):
        """Loss (8)'s first bracket: -log p_sigma(i)(c_i), or section 8's marginal for B*."""
        unknown = event_classes == CLASS_UNKNOWN

        # JA: safe is event_classes with every CLASS_UNKNOWN (-1) replaced by 0, purely
        # so gather gets a legal index.
        safe = torch.where(unknown, torch.zeros_like(event_classes), event_classes)

        # JA: Here 'event' in `event_classes` means beat events. `per_event_nll` is the
        # class negative log-likelihood for each ground-truth event, shape (M,)
        per_event_nll = -matched_log.gather(1, safe[:, None]).squeeze(1)

        if bool(unknown.any()):
            # JA: This _beat_only_term function corresponds to Algorithm 5 (hard case)
            beat_only, _ = self._beat_only_term(
                matched_log, target.get('segments'), phi_hat=match.phi, meter=match.meter)
            return torch.where(unknown, beat_only, per_event_nll), int(unknown.sum())
        else:
            return per_event_nll, 0

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
        """Algorithm 3 lines 10-15: sigma held fixed, loss (8) built from p and t.
        
        Both the sequential and joint E-steps use the same M-step."""
        event_classes, event_times = target['classes'], target['times']

        device = log_probabilities.device
        num_candidates = log_probabilities.shape[0]
        background_nll = -log_probabilities[:, BACKGROUND]

        sigma = torch.from_numpy(match.sigma).to(device)

        M = int(event_classes.numel())
        denominator = float(M) if self.normalize_by_events else 1.0

        omega = torch.where(
            event_classes == DOWNBEAT,
            torch.full_like(event_times, self.omega_downbeat),
            torch.full_like(event_times, OMEGA_BEAT))

        # line 13, first bracket
        per_event, unlabelled = self._class_term(
            log_probabilities[sigma], event_classes, target, match)

        # line 13, second bracket. Two distinct quantities: the loss uses the
        # eps-insensitive residual, but eq. (5)'s b and the ms diagnostic want the true
        # error -- clamping those would drive b toward b_min and report 0 ms while the
        # model is still tens of ms out.
        matched_residual = (event_times - t_hat[sigma]).abs()            # (M,), raw
        residual = matched_residual.sub(EPS).clamp(min=0.0)              # eps-insensitive
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

        return {
            'class': (omega * per_event).sum() / denominator,
            'time': time_term,
            'background': background_nll[unmatched].sum() / background_denominator,
            'precision': precision_term,
            'residual': matched_residual.detach(),
            'unlabelled': unlabelled,
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

        for i in range(batch_size):
            event_classes = targets[i]['classes']
            event_times = targets[i]['times']
            M = int(event_classes.numel())

            background_nll = -log_probabilities[i, :, BACKGROUND]

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
            # laplace_scale is (B, N); each step sees this fragment's own candidates.
            match = self._e_step(log_probabilities[i], t_hat[i], laplace_scale[i],
                                 event_classes, event_times)

            # M-step: sigma fixed and the loss evaluated at it (Alg. 3, 10-15).
            terms = self._m_step(match, log_probabilities[i], t_hat[i], targets[i], laplace_scale[i])

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
        losses['total'] = sum(losses[k] for k in
                              ('class', 'time', 'background'))
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
        print(f"[subset] b_hat={stats['b_hat_mean']:.5f} "
              f"[{stats['b_hat_min']:.5f}, {stats['b_hat_max']:.5f}] "
              f"({stats['b_hat_mean'] * FRAGMENT_SECONDS * 1000:.0f}ms) | "
              f"residual={stats.get('residual_mean', float('nan')):.5f} "
              f"({stats.get('residual_mean', 0.0) * FRAGMENT_SECONDS * 1000:.0f}ms) "
              f"min_gap={stats['min_gap']:.2e} events={stats['num_events']} "
              f"infeasible={stats['infeasible']}", flush=True)

    def _beat_only_term(self, matched_log, segments=None, phi_hat=None, meter=0):
        """Section 8 class term for events whose B/DB label was never observed."""
        active = torch.logsumexp(matched_log[:, [DOWNBEAT, BEAT]], dim=-1)     # log(1-p(empty))
        marginal = -active
        if self._call_count < self.beat_only_warmup:
            return marginal, None

        if phi_hat is not None and meter > 1:
            with torch.no_grad():
                r = event_is_downbeat_under(
                    matched_log.shape[0], int(meter), int(phi_hat), matched_log.device
                ).to(matched_log.dtype)
            weighted = -(r * matched_log[:, DOWNBEAT] + (1.0 - r) * matched_log[:, BEAT])
            return weighted, r

        posterior = self._phase_posterior_marginal(matched_log, segments)
        if posterior is None:
            return marginal, None
        r, valid = posterior

        with torch.no_grad():
            confident = valid & (torch.maximum(r, 1.0 - r) >= self.beat_only_confidence)
        weighted = -(r * matched_log[:, DOWNBEAT] + (1.0 - r) * matched_log[:, BEAT])
        return torch.where(confident, weighted, marginal), torch.where(
            confident, r, torch.zeros_like(r))

    def _span_phase_posterior(self, span, candidates, phases=None):
        """Eqs. (12)/(14), generalised to eqs. (32)/(34) when several meters are given.

        r_i = sum over hypotheses (L, p) of P(L, phi_0 = p | y, x; theta) for those under
        which event i is a downbeat. With a single candidate exactly one p qualifies per
        event, so this collapses to eq. (14)'s pi[p*_i] lookup.

        phases restricts which phi_0 are considered. The window crop is arbitrary, so the
        first span may begin anywhere in the bar and every p is open; a later span begins
        where a time signature changes, which is a bar line, so phases=(0,) there.
        """
        M = span.shape[0]
        device = span.device
        hypotheses, log_pi = [], []
        for meter in candidates:
            meter = int(meter)
            if meter <= 1 or M < meter:
                continue
            allowed = range(meter) if phases is None else [
                p for p in phases if 0 <= p < meter]
            # Uniform over (L, phi_0) when no prior is given: -log L, not 0.
            log_prior = (-math.log(meter) if self.meter_prior is None
                         else self.meter_prior.get(meter, -float('inf')))
            for p in allowed:
                score = torch.where(event_is_downbeat_under(M, meter, p, device),
                                    span[:, DOWNBEAT], span[:, BEAT]).sum() + log_prior
                hypotheses.append((meter, p))
                log_pi.append(score)

        if not hypotheses:
            return None

        posterior = torch.softmax(torch.stack(log_pi), dim=0)          # P(L, phi_0 | x)
        r = torch.zeros(M, device=device, dtype=span.dtype)
        for (meter, p), weight in zip(hypotheses, posterior):
            r += weight * event_is_downbeat_under(M, meter, p, device).to(r.dtype)
        return r

    def _phase_posterior_marginal(self, matched_log, segments=None):
        """r_i = P(phi_i = 0 | y, x; theta): one meter over the window, or eq. (17) per segment."""
        M = matched_log.shape[0]

        with torch.no_grad():
            # JA: No meter change for this fragment; treat the whole window as one meter.
            # It is not an assertion that the music has constant meter; it's the absence of
            # any claim about meter changes. The code then does the only reasonable thing:
            # one span covering all M events, one φ₀, with the meter taken from
            # meter_candidates (latent) or meter_length (fixed).
            if segments is None:
                # A fixed meter_length is just a one-element candidate set; the meter is
                # latent when several candidates are offered.
                candidates = self.meter_candidates or (int(self.meter_length),)
                r = self._span_phase_posterior(matched_log, candidates)
                if r is None:
                    return None
                return r, torch.ones(M, dtype=torch.bool, device=matched_log.device)

            r = torch.zeros(M, device=matched_log.device, dtype=matched_log.dtype)
            valid = torch.zeros(M, device=matched_log.device, dtype=torch.bool)
            for k, (start, meter) in enumerate(segments):
                end = segments[k + 1][0] if k + 1 < len(segments) else M
                # Each segment declares its own meter, so no marginalisation there. Only
                # the first span's phase is free: the window crop lands anywhere in a bar,
                # whereas a later span starts where the meter changes, i.e. on a bar line.
                span_r = self._span_phase_posterior(matched_log[start:end], (meter,),
                                                    phases=None if k == 0 else (0,))
                if span_r is None:
                    continue
                r[start:end] = span_r
                valid[start:end] = True
        return r, valid

    def _per_candidate_time_term(self, residual, b_j, eps=EPS):
        """-log p(r | b_j) for the uniform-core / Laplace-tail density, split per 4.1.3.

        residual is already eps-insensitive, so log(2 eps + 2 b_j) is the normaliser that
        makes this a likelihood in b_j: without it b_j -> inf minimises the loss and the
        timing channel switches itself off.
        """
        localisation = residual / b_j.detach()                       # gradient to t_hat
        precision = residual.detach() / b_j + torch.log(2.0 * eps + 2.0 * b_j)  # to b_j
        return localisation + precision

    def _precision_prior(self, b_j):
        """Mitigation two: a Gamma prior on the precision 1/b_j, as a MAP term."""
        alpha = self.precision_prior_alpha
        beta = (self.precision_prior_beta if self.precision_prior_beta is not None
                else float(F_MEASURE_TOLERANCE / FRAGMENT_SECONDS) * max(alpha - 1.0, 1e-6))
        return ((alpha - 1.0) * torch.log(b_j) + beta / b_j).sum()

