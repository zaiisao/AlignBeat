"""The training loss (equation 8) and its EM dispatch (Algorithms 3-9)."""
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignbeat.constants import (CLASS_BACKGROUND, CLASS_BEAT, CLASS_UNKNOWN, CLASS_DOWNBEAT,
                               F_MEASURE_TOLERANCE)
from alignbeat.inference.decode import meter_phase_scores
from alignbeat.training.dp import subset_select_dp

DIAGNOSTIC_EVERY = 200


class Match(NamedTuple):
    """What the E-step decided for one fragment."""
    sigma: object                   # (M,) numpy int array, the chosen candidates
    pi: object = None               # beat-only: pi_{omega,L}, line 40, frozen (line 46)


class SubsetCriterion(nn.Module):
    """Per-pair cost (3), the selection DP, and the training loss (8)."""

    def __init__(self,
                 data_prior, window_seconds, gamma=0.5,
                 meter_prior=None, match_event_cost=False):
        super(SubsetCriterion, self).__init__()

        self.window_seconds = float(window_seconds)

        self.lambda_l1 = 2.0 * self.window_seconds / F_MEASURE_TOLERANCE
        self.gamma = gamma

        self.match_event_cost = match_event_cost
        self.meter_candidates = tuple(sorted(int(L) for L in meter_prior)) if meter_prior else ()
        self.meter_prior = meter_prior

        self._call_count = 0

        data_priors = torch.tensor([data_prior["downbeat"], data_prior["beat"]],
                                    dtype=torch.float32)

        # JA: We are assuming for now that the document's pi_data is the same as pi_C
        # A buffer, not a bare attribute, so .double() and .cuda() reach it: left as a
        # plain tensor it stays float32 on the CPU while the rest of the E-step runs in
        # float64, which put a float32 epsilon into pi and broke Fisher's identity at
        # the 1e-08 level. Not persistent: no checkpoint records it.
        self.register_buffer("class_prior", data_priors.clone(), persistent=False)

        self.register_buffer("data_prior", data_priors / data_priors.sum())

        print(f"[subset-criterion] lambda_L1={self.lambda_l1:g} gamma={self.gamma} "
              f"meter_candidates={self.meter_candidates or 'off'} "
              f"match_event_cost={self.match_event_cost}", flush=True)


    def build_l_match(self, log_probabilities, t_hat, gt_class, gt_time):
        """Algorithm 1 lines 18 and 20: L_match(i, j), returned (M, N) for the DP.
        Line 18 charges the observed class's NLL; line 20 has no class to charge, so an
        unlabelled event pays the timing error alone."""
        labelled = gt_class != CLASS_UNKNOWN

        # The timing term both 18 and 20 carry
        time_loss_matrix = self.lambda_l1 * (t_hat[None, :] - gt_time[:, None]).abs()

        # Line 18, ind = 0: the observed class's own NLL
        labelled_class_loss_matrix = -log_probabilities[:, gt_class].T

        if self.match_event_cost:
            # Line 20, ind = 1: -log(1 - p_j(empty)), the mass the head puts on the event classes.
            unlabelled_class_loss_matrix = -torch.logsumexp(
                log_probabilities[:, [CLASS_DOWNBEAT, CLASS_BEAT]], dim=-1)[None, :]
        else:
            unlabelled_class_loss_matrix = torch.zeros_like(labelled_class_loss_matrix)

        class_loss_matrix = torch.where(labelled[:, None], 
                                        labelled_class_loss_matrix,
                                        unlabelled_class_loss_matrix)

        return class_loss_matrix + time_loss_matrix

    def _e_step(self, class_logits, t_hat, gt_class, gt_time):
        """Algorithm 1 lines 4-40, on one fragment, at theta_old. Takes the logits
        because the algorithm wants the head both ways: lines 18 and 20 write log p_hat,
        lines 7 and 10 write p_hat, and both come straight off them."""
        with torch.no_grad():
            has_class_labels = bool((gt_class != CLASS_UNKNOWN).any())
            class_posterior = F.softmax(class_logits, dim=-1)

            # Lines 15-23: L_match(i, j).
            l_match = self.build_l_match(torch.log(class_posterior),
                                         t_hat, gt_class, gt_time)

            # Line 26: sigma_hat <- SubsetSelectDP(L_match).
            sigma_np = subset_select_dp(l_match.cpu().numpy())
            sigma = torch.from_numpy(sigma_np).to(class_logits.device)
            pi = None

            if not has_class_labels:
                # Line 36: q_i(c) <- P_hat(C = c | x, sigma_hat(i)).
                matched_class_posterior = class_posterior[sigma]

                # Line 39: the numerator, pi_M(L) pi_omega(omega) prod_i q_i(c_i(omega, L)).
                scores_by_meter = meter_phase_scores(
                    matched_class_posterior, self.meter_candidates, self.meter_prior)
                if scores_by_meter:
                    # Line 40: pi_{omega,L} <- that numerator over its own sum across every
                    # (omega, L) pair. This is the E-step's whole output. Everything the M-step needs
                    # is a function of pi and of theta, so nothing else crosses the boundary.
                    flat = torch.cat([scores_by_meter[meter] for meter in scores_by_meter])
                    pi = flat / flat.sum()

            return Match(sigma, pi)
            

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

    def _m_step(self, match, class_logits, t_hat, target):
        """Algorithm 2 lines 48-56: sigma_hat held fixed, the loss built at theta."""
        gt_class, gt_time = target['classes'], target['times']

        log_probabilities = F.log_softmax(class_logits, dim=-1)

        device = log_probabilities.device
        num_candidates = log_probabilities.shape[0]
        background_nll = -log_probabilities[:, CLASS_BACKGROUND]

        sigma = match.sigma

        # lines 49-56, first term: cls(theta; ind)
        class_term, unlabelled = self._class_term(
            log_probabilities[sigma], gt_class, match)

        residual = (gt_time - t_hat[sigma]).abs()            # (M,), raw
        time_term = self.lambda_l1 * residual.sum()

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

            background_nll = -log_probabilities[b, :, CLASS_BACKGROUND]

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
            terms = self._m_step(match, class_logits[b], t_hat[b], targets[b])

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
              f"({stats.get('residual_mean', 0.0) * self.window_seconds * 1000:.0f}ms) "
              f"min_gap={stats['min_gap']:.2e} events={stats['num_events']} "
              f"infeasible={stats['infeasible']}", flush=True)

    def _class_posterior(self, p):
        """Algorithm 1 lines 7 and 10: Bayes over {DB, B}, pi_data divided out first.
        Neither line carries a logarithm, so neither does this; _class_log_posterior is
        the same quantity for line 52. float64 because a rejected class underflows f32."""
        p = p.double()
        class_prior = self.class_prior.double()
        data_class_prior = self.data_prior.double()

        likelihood_db = p[:, CLASS_DOWNBEAT] / data_class_prior[CLASS_DOWNBEAT]
        likelihood_b = p[:, CLASS_BEAT] / data_class_prior[CLASS_BEAT]

        joint_db = class_prior[CLASS_DOWNBEAT] * likelihood_db
        joint_b = class_prior[CLASS_BEAT] * likelihood_b
        total = joint_db + joint_b

        posterior = torch.zeros_like(p)
        posterior[:, CLASS_DOWNBEAT] = joint_db / total
        posterior[:, CLASS_BEAT] = joint_b / total
        return posterior

    def _log_likelihood(self, log_p):
        """Algorithm 1 line 7: l_hat_j(x | c) := p_hat_j(c | x) / pi_data(c)."""
        return (log_p[:, CLASS_DOWNBEAT] - self.data_prior[CLASS_DOWNBEAT].log(),
                log_p[:, CLASS_BEAT] - self.data_prior[CLASS_BEAT].log())

    def _class_log_posterior(self, log_p):
        """Algorithm 1 line 10: Bayes over {DB, B}, in logs."""
        log_l_db, log_l_b = self._log_likelihood(log_p)
        log_db = log_l_db + self.class_prior[CLASS_DOWNBEAT].log()
        log_b = log_l_b + self.class_prior[CLASS_BEAT].log()
        log_norm = torch.logaddexp(log_db, log_b)
        return log_db - log_norm, log_b - log_norm

    def _beat_only_term(self, matched_log, pi):
        """Algorithm 2 line 52's cls(theta; 1): pi frozen at theta_old, dotted into the
        same bracket recomputed at theta.
        """
        # pi is None when no meter hypothesis was viable, which the algorithm does not
        # contemplate; there is then no expectation to take and the term is zero.
        surrogate = torch.zeros((), dtype=matched_log.dtype, device=matched_log.device)
        if pi is not None:
            scores_by_meter = self._log_meter_phase_scores(
                (matched_log[:, CLASS_DOWNBEAT], matched_log[:, CLASS_BEAT]))
            flat = torch.cat([scores_by_meter[meter] for meter in scores_by_meter])
            surrogate = -(pi * flat).sum()
        return surrogate



    def _log_meter_phase_scores(self, matched_class_probs):
        """Algorithm 2 line 52's bracket, in the logs that line writes itself:
        log pi_M(L) + log pi_omega + sum_i log P_hat(C_i = c_i(omega, L)). Agreeing with
        meter_phase_scores up to exp is what makes Fisher's identity hold across E/M."""
        log_db, log_b = matched_class_probs
        M = log_db.shape[0]
        device = log_db.device

        i0 = torch.arange(M, device=device)
        scores_by_meter = {}

        for meter in self.meter_candidates:
            meter = int(meter)
            if meter <= 1:
                continue

            # x * (1/L) rather than x / L: not bit-identical, and the baseline did this
            prior = self.meter_prior.get(meter, 0.0) * (1.0 / meter)
            log_prior = math.log(prior) if prior > 0.0 else -float('inf')

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
