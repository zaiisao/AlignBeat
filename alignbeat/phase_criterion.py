"""The continuous-phase training loss and its EM dispatch.

The counterpart of SubsetCriterion for the phase arm. Same skeleton -- an E-step that
picks sigma under the current theta with an order-preserving DP, an M-step that
evaluates the loss at that sigma -- and the same order-preserving DP, literally: the
per-candidate skip cost the phase formulation adds is exactly a column subtraction,
which is the trick SubsetCriterion's own background correction already uses. Verified
against the standalone arm's separate DP on 200 random cases.

What differs from SubsetCriterion is the cost's content. There is no class channel and
no background class: an event's cost is timing plus circular phase distance, and a
candidate left unmatched pays L_agree, its disagreement with the phase interpolated
between the ground-truth events that bracket it.

Three guards SubsetCriterion has are deliberately ABSENT here, because the source
formulation does not have them: the eps-insensitive dead zone, the 2 eps in the scale
normaliser, and the Gamma prior on 1/b. See tests/test_phase_arm.py, which pins their
absence and what it costs, rather than leaving it to be rediscovered.
"""
import numpy as np
import torch
import torch.nn as nn

from alignbeat.dp import subset_select_dp

FALLBACK_METER = 4
LAMBDA_PHI = 3.0


def circ_dist(a, b):
    """d_circ(a, b) = min(|a - b| mod 1, 1 - |a - b| mod 1), the unit-circle distance."""
    d = torch.abs(a - b) % 1.0
    return torch.minimum(d, 1.0 - d)


def phases_from_downbeats(n_beats, downbeat_positions, fallback_meter=FALLBACK_METER):
    """phi_i in [0, 1) for every beat of a fully-labelled fragment.

    phi_i = (beats since this bar's downbeat) / (this bar's length), read per BAR rather
    than per track, so a fragment whose meter changes mid-window still gets correct
    phases without the change point being annotated.
    """
    phi = np.zeros(n_beats, dtype=np.float64)
    if len(downbeat_positions) == 0:
        return phi

    db = np.asarray(downbeat_positions, dtype=np.int64)
    for i in range(n_beats):
        # The bar this beat belongs to: the last downbeat at or before it, or, for beats
        # preceding the first downbeat, that first bar extended backwards.
        k = max(int(np.searchsorted(db, i, side="right")) - 1, 0)
        if k + 1 < len(db):
            bar_len = int(db[k + 1] - db[k])
        elif k > 0:
            bar_len = int(db[k] - db[k - 1])        # last bar: reuse the previous one
        else:
            bar_len = fallback_meter                # one downbeat, no spacing to read
        phi[i] = ((i - int(db[k])) % max(bar_len, 1)) / max(bar_len, 1)
    return phi


class PhaseCriterion(nn.Module):
    """Per-pair cost, the selection DP, and the phase arm's training loss."""

    def __init__(self, lambda_phi=LAMBDA_PHI, lambda_r=0.0,
                 beat_only_meter=FALLBACK_METER, normalize_by_events=True):
        super().__init__()
        self.lambda_phi = float(lambda_phi)
        self.lambda_r = float(lambda_r)
        self.beat_only_meter = int(beat_only_meter)
        self.normalize_by_events = normalize_by_events
        # Prior over phi_0 for beat-only fragments; uniform until something estimates it.
        self.register_buffer(
            "pi_L", torch.full((self.beat_only_meter,), 1.0 / self.beat_only_meter))

        print(f"[phase-criterion] lambda_phi={self.lambda_phi} "
              f"lambda_r={self.lambda_r} beat_only_meter={self.beat_only_meter} "
              f"normalize_by_events={self.normalize_by_events}", flush=True)

    # -- cost ---------------------------------------------------------------

    def match_cost(self, t_true, phi_true, t_hat, phi_hat, b_e, phase_blind):
        """L'_match(y_i, y_hat_j) for every pair, as an (M, N) cost."""
        timing = torch.abs(t_true[:, None] - t_hat[None, :]) / b_e
        if phase_blind:
            return timing
        return timing + self.lambda_phi * circ_dist(phi_true[:, None], phi_hat[None, :])

    @staticmethod
    def l_agree(phi_hat, t_true, phi_true, t_hat):
        """Each candidate's disagreement with the phase its own time implies.

        The target is interpolated between the two ground-truth events bracketing the
        candidate, and detached, so the gradient reaches phi_hat alone and never t_hat.
        """
        M = t_true.shape[0]
        idx = torch.searchsorted(t_true, t_hat.detach().contiguous()).clamp(1, max(M - 1, 1))

        t_i, t_i1 = t_true[idx - 1], t_true[idx]
        phi_i, phi_i1 = phi_true[idx - 1], phi_true[idx]

        w = (t_hat.detach() - t_i) / (t_i1 - t_i).clamp_min(1e-8)
        target = (phi_i + w * (phi_i1 - phi_i)) % 1.0
        return circ_dist(phi_hat, target.detach())

    def _hard_phi0(self, sigma, phi_hat, meter):
        """The best bar phase p for a beat-only fragment, and the phases it implies."""
        M = sigma.shape[0]
        matched = phi_hat[sigma]
        device, dtype = matched.device, matched.dtype

        best_cost, best_p = float("inf"), 0
        for p in range(meter):
            implied = torch.tensor([((p + i) % meter) / meter for i in range(M)],
                                   device=device, dtype=dtype)
            cost = (self.lambda_phi * circ_dist(implied, matched).sum()
                    - torch.log(self.pi_L.to(device)[p] + 1e-12))
            if float(cost) < best_cost:
                best_cost, best_p = float(cost), p

        return torch.tensor([((best_p + i) % meter) / meter for i in range(M)],
                            device=device, dtype=dtype)

    # -- EM -----------------------------------------------------------------

    def _e_step(self, t_true, t_hat, phi_hat, b_e, phi_true):
        """The MAP estimate of sigma under the current theta.

        phi_true is None for beat-only fragments, where the phase is unobserved: the DP
        then runs phase-blind on timing alone, and the bar phase is resolved afterwards
        against the sigma it chose. That ordering is what avoids the circularity of
        scoring phase hypotheses with a matching that phase itself decided.
        """
        with torch.no_grad():
            if phi_true is not None:
                agree = self.l_agree(phi_hat, t_true, phi_true, t_hat)
                cost = self.match_cost(t_true, phi_true, t_hat, phi_hat, b_e,
                                       phase_blind=False)
                # The skip cost enters the DP as a column subtraction; see the module
                # docstring. This is the same structure as section 8.4's correction.
                sigma = subset_select_dp((cost - agree[None, :]).cpu().numpy())
                phi_i = phi_true
            else:
                cost = self.match_cost(t_true, None, t_hat, phi_hat, b_e,
                                       phase_blind=True)
                sigma = subset_select_dp(cost.cpu().numpy())
                phi_i = self._hard_phi0(torch.from_numpy(sigma).to(phi_hat.device),
                                        phi_hat, self.beat_only_meter)

        sigma = torch.from_numpy(sigma).to(t_hat.device)

        unmatched = torch.ones(t_hat.shape[0], dtype=torch.bool, device=t_hat.device)
        unmatched[sigma] = False
        # Recomputed with gradient: the E-step's copy was built under no_grad, and this
        # term is the only path by which an UNMATCHED candidate reaches phi_hat.
        agree_unmatched = self.l_agree(phi_hat[unmatched], t_true, phi_i,
                                       t_hat[unmatched])
        return sigma, phi_i, agree_unmatched

    def _m_step(self, sigma, phi_i, t_true, t_hat, phi_hat, b_e, agree_unmatched):
        """sigma held fixed; the loss evaluated at it."""
        matched_t, matched_phi = t_hat[sigma], phi_hat[sigma]
        M = t_true.shape[0]

        terms = {
            "time": (torch.abs(t_true - matched_t) / b_e).sum(),
            "phase": self.lambda_phi * circ_dist(phi_i, matched_phi).sum(),
            # The normaliser that makes the timing term a likelihood in b_e rather than
            # a free discount. NOTE: unlike SubsetCriterion's log(2 eps + 2 b), this has
            # no eps floor, so it is unbounded below as b_e -> 0.
            "scale": M * torch.log(2 * b_e),
            "agree": (agree_unmatched.sum() if agree_unmatched.numel()
                      else t_hat.sum() * 0.0),
        }

        if self.lambda_r > 0.0:
            downbeats = (phi_i == 0.0)
            if int(downbeats.sum()) >= 2:
                db_times = matched_t[downbeats]
                spacings = db_times[1:] - db_times[:-1]
                mean_spacing = (matched_t[-1] - matched_t[0]) / max(M - 1, 1)
                if float(mean_spacing) > 0:
                    expected = (spacings.mean() / mean_spacing) * mean_spacing
                    terms["periodicity"] = self.lambda_r * ((spacings - expected) ** 2).sum()
        return terms

    def forward(self, phi_hat, t_hat, b_e, targets):
        """One EM step per fragment; returns (losses, stats)."""
        batch_size, num_candidates = t_hat.shape

        buckets, num_events, contributing, skipped = {}, 0, 0, 0
        phase_errors = []

        for i, target in enumerate(targets):
            t_true = target["t_true"]
            M = int(t_true.numel())

            # An order-preserving injection needs N >= M, and a fragment with fewer than
            # two events carries no spacing information. Skipping is what
            # SubsetCriterion does too; a surviving fragment would mask its own cause.
            if M < 2 or M > num_candidates:
                skipped += 1
                continue

            sigma, phi_i, agree = self._e_step(
                t_true, t_hat[i], phi_hat[i], b_e[i], target.get("phi_true"))
            terms = self._m_step(sigma, phi_i, t_true, t_hat[i], phi_hat[i],
                                 b_e[i], agree)

            for key, value in terms.items():
                buckets.setdefault(key, []).append(value)
            with torch.no_grad():
                phase_errors.append(circ_dist(phi_i, phi_hat[i][sigma]).mean())

            num_events += M
            contributing += 1

        zero = torch.nan_to_num(t_hat).sum() * 0.0
        n = max(contributing, 1) if self.normalize_by_events else 1
        losses = {k: (torch.stack(v).sum() / n if v else zero)
                  for k, v in buckets.items()}
        for key in ("time", "phase", "scale", "agree"):
            losses.setdefault(key, zero)
        losses["total"] = sum(losses[k] for k in losses if k != "total")

        with torch.no_grad():
            stats = {
                "num_events": num_events, "skipped": skipped, "used": contributing,
                "b_e_mean": float(b_e.mean()),
                # 0.25 is chance for a 4/4 corpus. A trained head must beat the best
                # CONSTANT phase, not merely chance -- see tests/test_phase_arm.py.
                "phase_err": (float(torch.stack(phase_errors).mean())
                              if phase_errors else float("nan")),
                "phi_std": float(phi_hat.std()),
            }
        return losses, stats
