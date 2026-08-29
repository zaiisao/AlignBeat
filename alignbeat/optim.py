"""Lookahead (Zhang et al. 2019), used by Beat Transformer on top of RAdam.

Beat Transformer's recipe is "RAdam [47] plus Lookahead [48], initial lr 1e-3,
reduced by a factor of 5 whenever validation loss is stuck for 2 epochs, floored
at 1e-7". torch ships RAdam but not Lookahead, so it lives here.

Wrapper, not a subclass: it forwards param_groups/state_dict so ReduceLROnPlateau
and the checkpoint-resume path in train.py keep working unchanged.
"""
from collections import defaultdict
import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        if k < 1:
            raise ValueError(f"Lookahead k must be >= 1, got {k}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Lookahead alpha must be in [0, 1], got {alpha}")
        self.base_optimizer = base_optimizer
        # Share the base optimizer's state rather than leaving self.state undefined:
        # torch.optim.Optimizer.__init__ is never called here (this is a wrapper, not a
        # subclass instance with its own param groups), so anything touching .state -
        # including some schedulers and checkpoint helpers - would raise.
        self.state = base_optimizer.state
        self.k = k
        self.alpha = alpha
        self._step = 0
        # slow weights, one copy per parameter
        self.slow = defaultdict(dict)
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow[p]['slow_param'] = p.detach().clone()

    # --- pass-throughs so schedulers / train.py see a normal optimizer ---
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Slow weights ARE optimizer state: dropping them meant every mid-cycle resume
        # silently restarted the slow sequence from the restored fast weights, perturbing
        # the trajectory. Stored by param index so the dict is portable.
        slow = []
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                entry = self.slow.get(p, {}).get('slow_param')
                slow.append(None if entry is None else entry.detach().cpu())
        return {'base': self.base_optimizer.state_dict(), 'k': self.k,
                'alpha': self.alpha, 'step': self._step, 'slow': slow}

    def load_state_dict(self, state_dict):
        if 'base' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base'])
            self.k = state_dict.get('k', self.k)
            self.alpha = state_dict.get('alpha', self.alpha)
            self._step = state_dict.get('step', 0)
        else:
            # a plain Adam/RAdam state_dict from an older run
            self.base_optimizer.load_state_dict(state_dict)
        stored = state_dict.get('slow') if isinstance(state_dict, dict) else None
        i = 0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if stored is not None and i < len(stored) and stored[i] is not None:
                    self.slow[p]['slow_param'] = stored[i].to(p.device).to(p.dtype)
                else:
                    self.slow[p]['slow_param'] = p.detach().clone()
                i += 1

    def _ensure_slow(self):
        """Seed slow weights for any parameter added after construction. Without this,
        add_param_group (inherited from Optimizer) leaves self.slow[p] empty and the
        k-th step raises KeyError."""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if 'slow_param' not in self.slow[p]:
                    self.slow[p]['slow_param'] = p.detach().clone()

    @torch.no_grad()
    def sync_to_slow(self):
        """Copy slow weights into the live parameters and return the fast weights.

        Lookahead's reported model is the SLOW sequence, but training, validation and
        checkpointing all run on the fast weights; with 313 iterations per epoch and
        k=5 an epoch boundary essentially never lands on a sync, so every score and
        checkpoint we produced was the fast model, not the Lookahead one. Call this
        before evaluating or saving, then restore_fast() to continue training.
        """
        self._ensure_slow()
        cache = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                cache[p] = p.detach().clone()
                p.detach().copy_(self.slow[p]['slow_param'])
        return cache

    @torch.no_grad()
    def restore_fast(self, cache):
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p in cache:
                    p.detach().copy_(cache[p])

    @torch.no_grad()
    def step(self, closure=None):
        self._ensure_slow()
        loss = self.base_optimizer.step(closure)
        self._step += 1
        if self._step % self.k != 0:
            return loss
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                slow = self.slow[p]['slow_param']
                slow.add_(p.detach() - slow, alpha=self.alpha)
                p.detach().copy_(slow)
        return loss
