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
        return {'base': self.base_optimizer.state_dict(), 'k': self.k,
                'alpha': self.alpha, 'step': self._step}

    def load_state_dict(self, state_dict):
        if 'base' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base'])
            self.k = state_dict.get('k', self.k)
            self.alpha = state_dict.get('alpha', self.alpha)
            self._step = state_dict.get('step', 0)
        else:
            # a plain Adam/RAdam state_dict from an older run
            self.base_optimizer.load_state_dict(state_dict)
        # slow weights are not checkpointed; re-seed them from the restored fast ones
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow[p]['slow_param'] = p.detach().clone()

    @torch.no_grad()
    def step(self, closure=None):
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
