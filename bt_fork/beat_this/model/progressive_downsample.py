"""
T -> N progressive downsampling (dp-short-1.pdf Section 2, eq. 1-3).

Without an FPN, takes the single (B,T,d_model) output of the encoder's last
layer (BeatThisEncoder) and progressively shrinks its length via
patch-merging into (B,N,d_model). N is not a fixed hyperparameter but a value
derived from N_min and the reduction ratio r (eq.1).
"""
import math

import torch
from torch import nn


def compute_schedule(T: int, N_min: int, r: float = 0.5):
    """eq.(1): T_0:=T, T_s:=ceil(r*T_{s-1}), S:=max{s : T_s >= N_min}, N:=T_S.

    Returns the list [T_1, T_2, ..., T_S] (T_0=T itself is not included,
    since that's just the input length).

    - Section 2 "Architecture" -

    compute_schedule implements the paper's eq.(1). eq.(1) is the formula
    that precomputes "how many steps (S) it takes to shrink T down to N, and
    what the final length N ends up being." At each step the length is
    roughly halved (r=1/2), but it keeps repeating only as long as it can
    stay at or above the floor N_min; the length at each step is returned as
    a list.

    In other words, if you keep halving T, there's no guarantee the result
    ever lands exactly on some arbitrarily chosen N. Instead, only N_min (the
    minimum number of candidates that must exist -- BPM_max x D_min = 100) is
    fixed, and the strategy is: "halve as many times as possible while
    staying at or above N_min, then take whatever length remains as N."

    The reason this shrinks over several steps instead of all at once: if
    1500 were compressed straight down to 188 in one step, each step would
    have to summarize an average of 8 (1500/188) positions at once, causing
    heavy information loss. Halving exactly in half each time keeps the
    amount of information any single step has to handle much smaller, at the
    cost of needing several steps to progressively summarize.
    """
    schedule = []
    T_s = T
    while True:
        next_T = math.ceil(r * T_s) # candidate length for this step
        if next_T < N_min: # if it would drop below N_min
            break # don't include this step, stop here
        T_s = next_T # confirm the candidate length
        schedule.append(T_s) # record it
    return schedule 


class DownsampleStep(nn.Module):
    """eq.(3): merges two adjacent positions into one via concat + linear
    projection. If the input length is odd, the last leftover position is
    handled on its own by a separate learnable linear projection (W'_s).

    Adjacent pairs within the step's input sequence (1&2, 3&4, ...) are
    concatenated, then a learnable linear projection W_s merges each pair
    into one. E.g. for a 750-length input: (g_1, g_2) -> g_1', (g_3, g_4) ->
    g_2', repeated 375 times to shrink to 375. Each step (1500 -> 750,
    750 -> 375, 375 -> 188) has its own independent W_s, b_s -- i.e. the
    three steps are separate DownsampleStep instances with separate
    parameters.

    Odd-length handling only triggers when the step's input length (T_{s-1})
    is odd. E.g. in the 375 -> 188 step, since 375 is odd, 187 pairs (374
    elements) go through the normal concat+W_s path, and the single leftover
    375th element goes through its own W'_s. The result is 187 + 1 = 188
    (even).

    This handling makes the resulting length exactly match eq.(1)'s schedule.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.merge = nn.Linear(2 * d_model, d_model)  # W_s, b_s: learnable params that merge a pair into one
        self.odd_proj = nn.Linear(d_model, d_model)   # W'_s, b'_s (only used when odd): separate params for the odd leftover

    def forward(self, x):
        # x: (B, T_prev, d_model)
        B, T_prev, D = x.shape
        n_pairs = T_prev // 2

        paired = x[:, : 2 * n_pairs, :].reshape(B, n_pairs, 2 * D)  # [g_2i-1 ; g_2i] concat
        merged = self.merge(paired)  # (B, n_pairs, D)

        if T_prev % 2 == 1: # if odd, the last element has no partner
            last = self.odd_proj(x[:, -1:, :])  # (B, 1, D)
            merged = torch.cat([merged, last], dim=1)  # (B, n_pairs+1, D), appended at the end

        return merged  # (B, ceil(T_prev/2), D)


class ProgressiveDownsample(nn.Module):
    """encoder output (B,T,d_model) -> candidate feature (B,N,d_model).

    N is precomputed and fixed at __init__ time via compute_schedule(T, N_min, r),
    and one DownsampleStep with its own parameters is created per step
    T_1..T_S (eq.3: "W_s ... learned per step" -- independent weights per step).

    ProgressiveDownsample first precomputes how many steps are needed via
    compute_schedule (eq.1), builds that many independent DownsampleStep
    instances (eq.3), and runs them in order in forward -- the full pipeline
    that shrinks T down to N.
    """

    def __init__(self, d_model: int, T: int, N_min: int, r: float = 0.5):
        super().__init__()
        self.schedule = compute_schedule(T, N_min, r) # precompute the schedule via eq.(1) -> [750, 375, 188]
        if not self.schedule:
            # if T is already below N_min there are no steps to take -- error
            raise ValueError(f"T={T} is already below N_min={N_min}; no downsampling steps possible")
        # the last value of the schedule is the final N
        self.N = self.schedule[-1]
        # create one independent DownsampleStep per schedule entry (S=3)
        self.steps = nn.ModuleList([DownsampleStep(d_model) for _ in self.schedule])

    def forward(self, x):
        for step in self.steps:
            x = step(x)
        # runs through the DownsampleSteps in order: 1500 -> 750 -> 375 -> 188
        return x  # (B, self.N, d_model)
