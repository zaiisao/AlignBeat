"""subset_select_logz against brute force: section 4.1's denominator must be exact.

m(theta, x) is a ratio of two things that are easy to get subtly wrong -- the cheapest
matching and the sum over ALL of them -- so the sum is checked against explicit
enumeration of every order-preserving injection on small cases, and m is checked to be
a probability. If logZ were wrong, section 4.2's condition would be measured wrong.
"""
import itertools, sys
import numpy as np
sys.path.insert(0, '/home/sogang/jaehoon/beatFCOS_new')
from alignbeat.dp import subset_select_dp, subset_select_logz

def main():
  rng = np.random.default_rng(0)
  for trial in range(200):
      M = int(rng.integers(1, 5)); N = int(rng.integers(M, 8))
      cost = rng.random((M, N)) * 4
      brute = -np.inf
      for sigma in itertools.combinations(range(N), M):
        brute = np.logaddexp(brute, -sum(cost[i, j] for i, j in enumerate(sigma)))
      got = subset_select_logz(cost)
      assert abs(got - brute) < 1e-9, (M, N, got, brute)
      # and sigma_hat must be the argmin, so m <= 1
      s = subset_select_dp(cost)
      m = np.exp(-sum(cost[i, j] for i, j in enumerate(s)) - got)
      assert 0 < m <= 1 + 1e-9, m


def test_logz_matches_brute_force():
    main()


def test_empty_and_infeasible():
    assert subset_select_logz(np.zeros((0, 5))) == 0.0
    assert subset_select_logz(np.zeros((6, 3))) == float("-inf")
