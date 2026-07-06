"""Tests for plr-minimum-effective (plant-and-recover, no hardware)."""

import math
import random
import statistics as st

from min_effective.surface import LibrarySurface, to_real, to_norm, KNOBS
from min_effective.agent import URPDAgent
from min_effective.prior import default_expert_prior
from min_effective.loop import DiscoveryConfig, DiscoveryLoop


def _cfg(**kw):
    return DiscoveryConfig(candidate_n=300, true_grid=16, **kw)


def _agg(strategy, exclude=True, n=12):
    feas, gaps, max_runs = 0, [], 0
    for seed in range(n):
        d = DiscoveryLoop(_cfg(seed=seed, strategy=strategy, exclude_faults=exclude)).run()
        max_runs = max(max_runs, d["runs"])
        if d["found"] and d["feasible"]:
            feas += 1
            gaps.append(d["urpd_gap"])
    return feas, gaps, max_runs


# -- surface / plant ---------------------------------------------------------
def test_optimum_is_feasible_and_interior_in_cycles():
    s = LibrarySurface(seed=0)
    x, u = s.true_optimum(grid=18)
    assert s.feasible(x) and u > 0
    lo = (0.0, x[1], x[2])   # too few cycles
    hi = (1.0, x[1], x[2])   # too many cycles (duplicates)
    assert s.urpd_true(lo) < u and s.urpd_true(hi) < u


def test_unique_collapses_below_reagent_cliff():
    s = LibrarySurface(seed=0)
    assert s.unique_true((0.3, 0.7, 0.0)) < 0.05     # reagent 0.25 is below the miniaturization floor


def test_to_norm_inverts_to_real():
    x = (0.3, 0.6, 0.5)
    xr = to_norm(to_real(x))
    assert abs(xr[1] - 0.6) < 0.02                    # input (log) recovered up to rounding
    assert abs(xr[2] - 0.5) < 1e-6                    # reagent (linear) exact
    assert abs(sum(k.cost_weight for k in KNOBS) - 1.0) < 1e-9


def test_evaluate_fault_rate_is_configured():
    s = LibrarySurface(seed=0, fault_rate=0.2)
    rng = random.Random(1)
    faults = sum(s.evaluate((0.3, 0.7, 0.6), rng).fault for _ in range(1000))
    assert 0.14 < faults / 1000 < 0.27


# -- teachable prior ---------------------------------------------------------
def test_prior_pseudo_observations_in_range():
    obs = default_expert_prior().pseudo_observations()
    assert len(obs) == 5
    for x, u, w in obs:
        assert all(0.0 <= v <= 1.0 for v in x) and w == 0.5 and u >= 0.0


def test_agent_surrogate_moves_toward_observation():
    a = URPDAgent(3, lambda x: 1.0, unique_target=0.5, candidate_n=10, seed=0)
    a.observe((0.5, 0.5, 0.5), 0.9)
    mean, n_eff = a._predict((0.5, 0.5, 0.5))
    assert abs(mean - 0.9) < 1e-9 and n_eff > 0


# -- recovery ----------------------------------------------------------------
def test_agent_beats_random_in_narrow_feasible_space():
    pf, _, max_runs = _agg("bo_prior")
    cf, _, _ = _agg("bo_cold")
    rf, _, _ = _agg("random")
    assert pf >= rf + 4 and cf >= rf + 3              # only ~5% feasible: random mostly misses it
    assert max_runs <= DiscoveryConfig().budget + 2


def test_teaching_lands_closer_to_the_optimum():
    _, pg, _ = _agg("bo_prior")
    _, cg, _ = _agg("bo_cold")
    assert st.median(pg) <= st.median(cg)             # taught prior lands nearer the max-URPD recipe
    assert st.median(pg) < 0.20


def test_cv_cleaning_keeps_the_search_alive():
    hf, _, _ = _agg("bo_prior", exclude=True)
    nf, _, _ = _agg("bo_prior", exclude=False)
    assert hf >= nf + 5                               # excluding flagged faults recovers feasibly far more


def test_determinism():
    a = DiscoveryLoop(_cfg(seed=5)).run()
    b = DiscoveryLoop(_cfg(seed=5)).run()
    assert a == b
