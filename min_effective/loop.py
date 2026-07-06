"""
Closed-loop discovery of the maximum-URPD recipe, plant-and-recover.

One iteration is: the agent proposes a recipe, the recipe is run (here the
LibrarySurface stands in for compile-and-run plus the UMI-deduplicated read
count), the readout updates the surrogate, and the agent proposes again, under a
run budget. At the end we recover the agent's recommended recipe and score its
unique reads per dollar against the known optimum.

Strategies:

    bo_prior   taught agent: seeded with expert demonstrations, then acquisition
    bo_cold    same agent, no prior (cold start)
    random     random proposals (no prior, no acquisition)

bo_prior vs bo_cold shows what teaching buys. bo_cold vs random shows what the
agent buys. A separate honest/naive split shows what CV-cleaning buys: with
exclude_faults False a mechanical fault reads as a bad recipe and drags the search.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional

from .surface import LibrarySurface, DIM, to_real, Vec
from .agent import URPDAgent
from .prior import default_expert_prior


@dataclass
class DiscoveryConfig:
    budget: int = 60
    n_seed: int = 8
    fault_rate: float = 0.10
    surface_target: float = 0.90
    unique_safety: float = 0.01
    bandwidth: float = 0.16
    beta: float = 0.6
    candidate_n: int = 500
    recommend_min_neff: float = 0.7
    true_grid: int = 24
    seed: int = 0
    strategy: str = "bo_prior"      # bo_prior | bo_cold | random
    exclude_faults: bool = True     # honest search; False keeps CV-flagged faults


class DiscoveryLoop:
    def __init__(self, cfg: DiscoveryConfig):
        self.cfg = cfg

    @property
    def _use_prior(self) -> bool:
        return self.cfg.strategy == "bo_prior"

    @property
    def _use_acquisition(self) -> bool:
        return self.cfg.strategy in ("bo_prior", "bo_cold")

    def _eval(self, s: LibrarySurface, agent: URPDAgent, x: Vec, rng: random.Random) -> int:
        obs = s.evaluate(x, rng)
        if not obs.fault:
            agent.observe(x, obs.unique_obs)
            return 1
        if not self.cfg.exclude_faults:              # naive: keep the faulted read
            agent.observe(x, obs.unique_obs)
            return 1
        obs2 = s.evaluate(x, rng)                     # honest: flagged, so re-run once
        if not obs2.fault:
            agent.observe(x, obs2.unique_obs)
        return 2

    def run(self) -> Dict:
        cfg = self.cfg
        s = LibrarySurface(seed=cfg.seed, target=cfg.surface_target, fault_rate=cfg.fault_rate)
        rng = random.Random(cfg.seed + 1)
        agent = URPDAgent(DIM, s.cost, unique_target=s.min_feasible_unique() + cfg.unique_safety,
                          bandwidth=cfg.bandwidth, beta=cfg.beta,
                          candidate_n=cfg.candidate_n, seed=cfg.seed + 2)
        if self._use_prior:
            for x, u, w in default_expert_prior().pseudo_observations():
                agent.observe(x, u, w)

        seed_rng = random.Random(cfg.seed + 3)
        prop_rng = random.Random(cfg.seed + 4)
        runs = 0
        for _ in range(cfg.n_seed):
            if runs >= cfg.budget:
                break
            x = tuple(seed_rng.random() for _ in range(DIM))
            runs += self._eval(s, agent, x, rng)
        while runs < cfg.budget:
            x = agent.propose() if self._use_acquisition else prop_rng.choice(agent.candidates)
            runs += self._eval(s, agent, x, rng)

        rec = agent.recommend(min_n_eff=cfg.recommend_min_neff)
        return self._score(s, rec, runs)

    def _score(self, s: LibrarySurface, rec: Optional[Vec], runs: int) -> Dict:
        true_x, true_urpd = s.true_optimum(grid=self.cfg.true_grid)
        out = {"strategy": self.cfg.strategy, "runs": runs,
               "true_urpd": round(true_urpd, 4), "true_recipe": to_real(true_x)}
        if rec is None:
            out.update({"found": False, "feasible": False, "rec_urpd": None,
                        "urpd_gap": None, "distance": None, "rec_recipe": None})
            return out
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(rec, true_x)))
        out.update({
            "found": True,
            "feasible": s.feasible(rec),
            "rec_urpd": round(s.urpd_true(rec), 4),
            "urpd_gap": round(true_urpd - s.urpd_true(rec), 4),     # URPD left on the table
            "distance": round(dist, 3),
            "rec_recipe": to_real(rec),
        })
        return out


def compare(budget: int = 60, seed: int = 0, fault_rate: float = 0.18) -> Dict[str, Dict]:
    results = {}
    for strat in ("bo_prior", "bo_cold", "random"):
        cfg = DiscoveryConfig(budget=budget, seed=seed, fault_rate=fault_rate, strategy=strat)
        results[strat] = DiscoveryLoop(cfg).run()
    return results
