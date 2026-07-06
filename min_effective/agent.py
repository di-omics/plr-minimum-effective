"""
Cost-aware agent that maximizes unique reads per dollar (pure Python, no numpy).

The agent does not chase yield or minimize cost alone. It maximizes URPD, unique
reads per dollar, subject to reading enough unique molecules to decide. That is
the objective that makes a recipe worth running: cheaper AND more informative per
dollar, not one at the expense of the other.

Surrogate: a weighted kernel regression of observed unique reads over the runs it
has seen. Prior demonstrations enter as down-weighted pseudo-observations, so what
you teach it up front biases the search before any hardware runs, and real data
takes over as it accumulates. A poor person's Gaussian process that runs anywhere.

Acquisition: among candidates that are optimistically feasible (upper-confidence
unique clears the bar), take the one with the highest upper-confidence URPD. Before
anything looks feasible, push toward the most promising region.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

Vec = Tuple[float, ...]


class URPDAgent:
    def __init__(self, dim: int, cost_fn, *, unique_target: float,
                 bandwidth: float = 0.16, beta: float = 0.6, unique_scale: float = 0.15,
                 candidate_n: int = 500, seed: int = 0):
        self.dim = dim
        self.cost_fn = cost_fn
        self.unique_target = unique_target      # unique reads a recipe must reach to be feasible
        self.bw = bandwidth
        self.beta = beta
        self.unique_scale = unique_scale        # unique-read units per unit of uncertainty
        self.rng = random.Random(seed)
        self.candidates: List[Vec] = [
            tuple(self.rng.random() for _ in range(dim)) for _ in range(candidate_n)
        ]
        self.obs: List[Tuple[Vec, float, float]] = []   # (x, unique reads, weight)

    def observe(self, x: Vec, unique: float, weight: float = 1.0):
        self.obs.append((x, unique, weight))

    def _predict(self, x: Vec) -> Tuple[float, float]:
        """Return (predicted unique reads, effective sample size) at x."""
        if not self.obs:
            return 0.0, 0.0
        num = den = 0.0
        inv = 1.0 / (2.0 * self.bw * self.bw)
        for xi, u, w in self.obs:
            dist2 = sum((a - b) ** 2 for a, b in zip(x, xi))
            kw = w * math.exp(-dist2 * inv)
            num += kw * u
            den += kw
        if den == 0.0:
            return 0.0, 0.0
        return num / den, den

    def _uncertainty(self, n_eff: float) -> float:
        return 1.0 / math.sqrt(n_eff + 1.0)

    def propose(self) -> Vec:
        best_x, best_urpd = None, -1.0
        fallback_x, fallback_v = None, -1.0
        for x in self.candidates:
            mean, n_eff = self._predict(x)
            unc = self._uncertainty(n_eff)
            ucb_unique = mean + self.beta * self.unique_scale * unc
            if ucb_unique > fallback_v:
                fallback_v, fallback_x = ucb_unique, x
            if ucb_unique >= self.unique_target:              # optimistically feasible
                urpd_ucb = ucb_unique / self.cost_fn(x)
                if urpd_ucb > best_urpd:
                    best_urpd, best_x = urpd_ucb, x
        return best_x if best_x is not None else fallback_x

    def recommend(self, min_n_eff: float = 0.8) -> Optional[Vec]:
        """Highest-URPD recipe the surrogate calls feasible with local evidence.
        This is the recipe you would hand to the validation ladder to confirm."""
        best_x, best_urpd = None, -1.0
        for x in self.candidates:
            mean, n_eff = self._predict(x)
            if n_eff >= min_n_eff and mean >= self.unique_target:
                urpd = mean / self.cost_fn(x)
                if urpd > best_urpd:
                    best_urpd, best_x = urpd, x
        return best_x
