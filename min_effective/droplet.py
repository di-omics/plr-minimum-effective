"""
Plate-to-droplet readiness bridge: when is a chemistry ready to miniaturize?

Droplet microfluidics is where "minimum effective" ends: tiny volumes, a fraction
of the consumable cost, orders of magnitude more throughput for single-cell work.
But droplets change the physics. There is no per-reaction rescue (a bad droplet
just dies), and tiny volumes raise the process noise. So a recipe that is merely
feasible on a plate, sitting on the razor edge of the decision bar, fails when you
port it. Only a ROBUST recipe survives: one whose unique-read margin above the bar
is wide enough to absorb the droplet noise.

That is the decision this module makes. Readiness is the probability a recipe still
decides under droplet noise. A recipe is droplet-ready when that probability clears
a bar. Porting a ready recipe unlocks a large jump in value per dollar (cheap
consumables times high throughput); porting a fragile one wastes an expensive run.

    readiness(x)         P(decisionable under droplet-scale noise)
    droplet_ready(x)     readiness(x) >= bar
    value_per_dollar     effective unique * throughput * target relevance / (cost * cost_mult)

The agent that searches for maximum value per dollar is therefore pushed toward
robust recipes on their own, because only robust recipes can claim the droplet jump.
`target_relevance` is the hook for a guided, clinical objective: weight the value by
how much a recipe advances the target so the search indexes on the therapy, not on
raw reads. Default is uniform. Pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .surface import LibrarySurface, cost, to_real, Vec


@dataclass(frozen=True)
class Platform:
    name: str
    cost_mult: float        # consumable cost multiplier (droplet is a fraction of plate)
    throughput: float       # relative throughput -> value multiplier (droplet is high)
    noise_mult: float       # process noise multiplier (droplet is higher, tiny volumes)
    rescue: bool            # can you intervene per reaction (plate yes, droplet no)


PLATE = Platform("plate", cost_mult=1.0, throughput=1.0, noise_mult=1.0, rescue=True)
DROPLET = Platform("droplet", cost_mult=0.12, throughput=25.0, noise_mult=2.5, rescue=False)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class DropletBridge:
    def __init__(self, surface: LibrarySurface, *, readiness_bar: float = 0.85,
                 target_relevance: Optional[Callable[[Vec], float]] = None,
                 plate: Platform = PLATE, droplet: Platform = DROPLET):
        self.s = surface
        self.readiness_bar = readiness_bar
        self.target_relevance = target_relevance or (lambda x: 1.0)
        self.plate = plate
        self.droplet = droplet

    def readiness(self, x: Vec) -> float:
        """P(a droplet still reads decisionable), given droplet-scale noise."""
        sd = self.s.noise_sd * self.droplet.noise_mult
        z = (self.s.unique_true(x) - self.s.unique_threshold) / sd
        return _normal_cdf(z)

    def droplet_ready(self, x: Vec) -> bool:
        return self.readiness(x) >= self.readiness_bar

    def _effective_unique(self, x: Vec, platform: Platform) -> float:
        """Unique reads actually recovered on a platform. On droplet there is no
        rescue, so each compartment succeeds only with probability readiness."""
        u = self.s.unique_true(x)
        if platform is self.droplet:
            return u * self.readiness(x)
        return u

    def value_per_dollar(self, x: Vec, platform: Platform) -> float:
        eff = self._effective_unique(x, platform)
        w = self.target_relevance(x)
        return (eff * platform.throughput * w) / (cost(x) * platform.cost_mult)

    def best_platform(self, x: Vec) -> Tuple[Platform, float]:
        vp = self.value_per_dollar(x, self.plate)
        vd = self.value_per_dollar(x, self.droplet)
        return (self.droplet, vd) if vd >= vp else (self.plate, vp)

    def true_optimum(self, grid: int = 22) -> Tuple[Vec, Platform, float]:
        """Best (recipe, platform) by value per dollar, among plate-feasible recipes."""
        best = (None, self.plate, -1.0)
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    x = (i / (grid - 1), j / (grid - 1), k / (grid - 1))
                    if not self.s.feasible(x):
                        continue
                    plat, v = self.best_platform(x)
                    if v > best[2]:
                        best = (x, plat, v)
        return best

    def describe(self, x: Vec) -> dict:
        plat, v = self.best_platform(x)
        return {"recipe": to_real(x), "platform": plat.name,
                "readiness": round(self.readiness(x), 3),
                "droplet_ready": self.droplet_ready(x),
                "value_per_dollar": round(v, 2)}
