"""
Library-economics response surface: the objective is unique reads per dollar.

A sequencing library is worth its UNIQUE molecules, not its raw reads. UMIs let
you count them by deduplication, so "unique reads per dollar" (URPD) is the honest
value of a recipe. This surface models that end to end from three titratable,
cost-bearing knobs (each normalized to [0,1]):

    pcr_cycles    too few, not enough usable library; too many, PCR duplicates
    input_ng      raises library complexity (unique molecules), spends sample
    reagent_frac  miniaturization: below a floor, efficiency falls off a cliff

    complexity    C(input, reagent)       unique molecules available
    informative   depth * mass * (1-dup)  reads that are not duplicates
    unique_reads  C * (1 - e^(-informative/C))   rarefaction, saturates at C
    cost          weighted knobs + a fixed floor
    URPD          unique_reads / cost

The optimum is interior: cycles have a sweet spot (under- vs over-amplification),
input trades complexity against cost, reagent must clear the cliff but no more.
That is what makes it worth searching, and worth teaching a prior about.

A run can also suffer a mechanical fault (bead loss): unique collapses, but the
fault is flagged (the CV layer), so an honest search excludes it. Pure Python.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

Vec = Tuple[float, ...]

CMAX = 2.0          # ceiling on library complexity (millions of unique molecules)
INP_SCALE = 55.0    # ng at which complexity is ~63% of the ceiling
DEPTH = 3.0         # sequencing depth actually run (millions of reads)


@dataclass(frozen=True)
class Knob:
    name: str
    lo: float
    hi: float
    is_int: bool
    log: bool
    cost_weight: float


KNOBS: List[Knob] = [
    Knob("pcr_cycles", 3, 16, True, False, 0.40),
    Knob("input_ng", 1, 250, False, True, 0.35),
    Knob("reagent_frac", 0.25, 1.0, False, False, 0.25),
]
DIM = len(KNOBS)
COST_FLOOR = 0.15   # fixed per-run cost (sequencing + handling), so URPD stays finite


def to_real(x: Vec) -> dict:
    out = {}
    for xi, k in zip(x, KNOBS):
        if k.log:
            v = math.exp(math.log(k.lo) + xi * (math.log(k.hi) - math.log(k.lo)))
        else:
            v = k.lo + xi * (k.hi - k.lo)
        out[k.name] = int(round(v)) if k.is_int else round(v, 3)
    return out


def to_norm(real: dict) -> Vec:
    """Inverse of to_real: map real parameters back to a [0,1]^d vector."""
    out = []
    for k in KNOBS:
        v = real[k.name]
        if k.log:
            xi = (math.log(v) - math.log(k.lo)) / (math.log(k.hi) - math.log(k.lo))
        else:
            xi = (v - k.lo) / (k.hi - k.lo)
        out.append(min(1.0, max(0.0, xi)))
    return tuple(out)


def cost(x: Vec) -> float:
    return COST_FLOOR + sum(xi * k.cost_weight for xi, k in zip(x, KNOBS))


def _smoothstep(v: float, a: float, b: float) -> float:
    if v <= a:
        return 0.0
    if v >= b:
        return 1.0
    t = (v - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Observation:
    x: Vec
    unique_obs: float       # observed unique reads (millions), after UMI dedup
    cost: float
    decision: bool          # did this run read enough unique molecules to decide?
    fault: bool             # mechanical fault flagged by the CV layer


class LibrarySurface:
    def __init__(self, seed: int = 0, unique_threshold: float = 0.75, target: float = 0.90,
                 fault_rate: float = 0.18, noise_sd: float = 0.04):
        self.seed = seed
        self.unique_threshold = unique_threshold   # unique reads (M) needed to decide
        self.target = target                       # required P(decisionable) for feasibility
        self.fault_rate = fault_rate
        self.noise_sd = noise_sd

    # -- planted economics ---------------------------------------------------
    def complexity(self, x: Vec) -> float:
        _, inp, rg = x
        input_ng = 250.0 ** inp
        eff = _smoothstep(0.25 + rg * 0.75, 0.30, 0.55)
        return CMAX * (1.0 - math.exp(-input_ng / INP_SCALE)) * eff

    def unique_true(self, x: Vec) -> float:
        cy = x[0]
        mass = 1.0 - math.exp(-2.5 * (cy + 0.10))   # too few cycles: not enough library
        dup = 1.0 - math.exp(-1.6 * cy)             # too many cycles: PCR duplicates
        informative = DEPTH * mass * (1.0 - dup)
        C = self.complexity(x)
        if C <= 1e-6:
            return 0.0
        return C * (1.0 - math.exp(-informative / C))

    def cost(self, x: Vec) -> float:
        return cost(x)

    def urpd_true(self, x: Vec) -> float:
        return self.unique_true(x) / cost(x)

    def decision_prob(self, x: Vec) -> float:
        z = (self.unique_true(x) - self.unique_threshold) / self.noise_sd
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def feasible(self, x: Vec) -> bool:
        return self.decision_prob(x) >= self.target

    def min_feasible_unique(self) -> float:
        lo, hi = self.unique_threshold, CMAX
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            z = (mid - self.unique_threshold) / self.noise_sd
            if 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) < self.target:
                lo = mid
            else:
                hi = mid
        return hi

    # -- a noisy run, with the occasional flagged fault ----------------------
    def evaluate(self, x: Vec, rng: random.Random) -> Observation:
        if rng.random() < self.fault_rate:
            return Observation(x, unique_obs=max(0.0, rng.gauss(0.02, 0.02)),
                               cost=cost(x), decision=False, fault=True)
        u = max(0.0, self.unique_true(x) + rng.gauss(0.0, self.noise_sd))
        return Observation(x, unique_obs=u, cost=cost(x),
                           decision=(u >= self.unique_threshold), fault=False)

    # -- the answer key: highest URPD among feasible recipes -----------------
    def true_optimum(self, grid: int = 24) -> Tuple[Vec, float]:
        best_x, best_u = None, -1.0
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    x = (i / (grid - 1), j / (grid - 1), k / (grid - 1))
                    if self.feasible(x):
                        u = self.urpd_true(x)
                        if u > best_u:
                            best_u, best_x = u, x
        return best_x, best_u
