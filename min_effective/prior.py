"""
A teachable prior: what you tell the agent before it runs anything.

This is where end-to-end tacit knowledge enters. An expert who has run the assay
knows things the agent would otherwise pay runs to learn: the reagent floor below
which nothing works, that a handful of cycles beats a pile of them, that input is
the complexity lever. You teach those as example runs, in real units, with the
unique reads you expect. They seed the surrogate as down-weighted pseudo-
observations, so the search starts in the right neighborhood and spends its real
budget refining rather than rediscovering what you already know.

    prior = TeachablePrior()
    prior.teach(pcr_cycles=6, input_ng=90, reagent_frac=0.60, unique=0.85)
    prior.teach(pcr_cycles=8, input_ng=90, reagent_frac=0.28, unique=0.05)  # reagent below the cliff
    ...

The demonstrations are approximate on purpose. A prior points; data corrects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .surface import to_norm, Vec


@dataclass
class Demo:
    real: dict
    unique: float


@dataclass
class TeachablePrior:
    weight: float = 0.5                       # how much a demonstration counts vs a real run
    demos: List[Demo] = field(default_factory=list)

    def teach(self, *, pcr_cycles: float, input_ng: float, reagent_frac: float, unique: float):
        self.demos.append(Demo({"pcr_cycles": pcr_cycles, "input_ng": input_ng,
                                "reagent_frac": reagent_frac}, unique))
        return self

    def pseudo_observations(self) -> List[Tuple[Vec, float, float]]:
        """(normalized x, unique, weight) triples to seed the surrogate."""
        return [(to_norm(d.real), d.unique, self.weight) for d in self.demos]


def default_expert_prior() -> TeachablePrior:
    """A small, deliberately approximate prior encoding common tacit knowledge:
    reagent must clear the miniaturization floor, a few cycles beat many, and
    input is the complexity lever. Roughly right, not exact."""
    p = TeachablePrior(weight=0.5)
    p.teach(pcr_cycles=6, input_ng=90, reagent_frac=0.60, unique=0.90)   # good neighborhood
    p.teach(pcr_cycles=7, input_ng=120, reagent_frac=0.55, unique=0.85)
    p.teach(pcr_cycles=8, input_ng=90, reagent_frac=0.28, unique=0.05)   # reagent below the cliff fails
    p.teach(pcr_cycles=15, input_ng=90, reagent_frac=0.60, unique=0.45)  # too many cycles: duplicates
    p.teach(pcr_cycles=6, input_ng=8, reagent_frac=0.60, unique=0.20)    # too little input: low complexity
    return p
