"""
The role of computer vision in the loop: it decides which runs the agent trusts.

The agent learns from readouts. If a run misfired mechanically (bead loss, air gap,
a collapsed droplet) but its bad number is fed in as if it were real chemistry, the
surrogate is poisoned and the recommendation drifts. CV is what catches those
misfires in-process, at the step where they happen, and flags the run so it is
excluded and re-run instead of trusted. So CV is not a nice-to-have QC add-on. It is
the gate on the training data, and the quality of the recommendation is a steep
function of how good the camera is.

Real CV is not perfect. `VisionQC` models it with a recall (probability it flags a
true fault) and a specificity (probability it leaves a good run alone). A missed
fault poisons the fit; a false alarm wastes budget on a needless re-run. Sweeping
recall from 0 (no CV) to 1 (perfect CV) shows how hard the camera has to hit for the
autonomous loop to work, which is a concrete spec for the vision system.

Where CV sees the flow (the checkpoints) differs by platform, plate vs droplet.
This is the CV strategy: where to point the camera and what each view catches.
"""

from __future__ import annotations

import random
import statistics as st
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VisionQC:
    """An imperfect in-process CV verdict. recall = P(flag | fault) (sensitivity);
    specificity = P(no flag | good run). recall 0 is no camera; recall 1, specificity
    1 is a perfect one. Real detectors live in between."""

    recall: float = 0.95
    specificity: float = 0.98

    def inspect(self, true_fault: bool, rng: random.Random) -> bool:
        if true_fault:
            return rng.random() < self.recall            # caught it (or missed it)
        return rng.random() < (1.0 - self.specificity)   # false alarm on a good run


# where CV is pointed, and what each view catches. (platform, what it catches)
CHECKPOINTS: Dict[str, tuple] = {
    "tip_pickup":        ("plate",   "missed or double tip; a whole channel off"),
    "aspiration":        ("plate",   "air gap, clot, or empty reservoir; no liquid moved"),
    "bead_pellet":       ("plate",   "beads not pelleted, or pulled off with the supernatant"),
    "etoh_removal":      ("plate",   "residual ethanol; downstream inhibition"),
    "droplet_formation": ("droplet", "polydisperse or satellite drops; variable reaction volume"),
    "encapsulation":     ("droplet", "double or empty drops (Poisson); doublets and blanks"),
    "junction_clog":     ("droplet", "channel clog; the run collapses"),
}


def where_cv_sees(platform: str) -> Dict[str, str]:
    """The checkpoints CV should watch for a given platform."""
    return {k: v[1] for k, v in CHECKPOINTS.items() if v[0] == platform}


def recall_sweep(recalls=(0.0, 0.5, 0.8, 0.95, 1.0), *, seeds: int = 16,
                 specificity: float = 0.98, budget: int = 60,
                 candidate_n: int = 300) -> List[Dict]:
    """Recommendation quality as a function of CV recall. Lower recall means more
    mechanical faults slip through as if they were chemistry, poisoning the search."""
    from .loop import DiscoveryLoop, DiscoveryConfig
    rows = []
    for r in recalls:
        feas, gaps = 0, []
        for seed in range(seeds):
            d = DiscoveryLoop(DiscoveryConfig(
                seed=seed, strategy="bo_prior", budget=budget, candidate_n=candidate_n,
                vision=VisionQC(recall=r, specificity=specificity))).run()
            if d["found"] and d["feasible"]:
                feas += 1
                gaps.append(d["urpd_gap"])
        rows.append({"recall": r, "feasible": feas, "seeds": seeds,
                     "median_urpd_gap": round(st.median(gaps), 3) if gaps else None})
    return rows


def main(argv=None) -> int:
    print("\nWhere CV sees the flow (point the camera here):")
    for platform in ("plate", "droplet"):
        print(f"  {platform}:")
        for name, catches in where_cv_sees(platform).items():
            print(f"    {name:<18} {catches}")

    print("\nHow hard the camera has to hit (recommendation quality vs CV recall):")
    print(f"  {'CV recall':<12}{'feasible':>10}{'median URPD gap':>18}")
    for row in recall_sweep():
        g = row["median_urpd_gap"]
        feas = f"{row['feasible']}/{row['seeds']}"
        gap = f"{g:.3f}" if g is not None else "-"
        print(f"  {row['recall']:<12}{feas:>10}{gap:>18}")
    print("\n  recall 0 is no camera (every mechanical fault is read as bad chemistry); recall 1")
    print("  is a perfect one. The climb between them is CV being heavy-hitting in the recommends:")
    print("  the detector's recall is a hard requirement for the autonomous loop, not a garnish.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
