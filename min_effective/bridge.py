"""
The port decision, and why the objective has to change to make it.

Porting a droplet-ready chemistry unlocks a large jump in value per dollar. Porting
a fragile one wastes a whole droplet campaign: precious primary cells and weeks,
gone. So the question is not "can I port" but "which recipe, and is it ready".

The sharp result: the recipe that maximizes plate reads-per-dollar is NOT the recipe
you want to port. The plate optimum sits on the razor edge of the decision bar (it
spends the least to just clear it), so it is not droplet-ready, and porting it is a
setback. The recipe you want is a robust one, a little more input for a wide margin
above the bar, which survives miniaturization. Those are two different recipes.

    plate reads-per-dollar optimum   razor edge, not droplet-ready  -> porting is a setback
    droplet value optimum            robust, droplet-ready          -> porting is the jump

The implication is the whole thesis of this repo taken one layer up: an agent that
optimizes plate reads-per-dollar will systematically avoid the robust chemistries
that can miniaturize. To decide droplet-readiness well, the agent has to optimize
the droplet (and, one layer further, the clinical) objective, which values
robustness. `DropletBridge` is that objective. Pure Python, plant-and-recover.
"""

from __future__ import annotations

import statistics as st
from typing import Dict

from .surface import LibrarySurface, to_real
from .droplet import DropletBridge, PLATE, DROPLET

FAIL_SETBACK = 0.30     # realized value of a botched droplet campaign, relative to staying on plate


def realized_value(bridge: DropletBridge, x, port: bool) -> float:
    if not port:
        return bridge.value_per_dollar(x, PLATE)
    if bridge.droplet_ready(x):
        return bridge.value_per_dollar(x, DROPLET)
    return bridge.value_per_dollar(x, PLATE) * FAIL_SETBACK   # ported a fragile recipe -> setback


def structural(seed: int = 0, grid: int = 24) -> Dict:
    """Compare the plate reads-per-dollar optimum against the droplet value optimum
    on the true surface. No agent noise: this is the structural fact."""
    s = LibrarySurface(seed=seed)
    b = DropletBridge(s)
    a_x, _ = s.true_optimum(grid=grid)                    # plate URPD optimum (razor edge)
    b_x, _, _ = b.true_optimum(grid=grid)                 # droplet value optimum (robust)

    a_ported = realized_value(b, a_x, True)
    a_stay = realized_value(b, a_x, False)
    b_ported = realized_value(b, b_x, True)
    return {
        "plate_reads_per_dollar_optimum": {
            **b.describe(a_x),
            "ported_realized": round(a_ported, 1),
            "stay_on_plate": round(a_stay, 1),
        },
        "droplet_value_optimum": {
            **b.describe(b_x),
            "ported_realized": round(b_ported, 1),
        },
        "same_recipe": to_real(a_x) == to_real(b_x),
        "jump_over_plate": round(b_ported / max(a_stay, 1e-9), 1),      # value of miniaturizing right
        "cost_of_porting_wrong": round(a_stay / max(a_ported, 1e-9), 1),  # setback from porting the razor
    }


def _fmt(d):
    return (f"{d['recipe']['pcr_cycles']} cyc, {d['recipe']['input_ng']:.0f} ng, "
            f"reagent {d['recipe']['reagent_frac']:.2f}")


def main(argv=None) -> int:
    r = structural(seed=0)
    a = r["plate_reads_per_dollar_optimum"]
    b = r["droplet_value_optimum"]
    print("\nPLATE reads-per-dollar optimum (what a plate objective picks):")
    print(f"  {_fmt(a)}  readiness {a['readiness']}  droplet_ready {a['droplet_ready']}")
    print(f"  stay on plate = {a['stay_on_plate']} value/$   |   port it anyway = {a['ported_realized']} value/$ (setback)")
    print("\nDROPLET value optimum (what the droplet objective picks):")
    print(f"  {_fmt(b)}  readiness {b['readiness']}  droplet_ready {b['droplet_ready']}")
    print(f"  ported = {b['ported_realized']} value/$")
    print(f"\n  different recipes: {not r['same_recipe']}")
    print(f"  porting the RIGHT (robust) recipe is a {r['jump_over_plate']}x jump over staying on plate.")
    print(f"  porting the WRONG (razor) recipe is a setback ({r['cost_of_porting_wrong']}x worse than not porting).")
    print("\n  So the agent cannot just optimize plate reads-per-dollar and then port. It has to")
    print("  optimize the droplet (and, one layer up, the clinical) objective, which values the")
    print("  robustness that survives miniaturization. That objective is DropletBridge.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
