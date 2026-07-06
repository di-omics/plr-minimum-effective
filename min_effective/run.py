"""
Discover the maximum unique-reads-per-dollar recipe, plant-and-recover.

    python -m min_effective.run                 # aggregate over seeds, all three arms
    python -m min_effective.run --seeds 20 --budget 45 --report out.json

The value here is statistical, so this aggregates over seeds rather than trusting
one run. It plants a known maximum-URPD recipe, then reports how three strategies
recover it under a run budget:

    bo_prior   taught agent (expert demonstrations) + acquisition
    bo_cold    same agent, cold start
    random     random proposals

and, for the taught agent, honest (CV faults excluded) vs naive (faults kept).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys

from .loop import DiscoveryLoop, DiscoveryConfig
from .surface import LibrarySurface, to_real


def _fmt(r):
    return f"{r['pcr_cycles']} cyc, {r['input_ng']:.0f} ng, reagent {r['reagent_frac']:.2f}"


def _arm(strategy, exclude, seeds, budget, fault_rate, candidate_n):
    feas, gaps, per = 0, [], []
    for seed in range(seeds):
        d = DiscoveryLoop(DiscoveryConfig(
            seed=seed, strategy=strategy, exclude_faults=exclude,
            budget=budget, fault_rate=fault_rate, candidate_n=candidate_n)).run()
        per.append(d)
        if d["found"] and d["feasible"]:
            feas += 1
            gaps.append(d["urpd_gap"])
    return {"feasible": feas, "seeds": seeds,
            "median_urpd_gap": round(st.median(gaps), 3) if gaps else None, "per": per}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Discover the maximum unique-reads-per-dollar recipe")
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--fault-rate", type=float, default=0.10)
    p.add_argument("--candidate-n", type=int, default=400)
    p.add_argument("--report", default="")
    a = p.parse_args(argv)

    s = LibrarySurface(seed=0, fault_rate=a.fault_rate)
    tx, turpd = s.true_optimum(grid=24)
    rng = random.Random(0)
    frac = sum(s.feasible(tuple(rng.random() for _ in range(3))) for _ in range(4000)) / 4000

    print(f"\nPLANT: max unique-reads-per-dollar recipe = {_fmt(to_real(tx))}")
    print(f"       URPD {turpd:.3f} unique reads (M) per cost unit;"
          f" only ~{frac*100:.0f}% of recipes are even feasible\n")

    arms = {k: _arm(k, True, a.seeds, a.budget, a.fault_rate, a.candidate_n)
            for k in ("bo_prior", "bo_cold", "random")}
    print(f"RECOVERY over {a.seeds} seeds, budget {a.budget} (lower URPD gap = closer to optimum):")
    print(f"  {'strategy':<11}{'feasible':>10}{'median URPD gap':>18}")
    for k in ("bo_prior", "bo_cold", "random"):
        g = arms[k]["median_urpd_gap"]
        feas = f"{arms[k]['feasible']}/{a.seeds}"
        gap = f"{g:.3f}" if g is not None else "-"
        print(f"  {k:<11}{feas:>10}{gap:>18}")

    naive = _arm("bo_prior", False, a.seeds, a.budget, a.fault_rate, a.candidate_n)
    print(f"\nCV honesty (taught agent): honest {arms['bo_prior']['feasible']}/{a.seeds} feasible"
          f"  vs  naive {naive['feasible']}/{a.seeds} feasible")

    print("\n  Teaching (bo_prior vs bo_cold) starts the search in the right neighborhood, so it")
    print("  lands closer to the max-URPD recipe. The agent (vs random) matters because only a")
    print("  few percent of recipes are feasible. Excluding CV-flagged faults (honest vs naive)")
    print("  keeps mechanical misfires from masquerading as bad chemistry and sinking the search.\n")

    if a.report:
        with open(a.report, "w") as fh:
            json.dump({"plant": {"recipe": to_real(tx), "urpd": turpd, "feasible_fraction": frac},
                       "arms": {k: {kk: vv for kk, vv in arms[k].items() if kk != "per"} for k in arms},
                       "naive": {kk: vv for kk, vv in naive.items() if kk != "per"}}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
