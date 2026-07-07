# plr-minimum-effective

An agent that discovers a **minimal effective reaction**: the recipe with the most
**unique reads per dollar** that still yields a decisionable result. It is warm-
started by what you teach it, and it keeps the search honest by excluding
CV-flagged mechanical faults. Plant-and-recover, pure Python, no hardware.

The name is the point. A library is worth its unique molecules, not its raw reads,
and UMIs let you count them by deduplication. So the objective is not "cheapest"
and not "highest yield" but **unique reads per dollar** (URPD), which balances
library complexity, PCR duplication, and cost end to end.

```bash
python -m min_effective.run --seeds 16
```

## What it does

It plants a known maximum-URPD recipe on a library-economics response surface,
then lets a cost-constrained agent rediscover it blind under a run budget, and
scores recovery against the plant. In production, evaluating a candidate means
compiling a [plr-clarity](https://github.com/di-omics/plr-clarity) `dna_ultra2_umi`
config to a runnable Hamilton STAR method and reading the UMI-deduplicated unique
count; here a synthetic surface stands in so recovery can be scored against a
known answer.

The surface models the economics from three titratable, cost-bearing knobs:

    pcr_cycles    a sweet spot: too few, not enough library; too many, PCR duplicates
    input_ng      raises complexity (unique molecules), spends sample
    reagent_frac  miniaturization: below a floor, efficiency falls off a cliff

    complexity C  -> informative reads (depth minus duplicates) -> unique reads
    URPD          = unique reads / cost

The optimum is interior and the feasible region is narrow (only a few percent of
recipes clear the decision bar), which is exactly why a taught prior and a real
agent earn their keep.

## The three effects, measured honestly

Over fixed seeds at a 60-run budget (numbers are printed by `min_effective.run`):

- **The agent finds the needle; random does not.** Only about 5% of recipes are
  feasible, so blind space-filling mostly misses it. The agent (taught or cold)
  recovers a feasible high-URPD recipe far more often than random.
- **Teaching lands closer to the optimum.** `bo_prior` (seeded with a few expert
  demonstrations, in real units) starts in the right neighborhood, so it recovers
  a recipe nearer the true maximum URPD than the cold-start agent (a lower URPD
  gap). This is where end-to-end tacit knowledge enters: you teach it that a
  reaction is cheaper AND reads more unique molecules per dollar, and it refines
  from there.
- **CV-cleaning keeps the search alive.** A run can suffer a mechanical fault
  (bead loss); the CV layer flags it. The honest arm excludes flagged faults and
  re-runs; the naive arm keeps them, so a botched well reads as a bad recipe and
  sinks the search. Honest recovers a feasible recipe far more often than naive.

Every discovered recipe is a candidate for a validation ladder, which confirms it
with a liquid test before it is trusted.

## Teach it what you know

```python
from min_effective import TeachablePrior, DiscoveryConfig, DiscoveryLoop

prior = TeachablePrior()
prior.teach(pcr_cycles=6, input_ng=90, reagent_frac=0.60, unique=0.90)   # good neighborhood
prior.teach(pcr_cycles=8, input_ng=90, reagent_frac=0.28, unique=0.05)   # reagent below the cliff fails
# ... a few examples in real units, with the unique reads you expect
```

Demonstrations enter as down-weighted pseudo-observations: they point the search,
and real runs correct them. See `min_effective/prior.py`.

## The next layer: plate to droplet

Minimum effective has a ceiling, and droplet microfluidics is it: a fraction of the
consumable cost and orders of magnitude more throughput for single-cell work. But
droplets change the physics. There is no per-reaction rescue, and tiny volumes raise
the noise, so a recipe has to be robust to survive the port. `DropletBridge` scores
that: readiness is the probability a recipe still decides under droplet noise.

```bash
python -m min_effective.bridge
```

The sharp result is structural, not a tuning artifact:

    plate reads-per-dollar optimum   razor edge, not droplet-ready  -> porting is a setback
    droplet value optimum            robust, droplet-ready          -> porting is a ~180x jump

Those are two different recipes. The plate optimum spends the least to just clear the
bar, so it cannot miniaturize; porting it wastes a droplet campaign. The recipe you
want is a robust one with a wide margin. So an agent that optimizes plate reads-per-
dollar will systematically avoid the chemistries that can port. To decide droplet-
readiness well, the agent has to optimize the droplet objective, which values
robustness, and one layer further a **clinical objective** (`target_relevance` in
`DropletBridge`) so the loop indexes on advancing the therapy, not on raw reads. That
is how you keep R and D pointed at biological truth instead of meandering toward cheap
minima: reward information toward the target decision per dollar, and miniaturize the
moment a chemistry is robust enough to survive it.

## Layout

    min_effective/surface.py   library economics: complexity, duplication, unique reads, URPD
    min_effective/agent.py     cost-aware agent that maximizes URPD, warm-startable
    min_effective/prior.py     the teachable prior (expert demonstrations)
    min_effective/loop.py      closed loop + scoring against the plant
    min_effective/droplet.py   plate-vs-droplet economics + readiness (the port objective)
    min_effective/bridge.py    the port decision, structural plant-and-recover
    min_effective/run.py       aggregate CLI
    tests/                     plant-and-recover tests

## Note

Synthetic surface throughout; results are for the modeled economics, generated
from a fixed seed. No real or proprietary data. Pure Python, no numpy, runs
anywhere.

## License

[MIT](LICENSE)
