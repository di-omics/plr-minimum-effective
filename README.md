# plr-minimum-effective

**Sequencing is billed per read, but only unique molecules carry information.** Run
too many PCR cycles and you pay full price for duplicates. Skimp on input DNA or
dilute the reagent too far and the library is too thin to call anything. Somewhere
in between sits the cheapest prep that still answers the question.

This repo finds that prep automatically. An agent searches the reaction parameters
for the highest **unique reads per dollar (URPD)** that still clears a decision bar,
warm-started by what a bench scientist teaches it, and it refuses to learn from runs
that a camera flagged as mechanical failures.

*Minimum effective* is borrowed from minimum effective dose: not the cheapest
reaction and not the biggest, but the smallest one that still works.

Pure Python, no dependencies, no hardware required to run any of it.

## Quickstart

```bash
python -m min_effective.run --seeds 16
```

```
PLANT: max unique-reads-per-dollar recipe = 5 cyc, 96 ng, reagent 0.54
       URPD 1.345 unique reads (M) per cost unit; only ~5% of recipes are even feasible

RECOVERY over 16 seeds, budget 60 (lower URPD gap = closer to optimum):
  strategy     feasible   median URPD gap
  bo_prior        11/16             0.164
  bo_cold         13/16             0.202
  random           5/16             0.230

CV honesty (taught agent): honest 11/16 feasible  vs  naive 3/16 feasible
```

## The objective

Three knobs are titratable and each one costs money:

    pcr_cycles    a sweet spot: too few, not enough library; too many, PCR duplicates
    input_ng      raises complexity (unique molecules), but spends sample
    reagent_frac  miniaturization: below a floor, efficiency falls off a cliff

They compose into value the way the chemistry does:

    complexity C  -> informative reads (depth minus duplicates) -> unique reads
    URPD          = unique reads / cost

A recipe is **feasible** only if it reads enough unique molecules to call the result
reliably, not just on average. The optimum is interior (every knob has a wrong answer
in both directions) and only about 5% of the space is feasible at all. That narrowness
is the whole reason a taught prior and a real optimizer beat guessing.

## How it is tested: plant and recover

You cannot score an optimizer against a real assay, because nobody knows the real
assay's true optimum. So this **plants** a known maximum-URPD recipe on a simulated
library-economics surface, hides it, lets the agent search blind under a fixed run
budget, and scores how close it got. That is what "plant and recover" means, and it
is why the numbers above are trustworthy as a statement about the *search*, even
though the surface is synthetic.

Three results fall out, each measured over fixed seeds at a 60-run budget:

**1. The optimizer earns its keep.** Only ~5% of recipes are feasible, so blind
space-filling mostly misses. Random recovers a feasible recipe 5 of 16 times; the
agent does it 11 to 13 of 16.

**2. Teaching lands closer to the optimum.** Seeded with a handful of expert
demonstrations in real units, `bo_prior` starts in the right neighborhood and closes
to a median URPD gap of 0.164, against 0.202 cold and 0.230 random. Note the tradeoff
that shows up honestly in the table: the cold agent clears feasibility slightly more
often, while the taught agent lands *nearer the actual optimum* when it lands. Tacit
knowledge buys you precision, not coverage.

**3. Excluding flagged faults keeps the search alive.** Runs sometimes fail
mechanically (bead loss, air gap). The honest arm drops CV-flagged runs and re-runs
them; the naive arm keeps them, so a botched well reads as bad chemistry and poisons
the model. Honest recovers 11 of 16; naive collapses to 3 of 16.

## Teach it what you know

```python
from min_effective import TeachablePrior, DiscoveryConfig, DiscoveryLoop

prior = TeachablePrior()
prior.teach(pcr_cycles=6, input_ng=90, reagent_frac=0.60, unique=0.90)   # good neighborhood
prior.teach(pcr_cycles=8, input_ng=90, reagent_frac=0.28, unique=0.05)   # reagent below the cliff fails
```

Demonstrations enter as down-weighted pseudo-observations: they point the search, and
real runs correct them. Being wrong costs you a little budget, not the result. See
[min_effective/prior.py](min_effective/prior.py).

## Why CV is the gate, not a garnish

The agent learns from readouts. If a run misfired mechanically and its bad number is
fed back as though it were chemistry, the model is poisoned and the recommendation
drifts. In-process computer vision is what catches those misfires at the step where
they happen, so the run is excluded and repeated instead of trusted.

```bash
python -m min_effective.vision
```

`VisionQC` models an imperfect camera with a recall (chance it flags a true fault)
and a specificity (chance it leaves a good run alone). Sweeping recall gives the
detector a hard spec:

    CV recall     feasible   median URPD gap
    0.0             3/16              0.194
    0.5             2/16              0.201
    0.8             7/16              0.130
    0.95            9/16              0.119
    1.0             9/16              0.106

**The loop needs roughly 0.8 recall to function.** Below that, the search is no
better than having no camera. That is a concrete number you can hand a vision team,
and it is what [lab-cv](https://github.com/di-omics/lab-cv) is built to earn.
`where_cv_sees(platform)` lists where to point it: on a plate, tip pickup,
aspiration, bead pellet, ethanol removal; in droplets, formation and monodispersity,
encapsulation, and junction clogs.

## Where the automation comes in

Optimizing on paper is the easy half. The reason this lives next to a
[PyLabRobot](https://github.com/PyLabRobot/pylabrobot) stack is that the loop only
closes if a proposed recipe can be executed, measured, and fed back without a human
in the middle:

    agent proposes a recipe
      -> compiled into a runnable liquid-handler method (PyLabRobot, liquid handler)
      -> CV watches the deck and flags mechanical faults in-process
      -> UMI-deduplicated unique count comes back as the readout
      -> flagged runs are excluded and repeated; clean runs update the model

In this repo the execute-and-measure step is a synthetic surface, which is what makes
recovery scorable against a known answer. The machine-side pieces live in
[plr-mcp](https://github.com/di-omics/plr-mcp) (protocol execution) and
[lab-cv](https://github.com/di-omics/lab-cv) (the fault detector). Nothing here has
been run on an instrument, and a discovered recipe is a candidate for wet validation,
not a validated protocol.

## The next layer: plate to droplet

Droplet microfluidics is where the cost curve bends again: a fraction of the
consumable cost and far more throughput for single-cell work. But droplets change the
physics. There is no per-reaction rescue and small volumes are noisier, so a recipe
has to be robust to survive the move. `DropletBridge` scores readiness as the
probability a recipe still decides under droplet noise.

```bash
python -m min_effective.bridge
```

    plate reads-per-dollar optimum   5 cyc, 96 ng   readiness 0.744   not droplet-ready
    droplet value optimum            6 cyc, 197 ng  readiness 0.891   droplet-ready

These are two different recipes, and that is the point. The plate optimum spends the
least to *just* clear the bar, so it has no margin left to miniaturize: porting it is
a 3.3x setback. Porting the robust recipe is a 180x gain. So an agent that optimizes
plate cost alone will systematically walk away from the chemistries that could have
ported. If you want droplet readiness, you have to optimize the droplet objective,
which prices robustness. `DropletBridge` also carries a `target_relevance` term, so
the objective can be pointed at the decision you actually care about rather than at
raw reads.

## Layout

    min_effective/surface.py   library economics: complexity, duplication, unique reads, URPD
    min_effective/agent.py     cost-aware agent that maximizes URPD, warm-startable
    min_effective/prior.py     the teachable prior (expert demonstrations)
    min_effective/loop.py      closed loop + scoring against the plant
    min_effective/droplet.py   plate-vs-droplet economics + readiness
    min_effective/bridge.py    the port decision, structural plant-and-recover
    min_effective/vision.py    CV as the fault-detection gate; recall vs recommend quality
    min_effective/run.py       aggregate CLI
    tests/                     plant-and-recover tests

## Scope

Synthetic surface throughout. The numbers describe the modeled economics from a fixed
seed, not a wet-lab measurement, and no real or proprietary data is used anywhere in
this repo. What the results support is a claim about the search strategy: teaching
helps, the optimizer beats random, and CV recall below ~0.8 breaks the loop.

## License

[MIT](LICENSE)
