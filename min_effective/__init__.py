"""plr-minimum-effective: discover the maximum unique-reads-per-dollar recipe.

An agent learns a minimal effective reaction under a run budget, warm-started by
what you teach it, with CV-flagged mechanical faults excluded so the search stays
honest. Plant-and-recover; pure Python; no hardware.
"""

from .surface import LibrarySurface, KNOBS, to_real, to_norm, cost
from .agent import URPDAgent
from .prior import TeachablePrior, default_expert_prior
from .loop import DiscoveryConfig, DiscoveryLoop, compare
from .droplet import DropletBridge, Platform, PLATE, DROPLET
from .bridge import structural, realized_value

__all__ = [
    "LibrarySurface", "KNOBS", "to_real", "to_norm", "cost",
    "URPDAgent", "TeachablePrior", "default_expert_prior",
    "DiscoveryConfig", "DiscoveryLoop", "compare",
    "DropletBridge", "Platform", "PLATE", "DROPLET", "structural", "realized_value",
]
