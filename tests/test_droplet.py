"""Tests for the plate-to-droplet readiness bridge."""

from min_effective.surface import LibrarySurface
from min_effective.droplet import DropletBridge, PLATE, DROPLET
from min_effective.bridge import structural, realized_value


def test_readiness_increases_with_unique():
    s = LibrarySurface(seed=0)
    b = DropletBridge(s)
    lo = (0.2, 0.5, 0.6)      # ~16 ng: low complexity, low unique
    hi = (0.3, 0.95, 0.6)     # ~190 ng: high complexity, high unique
    assert b.readiness(hi) > b.readiness(lo)


def test_plate_optimum_is_not_droplet_ready():
    s = LibrarySurface(seed=0)
    b = DropletBridge(s)
    ax, _ = s.true_optimum(grid=20)
    assert not b.droplet_ready(ax)                     # the plate optimum is a razor edge


def test_structural_optima_differ_and_only_robust_ports():
    r = structural(seed=0, grid=22)
    assert r["plate_reads_per_dollar_optimum"]["droplet_ready"] is False
    assert r["droplet_value_optimum"]["droplet_ready"] is True
    assert r["same_recipe"] is False                   # different recipes
    assert r["jump_over_plate"] > 20                   # porting the robust recipe is a big jump
    assert r["cost_of_porting_wrong"] > 1              # porting the razor recipe is a setback


def test_droplet_value_beats_plate_for_a_ready_recipe():
    s = LibrarySurface(seed=0)
    b = DropletBridge(s)
    bx, _, _ = b.true_optimum(grid=20)
    assert b.value_per_dollar(bx, DROPLET) > 50 * b.value_per_dollar(bx, PLATE)
    assert realized_value(b, bx, True) > realized_value(b, bx, False)   # porting the ready recipe wins
