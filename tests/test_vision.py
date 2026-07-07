"""Tests for the role of CV in the loop: fault detection gates the training data."""

import random

from min_effective.vision import VisionQC, CHECKPOINTS, where_cv_sees
from min_effective.loop import DiscoveryConfig, DiscoveryLoop


def _feasible(recall, n=12, specificity=0.98):
    feas = 0
    for seed in range(n):
        d = DiscoveryLoop(DiscoveryConfig(
            seed=seed, strategy="bo_prior", candidate_n=300,
            vision=VisionQC(recall=recall, specificity=specificity))).run()
        if d["found"] and d["feasible"]:
            feas += 1
    return feas


def test_visionqc_hits_its_recall_and_specificity():
    v = VisionQC(recall=0.9, specificity=0.95)
    rng = random.Random(0)
    caught = sum(v.inspect(True, rng) for _ in range(4000)) / 4000
    false_alarm = sum(v.inspect(False, rng) for _ in range(4000)) / 4000
    assert 0.86 < caught < 0.94                    # recall
    assert 0.02 < false_alarm < 0.08               # 1 - specificity


def test_checkpoints_split_by_platform():
    plate = where_cv_sees("plate")
    droplet = where_cv_sees("droplet")
    assert plate and droplet
    assert set(plate).isdisjoint(droplet)          # different views for different flows
    assert set(plate) | set(droplet) == set(CHECKPOINTS)


def test_recall_is_heavy_hitting_in_the_recommends():
    high = _feasible(1.0)
    low = _feasible(0.0)
    assert high >= low + 2                          # a good camera recovers far more feasibly
    assert high >= 5                                # perfect CV works
    assert low <= 6                                 # no camera struggles


def test_vision_is_opt_in_and_backward_compatible():
    # with no VisionQC, the loop uses the exclude_faults oracle unchanged
    d = DiscoveryLoop(DiscoveryConfig(seed=0, strategy="bo_prior",
                                      candidate_n=300, true_grid=16)).run()
    assert "found" in d and "runs" in d
