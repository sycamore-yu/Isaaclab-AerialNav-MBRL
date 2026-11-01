"""Default promotion must not silently change other optimization paths."""

import argparse

import pytest

from aerialnav.diffnav.cli import add_train_args
from aerialnav.diffnav.config import DiffNavTrainCfg


def test_default_and_explicit_deferred_override():
    assert DiffNavTrainCfg().loss_timing == "pre_reset"
    assert DiffNavTrainCfg(loss_timing="deferred").loss_timing == "deferred"


@pytest.mark.parametrize("algorithm", ("bptt", "shac", "ahac"))
def test_cli_resolves_default_timing_after_algorithm_and_velocity_selection(algorithm):
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args(["--algorithm", algorithm])
    cfg = DiffNavTrainCfg(algorithm=algorithm, velocity_loss_mode=args.velocity_loss_mode, loss_timing=args.loss_timing)
    cfg.validate()
    assert cfg.loss_timing == ("pre_reset" if algorithm == "bptt" else "deferred")
