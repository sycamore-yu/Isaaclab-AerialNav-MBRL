"""Training configuration, CLI defaults and geometry selection."""

import argparse
import contextlib
import io
import json
import unittest
from dataclasses import asdict

import pytest
import torch

from aerialnav.diffnav.cli import add_train_args, parse_train_args
from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.physics import DEFAULT_FRONTEND, DEFAULT_PHYSICS, parse_physics_args
from aerialnav.tasks.navigation.geometry import nearest_obstacle_signed_distance


class DiffNavTrainConfigTests(unittest.TestCase):
    def test_runtime_selectors_preserve_training_remainder_and_checkpoint_identity(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument("--train_args", nargs=argparse.REMAINDER, default=[])
        defaults = parse_physics_args(parser, [])
        self.assertEqual((defaults.physics_preset, defaults.frontend), (DEFAULT_PHYSICS, DEFAULT_FRONTEND))
        self.assertEqual((DiffNavTrainCfg().physics, DiffNavTrainCfg().frontend), (DEFAULT_PHYSICS, DEFAULT_FRONTEND))
        for selectors in (
            ["physics=newton_mjwarp", "--frontend", "warp"],
            ["--frontend=warp", "physics=newton_mjwarp"],
        ):
            with self.subTest(selectors=selectors):
                args = parse_physics_args(parser, ["--train_args", "--timesteps", "150", *selectors])
                self.assertEqual((args.physics_preset, args.frontend), ("newton_mjwarp", "warp"))
                self.assertFalse(hasattr(args, "physics"))
                self.assertEqual(args.train_args, ["--timesteps", "150"])
                cfg = DiffNavTrainCfg(physics=args.physics_preset, frontend=args.frontend)
                cfg.validate()
                restored = DiffNavTrainCfg(**json.loads(json.dumps(asdict(cfg))))
                restored.validate()
                self.assertEqual((restored.physics, restored.frontend), ("newton_mjwarp", "warp"))
                self.assertIsNone(restored.max_iterations)
                self.assertFalse(restored.stop_after_acceptance)

    def test_torch_frontend_accepts_both_physics_backends(self):
        for physics in ("isaacsim_physx", "newton_mjwarp"):
            with self.subTest(physics=physics):
                args = parse_physics_args(argparse.ArgumentParser(), ["--frontend", "torch", f"physics={physics}"])
                self.assertEqual((args.physics_preset, args.frontend), (physics, "torch"))
                self.assertFalse(hasattr(args, "physics"))
                DiffNavTrainCfg(physics=args.physics_preset, frontend=args.frontend).validate()

    def test_explicit_physx_without_frontend_uses_compatible_torch(self):
        args = parse_physics_args(argparse.ArgumentParser(), ["physics=isaacsim_physx"])
        self.assertEqual((args.physics_preset, args.frontend), ("isaacsim_physx", "torch"))
        DiffNavTrainCfg(physics=args.physics_preset, frontend=args.frontend).validate()

    def test_invalid_physics_selector_cannot_silently_launch_physx(self):
        for tokens in (["physics=newton"], ["physics=newton_mjwarp", "physics=isaacsim_physx"]):
            with self.subTest(tokens=tokens), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_physics_args(argparse.ArgumentParser(), tokens)
        with self.assertRaisesRegex(ValueError, "physics"):
            DiffNavTrainCfg(physics="newton").validate()

    def test_invalid_frontend_cannot_silently_launch_torch(self):
        for tokens in (
            ["--frontend", "numpy"],
            ["--frontend"],
            ["--frontend=warp", "--frontend", "warp", "physics=newton_mjwarp"],
            ["--frontend=torch", "--frontend", "warp", "physics=newton_mjwarp"],
            ["--frontend", "warp", "physics=isaacsim_physx"],
        ):
            with self.subTest(tokens=tokens), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_physics_args(argparse.ArgumentParser(), tokens)
        with self.assertRaisesRegex(ValueError, "frontend"):
            DiffNavTrainCfg(frontend="numpy").validate()
        with self.assertRaisesRegex(ValueError, "requires physics=newton_mjwarp"):
            DiffNavTrainCfg(physics="isaacsim_physx", frontend="warp").validate()

    def test_restart_defaults_and_ahac_expansion_limit_are_explicit(self):
        cfg = DiffNavTrainCfg()
        cfg.validate()
        self.assertEqual((cfg.optimizer, cfg.weight_decay), ("adamw", 0.01))
        self.assertEqual(cfg.height_loss_mode, "goal_height_band")
        self.assertEqual(cfg.height_target_range, (0.5, 2.5))
        self.assertEqual(cfg.gradient_projection, "none")
        self.assertEqual(cfg.plateau_patience, 2)
        self.assertEqual((cfg.ahac_min_horizon, cfg.ahac_initial_horizon, cfg.ahac_max_horizon), (16, 32, 300))
        # A permitted initial H above the previous 150 cap is not silently
        # limited by the independent BPTT timesteps setting.
        DiffNavTrainCfg(algorithm="ahac", variant="terminal", ahac_initial_horizon=200).validate()

    def test_default_velocity_mode_uses_terminal_completion_for_continuous_and_dcol(self):
        for variant, expected in (
            ("step", "per_step"),
            ("continuous", "truncated_window"),
            ("terminal", "truncated_window"),
        ):
            with self.subTest(variant=variant):
                cfg = DiffNavTrainCfg(variant=variant, velocity_loss_mode="variant_default")
                cfg.validate()
                self.assertEqual(cfg.velocity_loss_mode, "variant_default")
                self.assertEqual(cfg.resolved_velocity_loss_mode, expected)

    def test_explicit_velocity_modes_are_independent_of_dcol_algorithm_and_preserve_critic_window(self):
        for algorithm in ("bptt", "shac", "ahac"):
            for mode in ("per_step", "truncated_window", "full_window"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    cfg = DiffNavTrainCfg(
                        variant="terminal", algorithm=algorithm, velocity_loss_mode=mode, velocity_window_s=2.0
                    )
                    cfg.validate()
                    self.assertEqual(cfg.resolved_velocity_loss_mode, "window" if mode == "full_window" else mode)
                    self.assertEqual(cfg.velocity_window_steps, 40)
                    payload = json.loads(json.dumps(asdict(cfg)))
                    self.assertEqual(payload["velocity_loss_mode"], mode)
                    restored = DiffNavTrainCfg(**payload)
                    restored.validate()
                    self.assertEqual(restored.resolved_velocity_loss_mode, "window" if mode == "full_window" else mode)
                    self.assertEqual(restored.velocity_window_steps, 40)
        for mode in ("per_step", "truncated_window", "full_window"):
            DiffNavTrainCfg(variant="continuous", velocity_loss_mode=mode).validate()

    def test_invalid_velocity_modes_fail_and_explicit_modes_keep_window_validation(self):
        for value in ("window", "step", "truncated", "TRUNCATED_WINDOW", "", True):
            with self.subTest(value=value):
                cfg = DiffNavTrainCfg(velocity_loss_mode=value)
                with self.assertRaisesRegex(ValueError, "velocity_loss_mode"):
                    cfg.validate()
                with self.assertRaisesRegex(ValueError, "velocity_loss_mode"):
                    _ = cfg.resolved_velocity_loss_mode
        for mode in ("per_step", "truncated_window", "full_window"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "velocity_window_s"):
                DiffNavTrainCfg(variant="terminal", velocity_loss_mode=mode, velocity_window_s=0.125).validate()
        with self.assertRaisesRegex(ValueError, "variant"):
            DiffNavTrainCfg(variant="truncated_window").validate()

    def test_height_modes_and_target_range_roundtrip_with_explicit_optimizer_controls(self):
        for mode in ("target_exp", "target_huber", "boundary_huber"):
            for optimizer in ("adam", "adamw"):
                for decay in (0.0, 0.01):
                    with self.subTest(mode=mode, optimizer=optimizer, decay=decay):
                        cfg = DiffNavTrainCfg(
                            height_loss_mode=mode,
                            height_target_range=(0.6, 2.4),
                            optimizer=optimizer,
                            weight_decay=decay,
                        )
                        cfg.validate()
                        restored = DiffNavTrainCfg(**json.loads(json.dumps(asdict(cfg))))
                        restored.validate()
                        self.assertEqual(restored.height_loss_mode, mode)
                        self.assertEqual(tuple(restored.height_target_range), (0.6, 2.4))
                        self.assertEqual((restored.optimizer, restored.weight_decay), (optimizer, decay))
                        self.assertEqual(restored.coef_height_boundary, 1.0)
        for mode in ("exp", "huber", "", None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "height_loss_mode"):
                DiffNavTrainCfg(height_loss_mode=mode).validate()
        for bounds in (
            (0.2, 2.5),
            (0.5, 4.0),
            (2.5, 0.5),
            (1.0, 1.0),
            (0.5,),
            (float("nan"), 2.5),
            (0.5, float("inf")),
        ):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "height_target_range"):
                DiffNavTrainCfg(height_target_range=bounds).validate()

    def test_invalid_optimizer_and_decay_cannot_enter_a_new_trial(self):
        for optimizer in ("sgd", "AdamW", "", None):
            with self.subTest(optimizer=optimizer), self.assertRaisesRegex(ValueError, "optimizer"):
                DiffNavTrainCfg(optimizer=optimizer).validate()
        for decay in (-0.01, float("nan"), float("inf")):
            with self.subTest(decay=decay), self.assertRaisesRegex(ValueError, "weight_decay"):
                DiffNavTrainCfg(weight_decay=decay).validate()

    def test_height_priority_is_explicit_bptt_only_and_serialized(self):
        for variant in ("step", "continuous", "terminal"):
            cfg = DiffNavTrainCfg(variant=variant, gradient_projection="height_priority")
            with self.subTest(variant=variant):
                cfg.validate()
                restored = DiffNavTrainCfg(**json.loads(json.dumps(asdict(cfg))))
                restored.validate()
                self.assertEqual(restored.gradient_projection, "height_priority")
        for algorithm in ("shac", "ahac"):
            with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValueError, "BPTT"):
                DiffNavTrainCfg(
                    variant="terminal", algorithm=algorithm, gradient_projection="height_priority"
                ).validate()
        with self.assertRaisesRegex(ValueError, "gradient_projection"):
            DiffNavTrainCfg(gradient_projection="pcgrad").validate()

    def test_plateau_patience_accepts_zero_and_two_but_requires_nonnegative_integers(self):
        for patience in (0, 2):
            cfg = DiffNavTrainCfg(lr_schedule="plateau", plateau_patience=patience)
            with self.subTest(patience=patience):
                cfg.validate()
                self.assertEqual(DiffNavTrainCfg(**asdict(cfg)).plateau_patience, patience)
        for patience in (-1, 0.5, True):
            with self.subTest(patience=patience), self.assertRaisesRegex(ValueError, "patience"):
                DiffNavTrainCfg(plateau_patience=patience).validate()

    def test_linear_remains_default_for_every_algorithm(self):
        for algorithm in ("bptt", "shac", "ahac"):
            with self.subTest(algorithm=algorithm):
                cfg = DiffNavTrainCfg(algorithm=algorithm, variant="terminal")
                cfg.validate()
                self.assertEqual(cfg.critic_value_transform, "linear")

    def test_ac_transform_is_serialized_and_restored_with_training_configuration(self):
        for algorithm in ("shac", "ahac"):
            for transform in ("linear", "softplus"):
                with self.subTest(algorithm=algorithm, transform=transform):
                    cfg = DiffNavTrainCfg(algorithm=algorithm, variant="terminal", critic_value_transform=transform)
                    cfg.validate()
                    payload = json.loads(json.dumps(asdict(cfg)))
                    self.assertEqual(payload["critic_value_transform"], transform)
                    restored = DiffNavTrainCfg(**payload)
                    restored.validate()
                    self.assertEqual(restored.critic_value_transform, transform)

    def test_rejects_unknown_transforms_and_inactive_bptt_option(self):
        for algorithm, transform in (("bptt", "softplus"), ("shac", "relu"), ("ahac", ""), ("shac", None)):
            with self.subTest(algorithm=algorithm, transform=transform):
                with self.assertRaisesRegex(ValueError, "critic_value_transform"):
                    DiffNavTrainCfg(
                        algorithm=algorithm, variant="terminal", critic_value_transform=transform
                    ).validate()


if __name__ == "__main__":
    unittest.main()


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


def test_optional_backend_cli_and_validation():
    assert DiffNavTrainCfg().geometry_backend == "curobo_v2"
    values = parse_train_args(["--geometry_backend", "curobo_v2"])
    assert values["geometry_backend"] == "curobo_v2"
    DiffNavTrainCfg(geometry_backend="curobo_v2").validate()
    with pytest.raises(ValueError, match="finite_difference"):
        DiffNavTrainCfg(geometry_backend="curobo_v2", approach_mode="relative").validate()


def test_all_geometry_consumers_can_select_the_backend_with_query_gradients():
    p = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], requires_grad=True)
    positions = torch.zeros(8, 3)
    sizes = torch.ones(8, 3)
    seen = []

    def backend(query, poses, extents, *, return_index):
        assert return_index and poses is positions and extents is sizes
        seen.append(query)
        return query[:, 0] * 0.5, torch.full((2,), 7, dtype=torch.long)

    distance = nearest_obstacle_signed_distance(p, p[:, 0] * 0 + 100.0, positions, sizes, dynamic_backend=backend)
    torch.testing.assert_close(
        torch.autograd.grad(distance.sum(), p)[0], torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )
    assert len(seen) == 1
