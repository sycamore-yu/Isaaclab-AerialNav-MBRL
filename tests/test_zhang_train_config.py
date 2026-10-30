"""Pure configuration checks; no simulator or training is performed."""

import json
import unittest
from dataclasses import asdict

from isaaclab_navrl.zhang.config import ZhangTrainCfg


class ZhangTrainConfigTests(unittest.TestCase):
    def test_restart_defaults_and_ahac_expansion_limit_are_explicit(self):
        cfg = ZhangTrainCfg()
        cfg.validate()
        self.assertEqual((cfg.optimizer, cfg.weight_decay), ("adamw", 0.01))
        self.assertEqual(cfg.height_loss_mode, "navrl_range")
        self.assertEqual(cfg.height_target_range, (0.5, 2.5))
        self.assertEqual(cfg.gradient_projection, "none")
        self.assertEqual(cfg.plateau_patience, 2)
        self.assertEqual((cfg.ahac_min_horizon, cfg.ahac_initial_horizon, cfg.ahac_max_horizon), (16, 32, 300))
        # A permitted initial H above the previous 150 cap is not silently
        # limited by the independent BPTT timesteps setting.
        ZhangTrainCfg(algorithm="ahac", variant="dcol", ahac_initial_horizon=200).validate()

    def test_default_velocity_mode_uses_terminal_completion_for_continuous_and_dcol(self):
        for variant, expected in (
            ("step", "per_step"),
            ("continuous", "truncated_window"),
            ("dcol", "truncated_window"),
        ):
            with self.subTest(variant=variant):
                cfg = ZhangTrainCfg(variant=variant, velocity_loss_mode="variant_default")
                cfg.validate()
                self.assertEqual(cfg.velocity_loss_mode, "variant_default")
                self.assertEqual(cfg.resolved_velocity_loss_mode, expected)

    def test_explicit_velocity_modes_are_independent_of_dcol_algorithm_and_preserve_critic_window(self):
        for algorithm in ("bptt", "shac", "ahac"):
            for mode in ("per_step", "truncated_window", "full_window"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    cfg = ZhangTrainCfg(
                        variant="dcol", algorithm=algorithm, velocity_loss_mode=mode, velocity_window_s=2.0
                    )
                    cfg.validate()
                    self.assertEqual(cfg.resolved_velocity_loss_mode, "window" if mode == "full_window" else mode)
                    self.assertEqual(cfg.velocity_window_steps, 40)
                    payload = json.loads(json.dumps(asdict(cfg)))
                    self.assertEqual(payload["velocity_loss_mode"], mode)
                    restored = ZhangTrainCfg(**payload)
                    restored.validate()
                    self.assertEqual(restored.resolved_velocity_loss_mode, "window" if mode == "full_window" else mode)
                    self.assertEqual(restored.velocity_window_steps, 40)
        for mode in ("per_step", "truncated_window", "full_window"):
            ZhangTrainCfg(variant="continuous", velocity_loss_mode=mode).validate()

    def test_invalid_velocity_modes_fail_and_explicit_modes_keep_window_validation(self):
        for value in ("window", "step", "truncated", "TRUNCATED_WINDOW", "", True):
            with self.subTest(value=value):
                cfg = ZhangTrainCfg(velocity_loss_mode=value)
                with self.assertRaisesRegex(ValueError, "velocity_loss_mode"):
                    cfg.validate()
                with self.assertRaisesRegex(ValueError, "velocity_loss_mode"):
                    _ = cfg.resolved_velocity_loss_mode
        for mode in ("per_step", "truncated_window", "full_window"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "velocity_window_s"):
                ZhangTrainCfg(variant="dcol", velocity_loss_mode=mode, velocity_window_s=0.125).validate()
        with self.assertRaisesRegex(ValueError, "variant"):
            ZhangTrainCfg(variant="truncated_window").validate()

    def test_height_modes_and_target_range_roundtrip_with_explicit_optimizer_controls(self):
        for mode in ("target_exp", "target_huber", "boundary_huber"):
            for optimizer in ("adam", "adamw"):
                for decay in (0.0, 0.01):
                    with self.subTest(mode=mode, optimizer=optimizer, decay=decay):
                        cfg = ZhangTrainCfg(
                            height_loss_mode=mode,
                            height_target_range=(0.6, 2.4),
                            optimizer=optimizer,
                            weight_decay=decay,
                        )
                        cfg.validate()
                        restored = ZhangTrainCfg(**json.loads(json.dumps(asdict(cfg))))
                        restored.validate()
                        self.assertEqual(restored.height_loss_mode, mode)
                        self.assertEqual(tuple(restored.height_target_range), (0.6, 2.4))
                        self.assertEqual((restored.optimizer, restored.weight_decay), (optimizer, decay))
                        self.assertEqual(restored.coef_height_boundary, 1.0)
        for mode in ("exp", "huber", "", None):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "height_loss_mode"):
                ZhangTrainCfg(height_loss_mode=mode).validate()
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
                ZhangTrainCfg(height_target_range=bounds).validate()

    def test_invalid_optimizer_and_decay_cannot_enter_a_new_trial(self):
        for optimizer in ("sgd", "AdamW", "", None):
            with self.subTest(optimizer=optimizer), self.assertRaisesRegex(ValueError, "optimizer"):
                ZhangTrainCfg(optimizer=optimizer).validate()
        for decay in (-0.01, float("nan"), float("inf")):
            with self.subTest(decay=decay), self.assertRaisesRegex(ValueError, "weight_decay"):
                ZhangTrainCfg(weight_decay=decay).validate()

    def test_height_priority_is_explicit_bptt_only_and_serialized(self):
        for variant in ("step", "continuous", "dcol"):
            cfg = ZhangTrainCfg(variant=variant, gradient_projection="height_priority")
            with self.subTest(variant=variant):
                cfg.validate()
                restored = ZhangTrainCfg(**json.loads(json.dumps(asdict(cfg))))
                restored.validate()
                self.assertEqual(restored.gradient_projection, "height_priority")
        for algorithm in ("shac", "ahac"):
            with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValueError, "BPTT"):
                ZhangTrainCfg(variant="dcol", algorithm=algorithm, gradient_projection="height_priority").validate()
        with self.assertRaisesRegex(ValueError, "gradient_projection"):
            ZhangTrainCfg(gradient_projection="pcgrad").validate()

    def test_plateau_patience_accepts_zero_and_two_but_requires_nonnegative_integers(self):
        for patience in (0, 2):
            cfg = ZhangTrainCfg(lr_schedule="plateau", plateau_patience=patience)
            with self.subTest(patience=patience):
                cfg.validate()
                self.assertEqual(ZhangTrainCfg(**asdict(cfg)).plateau_patience, patience)
        for patience in (-1, 0.5, True):
            with self.subTest(patience=patience), self.assertRaisesRegex(ValueError, "patience"):
                ZhangTrainCfg(plateau_patience=patience).validate()

    def test_linear_remains_default_for_every_algorithm(self):
        for algorithm in ("bptt", "shac", "ahac"):
            with self.subTest(algorithm=algorithm):
                cfg = ZhangTrainCfg(algorithm=algorithm, variant="dcol")
                cfg.validate()
                self.assertEqual(cfg.critic_value_transform, "linear")

    def test_ac_transform_is_serialized_and_restored_with_training_configuration(self):
        for algorithm in ("shac", "ahac"):
            for transform in ("linear", "softplus"):
                with self.subTest(algorithm=algorithm, transform=transform):
                    cfg = ZhangTrainCfg(algorithm=algorithm, variant="dcol", critic_value_transform=transform)
                    cfg.validate()
                    payload = json.loads(json.dumps(asdict(cfg)))
                    self.assertEqual(payload["critic_value_transform"], transform)
                    restored = ZhangTrainCfg(**payload)
                    restored.validate()
                    self.assertEqual(restored.critic_value_transform, transform)

    def test_rejects_unknown_transforms_and_inactive_bptt_option(self):
        for algorithm, transform in (("bptt", "softplus"), ("shac", "relu"), ("ahac", ""), ("shac", None)):
            with self.subTest(algorithm=algorithm, transform=transform):
                with self.assertRaisesRegex(ValueError, "critic_value_transform"):
                    ZhangTrainCfg(algorithm=algorithm, variant="dcol", critic_value_transform=transform).validate()


if __name__ == "__main__":
    unittest.main()
