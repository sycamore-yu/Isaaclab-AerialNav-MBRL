"""Pure configuration checks; no simulator or training is performed."""

import json
import unittest
from dataclasses import asdict

from isaaclab_navrl.zhang.config import ZhangTrainCfg


class ZhangTrainConfigTests(unittest.TestCase):
    def test_default_velocity_mode_preserves_each_variant(self):
        for variant, expected in (("step", "per_step"), ("continuous", "window"), ("dcol", "window")):
            with self.subTest(variant=variant):
                cfg = ZhangTrainCfg(variant=variant)
                cfg.validate()
                self.assertEqual(cfg.velocity_loss_mode, "variant_default")
                self.assertEqual(cfg.resolved_velocity_loss_mode, expected)

    def test_explicit_velocity_modes_are_independent_of_dcol_algorithm_and_preserve_critic_window(self):
        for algorithm in ("bptt", "shac", "ahac"):
            for mode in ("per_step", "truncated_window"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    cfg = ZhangTrainCfg(variant="dcol", algorithm=algorithm, velocity_loss_mode=mode,
                                        velocity_window_s=2.0)
                    cfg.validate()
                    self.assertEqual(cfg.resolved_velocity_loss_mode, mode)
                    self.assertEqual(cfg.velocity_window_steps, 20)
                    payload = json.loads(json.dumps(asdict(cfg)))
                    self.assertEqual(payload["velocity_loss_mode"], mode)
                    restored = ZhangTrainCfg(**payload)
                    restored.validate()
                    self.assertEqual(restored.resolved_velocity_loss_mode, mode)
                    self.assertEqual(restored.velocity_window_steps, 20)
        for mode in ("per_step", "truncated_window"):
            ZhangTrainCfg(variant="continuous", velocity_loss_mode=mode).validate()

    def test_invalid_velocity_modes_fail_and_explicit_modes_keep_window_validation(self):
        for value in ("window", "step", "truncated", "TRUNCATED_WINDOW", "", None, True):
            with self.subTest(value=value):
                cfg = ZhangTrainCfg(velocity_loss_mode=value)
                with self.assertRaisesRegex(ValueError, "velocity_loss_mode"):
                    cfg.validate()
                with self.assertRaisesRegex(ValueError, "velocity_loss_mode"):
                    _ = cfg.resolved_velocity_loss_mode
        for mode in ("per_step", "truncated_window"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "velocity_window_s"):
                ZhangTrainCfg(variant="dcol", velocity_loss_mode=mode, velocity_window_s=0.15).validate()
        with self.assertRaisesRegex(ValueError, "variant"):
            ZhangTrainCfg(variant="truncated_window").validate()

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
        for algorithm, transform in (("bptt", "softplus"), ("shac", "relu"),
                                     ("ahac", ""), ("shac", None)):
            with self.subTest(algorithm=algorithm, transform=transform):
                with self.assertRaisesRegex(ValueError, "critic_value_transform"):
                    ZhangTrainCfg(algorithm=algorithm, variant="dcol", critic_value_transform=transform).validate()


if __name__ == "__main__":
    unittest.main()
