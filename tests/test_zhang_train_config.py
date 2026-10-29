"""Pure configuration checks; no simulator or training is performed."""

import json
import unittest
from dataclasses import asdict

from isaaclab_navrl.zhang.config import ZhangTrainCfg


class ZhangTrainConfigTests(unittest.TestCase):
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
