"""Exercise the real PPO entry point with a small runner double, without simulation."""

import contextlib
import importlib.util
import io
import json
import random
import signal
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


def load_trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("_aerialnav_graceful_stop_entry", path)
    module = importlib.util.module_from_spec(spec)
    app = types.ModuleType("isaaclab.app")
    app.add_launcher_args = lambda parser: parser.add_argument("--device", default="cpu")
    app.launch_simulation = None
    with patch.dict(sys.modules, {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.app": app}):
        spec.loader.exec_module(module)
    return module


trainer = load_trainer()
STAGES = {
    "pretrain_s0": {"static_obstacles": 300, "dynamic_obstacles": 0, "randomization": False},
    "pretrain_s1": {"static_obstacles": 300, "dynamic_obstacles": 60, "randomization": False},
}


@dataclass
class SimToReal:
    enabled: bool = False


class EnvironmentConfig:
    def __init__(self):
        self.scene = types.SimpleNamespace(num_envs=64)
        self.sim = types.SimpleNamespace(dt=0.016, device="cpu")
        self.sim_to_real = SimToReal()
        self.decimation = 1
        self.episode_length_s = 35.2

    def configure_scene(self):
        pass


class AgentConfig:
    num_steps_per_env = 32

    def __init__(self, save_interval):
        self.algorithm = types.SimpleNamespace(learning_rate=0.0005)
        self.save_interval = save_interval

    def to_dict(self):
        return {"algorithm": {"learning_rate": self.algorithm.learning_rate}}


class Curriculum:
    def __init__(self, harness):
        self.harness = harness
        self.stage_passed = False
        self.acceptance_passed = False

    def state_dict(self):
        return {"stage": self.harness.cfg.training_stage, "stage_passed": self.stage_passed}

    def pop_report(self, update):
        self.harness.events.append(("report", update))
        return {
            "stage": self.harness.cfg.training_stage,
            "dynamic_obstacles": self.harness.cfg.num_dynamic_obstacles,
            "episodes": 64,
            "success_rate": 0.8 if self.stage_passed else 0.0,
            "new_best": False,
        }


class Writer:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def add_scalar(self, *_args):
        pass

    def flush(self):
        self.events.append(("report_flushed",))

    def close(self):
        self.closed = True


class Runner:
    def __init__(self, harness):
        self.harness = harness
        self.logger = types.SimpleNamespace(writer=Writer(harness.events))
        self.alg = types.SimpleNamespace(
            learning_rate=0.0005,
            optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.0005, "params": [object()]}]),
        )
        self.current_learning_iteration = 0
        self.optimizer_updates = 0
        self.learn_calls = 0
        self.in_learn = False

    def add_git_repo_to_log(self, _path):
        pass

    def load(self, path, *, strict, map_location):
        assert strict and map_location == "cpu"
        payload = json.loads(Path(path).read_text())
        self.optimizer_updates = payload["optimizer_updates"]
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"] = 0.123
        return payload["infos"]

    def learn(self, *, num_learning_iterations, init_at_random_ep_len):
        assert num_learning_iterations == 1 and init_at_random_ep_len is False
        self.in_learn = True
        self.learn_calls += 1
        self.harness.events.append(("learn_started", self.learn_calls))
        if self.learn_calls == self.harness.signal_at_update:
            signal.raise_signal(self.harness.stop_signal)
            self.harness.events.append(("signal_returned", self.learn_calls))
            assert not (self.harness.log_dir / "checkpoint_stopped.pt").exists()
        if self.harness.fail_learn:
            raise RuntimeError("incomplete optimizer update")
        self.optimizer_updates += 1
        self.in_learn = False
        self.harness.curriculum.stage_passed = self.harness.pass_stage
        self.harness.events.append(("learn_finished", self.learn_calls))

    def save(self, path, infos):
        assert not self.in_learn, "A partially completed PPO update must never be saved"
        path = Path(path)
        assert path.name.endswith(".tmp.pt"), "Only a complete temporary checkpoint may replace the destination"
        self.harness.events.append(("save", path.name, infos["completed_updates"]))
        if self.harness.fail_second_periodic_save and self.learn_calls == 2:
            path.write_text("partial checkpoint")
            raise RuntimeError("checkpoint write failed")
        if self.harness.signal_during_final_save and path.name == "checkpoint_final.tmp.pt":
            signal.raise_signal(self.harness.stop_signal)
        path.write_text(json.dumps({"infos": infos, "optimizer_updates": self.optimizer_updates}))


class Harness:
    def __init__(
        self,
        log_dir,
        *,
        signal_at_update=None,
        stop_signal=signal.SIGTERM,
        fail_learn=False,
        pass_stage=False,
        fail_second_periodic_save=False,
        signal_during_final_save=False,
    ):
        self.log_dir = log_dir
        self.signal_at_update = signal_at_update
        self.stop_signal = stop_signal
        self.fail_learn = fail_learn
        self.pass_stage = pass_stage
        self.fail_second_periodic_save = fail_second_periodic_save
        self.signal_during_final_save = signal_during_final_save
        self.events = []
        self.cfg = EnvironmentConfig()
        self.selected_physics = None
        self.factory_calls = 0
        self.curriculum = Curriculum(self)
        self.env_closed = False
        self.runner = None
        self.env = types.SimpleNamespace(
            cfg=self.cfg,
            curriculum=self.curriculum,
            nominal_dynamics={"total_mass_kg": 0.716},
            nonfinite_events=0,
            close=self.close_env,
        )

    def close_env(self):
        self.env_closed = True

    def make_runner(self, *_args, **_kwargs):
        self.runner = Runner(self)
        return self.runner

    def run(
        self,
        *,
        max_iterations=4,
        stage="pretrain_s0",
        checkpoint=None,
        stop_after_acceptance=False,
        physics="isaacsim_physx",
        frontend="torch",
    ):
        tokens = [
            "train_ppo.py",
            "--log_dir",
            str(self.log_dir),
            "--num_envs",
            "64",
            "--max_iterations",
            str(max_iterations),
            "--stage",
            stage,
            f"physics={physics}",
            "--frontend",
            frontend,
        ]
        if checkpoint is not None:
            tokens += ["--checkpoint", str(checkpoint)]
        if stop_after_acceptance:
            tokens += ["--stop_after_acceptance"]

        def resolve_presets(cfg, *, selected):
            assert cfg is self.cfg and selected == (physics,)
            self.selected_physics = selected[0]

        def launch(cfg, args):
            assert cfg is self.cfg and self.selected_physics == physics
            # A non-None launcher override would discard the resolved task preset.
            assert getattr(args, "physics", None) is None
            return contextlib.nullcontext()

        def create_isaaclab_env(task, cfg, args, *, convert_marl_to_single_agent):
            assert task == "Isaac-AerialNav-Static-PPO-v0"
            assert cfg is self.cfg and self.selected_physics == physics
            assert args.physics_preset == physics and args.frontend == frontend
            assert convert_marl_to_single_agent is False
            self.factory_calls += 1
            return types.SimpleNamespace(unwrapped=self.env)

        imports = {
            "isaaclab_tasks.utils.hydra": types.SimpleNamespace(resolve_presets=resolve_presets),
            "isaaclab_rl.entrypoints.common": types.SimpleNamespace(create_isaaclab_env=create_isaaclab_env),
            "numpy": types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _seed: None)),
            "torch": types.SimpleNamespace(
                manual_seed=lambda _seed: None,
                load=lambda path, **_kwargs: json.loads(Path(path).read_text()),
                isfinite=lambda _value: types.SimpleNamespace(all=lambda: True),
            ),
            "isaaclab_rl.rsl_rl.utils": types.SimpleNamespace(handle_deprecated_rsl_rl_cfg=lambda cfg, version: cfg),
            "isaaclab_rl.rsl_rl": types.SimpleNamespace(RslRlVecEnvWrapper=lambda env, **_kwargs: env),
            "aerialnav.tasks": types.SimpleNamespace(),
            "aerialnav.agents.continuous_runner": types.SimpleNamespace(ContinuousNavigationRunner=self.make_runner),
            "aerialnav.agents.rsl_rl_cfg": types.SimpleNamespace(
                NavigationPpoRunnerCfg=lambda: AgentConfig(1 if self.fail_second_periodic_save else 100)
            ),
            "aerialnav.tasks.navigation.config.hummingbird.env_cfg": types.SimpleNamespace(
                NavigationEnvCfg=lambda: self.cfg
            ),
            "aerialnav.tasks.navigation.curriculum": types.SimpleNamespace(STAGE_CONFIG=STAGES),
        }
        original_import = __import__

        def intercept(name, *args, **kwargs):
            return imports[name] if name in imports else original_import(name, *args, **kwargs)

        # An absent/broken handler should fail an assertion, not kill the test process.
        unexpected_signals = []

        def fallback(signum, _frame):
            unexpected_signals.append(signum)

        previous_handlers = {signum: signal.signal(signum, fallback) for signum in (signal.SIGINT, signal.SIGTERM)}
        rng_state = random.getstate()
        try:
            with patch("builtins.__import__", side_effect=intercept), patch.object(sys, "argv", tokens):
                with patch.object(trainer, "launch_simulation", side_effect=launch):
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.main()
        finally:
            self.handlers_restored = all(signal.getsignal(signum) is fallback for signum in previous_handlers)
            self.unexpected_signals = unexpected_signals
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            random.setstate(rng_state)

    def result(self):
        return json.loads((self.log_dir / "training_result.json").read_text())


class PpoGracefulStopTests(unittest.TestCase):
    def assert_closed(self, harness):
        self.assertEqual(harness.factory_calls, 1)
        self.assertTrue(harness.env_closed)
        self.assertTrue(harness.runner.logger.writer.closed)
        self.assertTrue(harness.handlers_restored)
        self.assertEqual(harness.unexpected_signals, [])
        self.assertEqual(list(harness.log_dir.glob("*.tmp.pt")), [])
        run_config = json.loads((harness.log_dir / "run_config.json").read_text())
        for metadata in (harness.result(), run_config):
            self.assertEqual(metadata["frontend"], "torch")
            self.assertEqual(metadata["environment_class"], "types.SimpleNamespace")

    def test_sigint_and_sigterm_finish_the_current_update_and_report_before_saving(self):
        with tempfile.TemporaryDirectory(prefix="aerialnav_stop_") as temporary:
            for signum in (signal.SIGINT, signal.SIGTERM):
                with self.subTest(signum=signum):
                    harness = Harness(Path(temporary) / str(signum), signal_at_update=2, stop_signal=signum)
                    harness.run()
                    result = harness.result()
                    checkpoint = harness.log_dir / "checkpoint_stopped.pt"
                    self.assertEqual(result["status"], "stopped")
                    self.assertEqual(result["stop_requested"], str(signum))
                    self.assertEqual(result["checkpoint"], str(checkpoint.resolve()))
                    self.assertEqual(
                        (result["completed_updates"], result["stage_updates"], result["environment_steps"]),
                        (2, 2, 2 * 32 * 64),
                    )
                    self.assertFalse(result["stage_passed"])
                    self.assertFalse(result["acceptance_passed"])
                    payload = json.loads(checkpoint.read_text())
                    self.assertEqual(payload["optimizer_updates"], 2)
                    self.assertEqual(payload["infos"]["completed_updates"], 2)
                    self.assertEqual(payload["infos"]["stop_requested"], str(signum))
                    self.assertEqual(payload["infos"]["frontend"], "torch")
                    self.assertEqual(payload["infos"]["environment_class"], "types.SimpleNamespace")
                    self.assertEqual(harness.runner.learn_calls, 2)
                    self.assertLess(harness.events.index(("learn_finished", 2)), harness.events.index(("report", 2)))
                    self.assertLess(
                        harness.events.index(("report", 2)),
                        harness.events.index(("save", "checkpoint_stopped.tmp.pt", 2)),
                    )
                    self.assertFalse((harness.log_dir / "checkpoint_final.pt").exists())
                    self.assert_closed(harness)

    def test_stop_takes_precedence_when_a_natural_report_also_passes_the_stage(self):
        with tempfile.TemporaryDirectory(prefix="aerialnav_stop_pass_") as temporary:
            harness = Harness(Path(temporary), signal_at_update=1, pass_stage=True)
            harness.run(stop_after_acceptance=True)
            result = harness.result()
            self.assertTrue(result["curriculum"]["stage_passed"])
            self.assertTrue(result["stage_passed"])
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(Path(result["checkpoint"]).name, "checkpoint_stopped.pt")
            self.assertFalse(result["acceptance_passed"])
            self.assert_closed(harness)

    def test_a_failed_partial_update_is_not_saved_even_after_a_stop_request(self):
        with tempfile.TemporaryDirectory(prefix="aerialnav_stop_failure_") as temporary:
            harness = Harness(Path(temporary), signal_at_update=1, fail_learn=True)
            with self.assertRaisesRegex(RuntimeError, "incomplete optimizer update"):
                harness.run()
            result = harness.result()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_updates"], 0)
            self.assertEqual(result["stop_requested"], str(signal.SIGTERM))
            self.assertEqual(list(harness.log_dir.glob("*.pt")), [])
            self.assert_closed(harness)

    def test_static_update_limit_and_rejection_of_dynamic_stages(self):
        with tempfile.TemporaryDirectory(prefix="aerialnav_complete_") as temporary:
            first = Harness(Path(temporary) / "s0")
            first.run(max_iterations=2, physics="newton_mjwarp")
            result = first.result()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["physics"], "newton_mjwarp")
            self.assertEqual(result["stage"], "pretrain_s0")
            self.assertEqual((result["completed_updates"], result["stage_updates"], result["environment_steps"]), (2, 2, 2 * 32 * 64))
            payload = json.loads(Path(result["checkpoint"]).read_text())
            self.assertEqual(payload["infos"]["frontend"], "torch")
            self.assertIsNone(result["stop_requested"])
            self.assert_closed(first)
            second = Harness(Path(temporary) / "unsupported_stage")
            with self.assertRaises(SystemExit):
                second.run(max_iterations=1, stage="pretrain_s1")

    def test_a_partial_checkpoint_write_preserves_the_preceding_complete_file(self):
        with tempfile.TemporaryDirectory(prefix="aerialnav_atomic_save_") as temporary:
            harness = Harness(Path(temporary), fail_second_periodic_save=True)
            with self.assertRaisesRegex(RuntimeError, "checkpoint write failed"):
                harness.run()
            checkpoint = json.loads((harness.log_dir / "checkpoint_latest.pt").read_text())
            self.assertEqual(checkpoint["infos"]["completed_updates"], 1)
            self.assertEqual(checkpoint["optimizer_updates"], 1)
            self.assertEqual(harness.result()["completed_updates"], 2)
            self.assertEqual(harness.result()["status"], "failed")
            self.assert_closed(harness)

    def test_stop_during_the_normal_final_save_still_returns_a_stopped_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="aerialnav_late_stop_") as temporary:
            harness = Harness(Path(temporary), signal_during_final_save=True)
            harness.run(max_iterations=1)
            result = harness.result()
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(Path(result["checkpoint"]).name, "checkpoint_stopped.pt")
            self.assertEqual(json.loads(Path(result["checkpoint"]).read_text())["optimizer_updates"], 1)
            self.assert_closed(harness)


if __name__ == "__main__":
    unittest.main()
