# Apply a training stage

Use the environment configuration to apply a stage before selecting physics and
rebuilding the scene:

```python
from aerialnav.tasks.navigation.config.hummingbird.env_cfg import DiffNavEnvCfg

cfg = DiffNavEnvCfg()
cfg.apply_training_stage("pretrain_s1")
cfg.scene.num_envs = 64
cfg.configure_scene()
```

For an explicit experiment override, pass the complete stage settings:

```python
from aerialnav.training import STAGE_CONFIG

stage = STAGE_CONFIG["pretrain_s1"] | {"dynamic_obstacles": 0}
cfg.apply_training_stage("pretrain_s1", stage_config=stage)
cfg.configure_scene()
```

`apply_training_stage()` sets the stage name, override metadata, obstacle counts,
randomization and episode duration. It returns the selected settings for run metadata
and learning-rate selection. A later call without an override clears the previous
`training_stage_config`. Copy a stage table entry before modifying it.

`configure_scene()` still accepts manual field assignments from existing callers.
It rebuilds scene configuration and applies stage duration without overwriting those
counts. The training and evaluation entry points retain their physics selection,
seeds, report windows and explicit overrides.

## Architecture decision

The existing environment configuration module owns stage application. Its interface
serves the training and evaluation adapter at the same seam; rules now have locality
rather than being repeated in three entry points. This provides depth and leverage
without introducing a separate configuration framework.

Consumers: `scripts/train_diffnav.py`, `scripts/train_ppo.py`, `scripts/evaluate.py`.

The release PPO entry point retains its 300-static / zero-dynamic / no-randomization
baseline. DiffNav retains the five-stage table and optional dynamic-count override.

## Validation scope

`tests/test_stage_configuration.py` exercises stage switching, clearing an override,
zero dynamic obstacles, static PPO overrides, manual scene overrides and unknown-stage
rejection. Existing curriculum and episode-accounting tests protect reporting semantics.
No task IDs, checkpoint fields, TensorBoard tags or algorithm defaults change.

The implementation follows the existing `configclass` ownership, snake_case method
names and Google-style public method documentation described in the
[Isaac Lab contribution guide](https://github.com/isaac-sim/IsaacLab/blob/159af106ce06165d7147683695b20d2e3ed63257/docs/source/refs/contributing.rst).

### Verified on 2026-11-02

With the existing Python 3.12 environment, `PYTHONPATH=src` and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, 46 distinct focused tests passed:

```sh
python -m pytest -q tests/test_stage_configuration.py tests/test_training_curriculum.py tests/test_navigation_contract.py tests/test_termination.py tests/test_evaluation_cli.py
```

The initial new test used three dynamic obstacles, which the existing geometry
contract rejects. Its fixture was corrected to eight, then all three stage tests
passed on both checkouts; the other successful test results remain applicable.
Ruff checks and `git diff --check` passed. TorchScript deprecation warnings originate
in the installed dependency. No training or simulation rollout was run.
