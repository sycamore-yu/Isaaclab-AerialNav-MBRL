"""SRU numerical provenance, complete memory lifecycle, and experiment identity."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from test_diffnav_contract import AnalyticEnv
from test_diffnav_runner_controls import _trainer

from aerialnav.diffnav.cli import parse_train_args
from aerialnav.diffnav.config import DiffNavTrainCfg
from aerialnav.diffnav.inference import load_diffnav_policy
from aerialnav.diffnav.model import DiffNavPolicy
from aerialnav.diffnav.runner import DiffNavTrainer
from aerialnav.diffnav.sru import LSTMSRUGateCell


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_cell_initialization_multistep_outputs_and_gradients_match_upstream():
    root = Path(os.environ.get("SRU_REFERENCE_ROOT", ""))
    source = root / "network/lstm_sru_gate.py"
    if not source.is_file():
        pytest.skip("Set SRU_REFERENCE_ROOT to the pinned upstream checkout for numerical parity")
    spec = importlib.util.spec_from_file_location("sru_gated_reference", source)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    torch.manual_seed(37)
    reference = reference_module.LSTMSRUGateCell(5, 7).double()
    torch.manual_seed(37)
    actual = LSTMSRUGateCell(5, 7).double()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    inputs = (torch.randn(4, 3, 5, dtype=torch.float64), *[torch.randn(3, 7, dtype=torch.float64) for _ in range(2)])
    outputs, gradients = [], []
    for model in (reference, actual):
        x, h, c = (value.clone().requires_grad_() for value in inputs)
        leaves = (x, h, c, *model.parameters())
        sequence = []
        for step in x:
            h, c = model(step, h, c)
            sequence.append(h)
        output = torch.stack(sequence)
        outputs.append((output, h, c))
        gradients.append(torch.autograd.grad(output.square().sum() + c.square().sum(), leaves))
    for expected, measured in zip(outputs[0] + gradients[0], outputs[1] + gradients[1], strict=True):
        assert torch.isfinite(measured).all()
        torch.testing.assert_close(measured, expected, rtol=0, atol=0)


def test_bptt_and_inference_reset_and_detach_both_states(tmp_path, monkeypatch):
    cfg = DiffNavTrainCfg(
        recurrent_model="sru_lstm_gated",
        variant="step",
        physics_dt=0.01,
        policy_dt=0.1,
        velocity_loss_mode="variant_default",
        timesteps=4,
        max_iterations=2,
        velocity_window_s=0.2,
    )
    env = AnalyticEnv()
    trainer = DiffNavTrainer(env, cfg, tmp_path)
    forward = trainer.model.forward_checkpointed
    calls = []

    def record(depth, features, hidden):
        result = forward(depth, features, hidden)
        calls.append((hidden.detach().clone(), hidden.grad_fn, result[1].detach().clone()))
        return result

    monkeypatch.setattr(trainer.model, "forward_checkpointed", record)
    result = trainer.run()
    assert result["recurrent_model"] == "sru_lstm_gated"
    assert result["completed_updates"] == 2 and result["nonfinite_grad_events"] == 0
    assert len(calls) == 8
    for index in (0, 4):
        assert calls[index][1] is None
    for index in (2, 4, 6):
        before = calls[index][0]
        assert before.shape == (2, 384) and torch.count_nonzero(before[0]) == 0
        torch.testing.assert_close(before[1], calls[index - 1][2][1], rtol=0, atol=0)
        assert all(torch.count_nonzero(value) > 0 for value in before[1].split(192))

    env.reset()
    policy = load_diffnav_policy(env, tmp_path / "checkpoint_final.pt")
    policy()
    before = policy.hidden.clone()
    inputs = []
    hook = policy.model.register_forward_pre_hook(lambda _, args: inputs.append(args[2].clone()))
    env.episode_id[0] += 1
    assert torch.isfinite(policy()).all()
    hook.remove()
    assert torch.count_nonzero(inputs[0][0]) == 0
    torch.testing.assert_close(inputs[0][1], before[1], rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ("shac", "ahac"))
def test_unsupported_algorithms_fail_in_config_and_queue_parser(algorithm):
    with pytest.raises(ValueError, match="requires --algorithm bptt"):
        DiffNavTrainCfg(recurrent_model="sru_lstm_gated", algorithm=algorithm).validate()
    with pytest.raises(ValueError, match="requires --algorithm bptt"):
        parse_train_args(["--recurrent_model", "sru_lstm_gated", "--algorithm", algorithm])


@pytest.mark.parametrize("model", ("sru_lstm", "lstm", ""))
def test_unknown_model_cannot_silently_select_gru(model):
    with pytest.raises(ValueError, match="recurrent_model"):
        DiffNavTrainCfg(recurrent_model=model).validate()
    with pytest.raises(ValueError, match="recurrent_model"):
        DiffNavPolicy(model)


@pytest.mark.parametrize(("source", "target"), (("gru", "sru_lstm_gated"), ("sru_lstm_gated", "gru")))
def test_stage_handoff_rejects_cross_model_checkpoint_before_loading_weights(tmp_path, source, target):
    previous = _trainer(tmp_path / "previous", recurrent_model=source)
    checkpoint = previous.log_dir / "checkpoint.pt"
    previous._checkpoint(checkpoint)
    current = _trainer(tmp_path / "current", stage="pretrain_s1", recurrent_model=target)
    before = {key: value.clone() for key, value in current.model.state_dict().items()}
    with pytest.raises(ValueError, match="cannot change recurrent_model"):
        current.load_stage_checkpoint(checkpoint)
    for name, value in current.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_legacy_gru_config_still_loads_as_gru(tmp_path):
    trainer = _trainer(tmp_path, recurrent_model="gru")
    checkpoint = tmp_path / "legacy.pt"
    trainer._checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    del payload["train_cfg"]["recurrent_model"]
    torch.save(payload, checkpoint)
    policy = load_diffnav_policy(AnalyticEnv(), checkpoint)
    assert policy.model.recurrent_model == "gru" and policy.hidden.shape == (2, 192)
    assert torch.isfinite(policy()).all()
