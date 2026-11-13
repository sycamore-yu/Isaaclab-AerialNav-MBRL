"""Completed physical transitions outlive autoreset; surrogate gradients do not."""

import pytest
import torch


def payload():
    return {
        "position": torch.ones(2, 3),
        "velocity": torch.zeros(2, 3),
        "episode_id": torch.tensor([3, 4]),
        "executed_action": torch.ones(2, 3),
        "terminated": torch.tensor([True, False]),
        "timeout": torch.tensor([False, True]),
        "step": torch.tensor([20, 40]),
        "time_s": torch.tensor([1.0, 2.0]),
        "critic_parameters": {"mass": torch.ones(2, 1)},
    }


def test_snapshot_owns_nested_storage_and_survives_autoreset():
    from isaaclab_navrl.tasks.navigation.transition import TransitionBatch

    source = payload()
    source["position"].requires_grad_()
    transition = TransitionBatch.capture(source)
    with torch.no_grad():
        source["position"].fill_(99.0)
    source["episode_id"].add_(1)
    source["critic_parameters"]["mass"].zero_()
    assert not transition["position"].requires_grad
    torch.testing.assert_close(transition["position"], torch.ones(2, 3))
    assert transition["episode_id"].tolist() == [3, 4]
    assert transition["critic_parameters"]["mass"].all()
    assert transition.done.all()
    with pytest.raises(TypeError):
        transition["position"] = torch.zeros(2, 3)
    with pytest.raises(TypeError):
        transition["critic_parameters"]["mass"] = torch.zeros(2, 1)


def test_reward_extension_does_not_mutate_the_published_snapshot():
    from isaaclab_navrl.tasks.navigation.transition import TransitionBatch

    original = TransitionBatch.capture(payload())
    returns = torch.ones(2)
    updated = original.with_fields(episode_return=returns)
    returns.zero_()
    assert "episode_return" not in original
    assert updated["episode_return"].all()
    assert updated["position"] is original["position"]


def test_consumer_is_one_shot_and_keeps_surrogate_gradients_live():
    from isaaclab_navrl.tasks.navigation.transition import TransitionPublisher
    from isaaclab_navrl.zhang.dynamics import align_state

    publisher = TransitionPublisher()
    source = payload()
    predicted = torch.full((2, 3), 2.0, requires_grad=True)
    losses = []

    def consume(transition):
        assert transition["episode_id"].tolist() == [3, 4]
        with torch.enable_grad():
            losses.append(align_state(predicted, transition["position"]).square().sum())

    publisher.set_consumer(consume)
    with pytest.raises(RuntimeError, match="pending"):
        publisher.set_consumer(consume)
    with torch.no_grad():
        publisher.publish(source)
    source["position"].fill_(99.0)
    source["episode_id"].add_(1)
    publisher.publish(source)
    assert len(losses) == 1
    losses[0].backward()
    torch.testing.assert_close(predicted.grad, torch.full_like(predicted, 2.0))


def test_failed_consumer_is_removed_before_it_runs():
    from isaaclab_navrl.tasks.navigation.transition import TransitionPublisher

    publisher = TransitionPublisher()

    def fail(transition):
        raise RuntimeError("test failure")

    publisher.set_consumer(fail)
    with pytest.raises(RuntimeError, match="test failure"):
        publisher.publish(payload())
    publisher.publish(payload())
