"""Task ownership must preserve the existing episode and partial-reset rules."""

import torch


def make_task():
    from isaaclab_navrl.tasks.navigation.task import NavigationTask

    task = NavigationTask(2, "cpu")
    task.reset(
        torch.arange(2),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        torch.tensor([[10.0, 0.0, 2.0], [20.0, 0.0, 3.0]]),
        torch.tensor([[2.0], [4.0]]),
    )
    return task


def terms(reached=(False, False), collision=(False, False), bounds=(False, False), invalid=(False, False)):
    return {
        name: torch.tensor(values)
        for name, values in (("reached", reached), ("collision", collision), ("bounds", bounds), ("invalid", invalid))
    }


def advance(task, events, seconds, terminated=None):
    return task.advance(
        events,
        torch.full((2,), int(seconds * 20)),
        policy_dt=0.05,
        max_episode_length=1000,
        wait_before_truncate=5.0,
        terminated=torch.zeros(2, dtype=torch.bool) if terminated is None else terminated,
    )


def test_partial_reset_has_one_owner_and_preserves_other_rows():
    task = make_task()
    task.record_substep(terms((True, False), (False, True)), torch.tensor([1.0, 2.0]))
    task.previous_velocity.fill_(3.0)
    task.episode_return.fill_(7.0)
    pointers = {name: value.data_ptr() for name, value in vars(task).items() if isinstance(value, torch.Tensor)}
    other = {name: value[1].clone() for name, value in vars(task).items() if isinstance(value, torch.Tensor)}
    task.reset(
        torch.tensor([0]), torch.tensor([[1.0, 0.0, 2.0]]), torch.tensor([[5.0, 0.0, 1.0]]), torch.tensor([[3.0]])
    )
    for name, value in vars(task).items():
        if isinstance(value, torch.Tensor):
            assert value.data_ptr() == pointers[name]
            torch.testing.assert_close(value[1], other[name])
    torch.testing.assert_close(task.target_dir[0], torch.tensor([4.0, 0.0, -1.0]))
    torch.testing.assert_close(task.height_range[0], torch.tensor([1.0, 2.0]))
    assert task.episode_id.tolist() == [1, 0]
    assert not task.arrived[0] and task.first_event_code[0] == 0
    assert task.previous_velocity[0].count_nonzero() == 0 and task.episode_return[0] == 0


def test_wait_uses_first_entry_and_requires_current_goal_membership():
    task = make_task()
    advance(task, terms((True, False)), 1.0)
    assert not advance(task, terms((True, False)), 6.0)["timeout"][0]
    assert not advance(task, terms(), 6.05)["timeout"][0]
    result = advance(task, terms((True, False)), 7.0)
    assert result["arrival_wait_timeout"].tolist() == [True, False]
    assert result["terminal_event_code"].tolist() == [1, 0]
    assert task.arrive_time[0] == 1.0 and task.first_reach_step[0] == 20


def test_nonterminal_failure_is_latched_separately_from_first_arrival():
    task = make_task()
    task.record_substep(terms((True, False)), torch.tensor([1.0, 1.0]))
    advance(task, terms((True, False)), 1.0)
    advance(task, terms(collision=(True, False)), 2.0)
    result = advance(task, terms((True, False)), 7.0)
    assert task.first_event_code[0] == 1
    assert result["terminal_event_code"][0] == 2


def test_substep_priority_and_timeout_first_event_match_existing_rules():
    task = make_task()
    task.record_substep(terms((True, True), (True, True), (True, True), (False, True)), torch.tensor([0.1, 0.1]))
    assert task.first_event_code.tolist() == [2, 4]
    task.reset(torch.arange(2), torch.zeros(2, 3), torch.ones(2, 3), torch.ones(2, 1))
    task.finish_first_events(torch.tensor([True, False]), torch.tensor([5.0, 5.0]))
    assert task.first_event_code.tolist() == [5, 0]
    assert task.first_event_time[0] == 5.0
