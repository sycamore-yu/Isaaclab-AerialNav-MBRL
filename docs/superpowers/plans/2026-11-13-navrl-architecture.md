# NavRL Architecture Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the four approved NavRL architecture improvements and commit local main.

**Architecture:** Extract task-state ownership, split observation sources, publish
owned pre-reset transitions and use one time/radius-aware clearance query.
Retain Isaac Lab, existing tensor kernels, controllers and differentiable training.

**Tech Stack:** Existing Python 3.12, PyTorch, Isaac Lab, Newton/PhysX and Warp.

**Spec:** `docs/NAVIGATION_ARCHITECTURE.md`; approved scope from the preceding conversation.

## Global Constraints

Preserve defaults, actor inputs, checkpoints, loss values/gradients, shared-world
obstacles, physics/policy clocks, same-step autoreset and 5-second arrival wait.
Do not touch active experiment worktrees, logs, public releases or unrelated files.
No new training campaign, external framework or environment/agent base hierarchy.

## Review Focus

Partial reset must leave other environments and shared obstacles unchanged.
Terminal snapshots must survive reset and retain the old episode's inputs.
Sensor reads must not advance noise/delay history twice.
Dynamic query offsets must not extrapolate an already aligned snapshot twice.
BPTT and timeout bootstrap gradients must remain live on the terminal surrogate.

## Tasks

- [x] 1. Add failing contract tests for NavigationTask, ObservationContext,
  TransitionBatch/consumer, and time/radius-aware clearance. Run them before implementation.
- [x] 2. Implement `tasks/navigation/task.py` and `context.py`; reuse current
  termination functions. Integrate task ownership and observation namespaces in
  `navrl_env.py`, Torch/Warp paths, inference, recording and trainer consumers.
- [x] 3. Implement `tasks/navigation/transition.py`; assemble common and Zhang
  terminal fields before publishing. Replace the loss-specific callback attribute
  with a one-shot consumer; update analytical test environments to the same API.
- [x] 4. Implement `query_clearance` in `geometry.py`; migrate environment,
  surrogate lookahead, collision endpoints, goal checks and recording callers.
  Keep the lower-level signed-surface kernel and its existing numerical tests.
- [x] 5. Run targeted tests, full pytest and lint; run short real Newton Torch/Warp
  checks and a finite BPTT smoke. Report existing failures separately.
- [x] 6. Review changed interfaces, write evidence, stage only scoped files and
  commit to local `main`. Verify the commit and preserve unrelated untracked files.

## Execution notes

Base: `625afcef08706b8d76af711ad087e051d48ea4a6`.
The user explicitly authorized main integration. No production process using
this checkout was found in the initial training-process check. Work proceeds
in this checkout; other experimental worktrees remain untouched.
The connector's advertised agent/plan actions returned “Tool not found”; execution
and review therefore stay in this session. No worker was launched.

Completed evidence is in `docs/results/navrl-architecture-20261113.md`:
142 final related tests pass; both loss-timing modes match the original short
trainer run bitwise; real Newton Torch/Warp checks and depth/PointNet BPTT runs
complete. The full sweep exposed three new zero-time geometry dependency
regressions, now fixed and retested. Its remaining 29 failing assertions match
the archived original commit. Three unrelated repository-wide import-order lint
issues remain unchanged; all scoped Python files pass lint and formatting.

This plan and its evidence are committed with the implementation on local main.
The actual commit identity is recorded by Git; no remote publication is performed.
