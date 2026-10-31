# Installation

1. Use Linux x86-64 with an NVIDIA GPU, working driver and Python 3.12.
2. Install `uv` and run `uv sync --frozen` from the repository root.
3. Use the local environment with `PYTHONPATH=src .venv/bin/python`.
4. Run the CPU suite with `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not integration' tests`.
5. Run the simulator smoke check shown in the top-level README before training.

Before the first PhysX/Kit launch, review and accept the NVIDIA Omniverse EULA yourself. If previously accepted, your environment may require `OMNI_KIT_ACCEPT_EULA=YES`; this repository does not accept legal terms automatically.

The pinned software target is Isaac Lab 3.0.0, Isaac Sim dependencies, and the declared Physics presets. Do not substitute the current/latest Isaac Lab API without validating its task configuration, sensor interfaces and physics backends.
