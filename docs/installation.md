# Installation

Use Linux x86-64, Python 3.12, a supported NVIDIA driver, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra curobo
```

The lockfile selects Isaac Lab 3.0.0, Isaac Sim 6.2, Torch 2.11/CUDA 13, Newton 1.6.1, and cuRobo revision `78fd485fa82d9b9a063fb4985e371814587e666a`. The `curobo` extra also installs its CUDA Python runtime; no workstation path or external `.pth` file is required by this repository.

First simulator startup can require accepting NVIDIA's license terms. Review and accept them yourself before unattended simulator runs. This project does not distribute Isaac Sim binaries.

For CPU source checks without the geometry extra:

```bash
uv sync --frozen
uv run --frozen --no-sync python scripts/audit_release.py
uv run --frozen --no-sync pytest -q -m 'not integration'
```

The default training configuration requires cuRobo and CUDA. `--geometry_backend native` selects the explicit analytic/mesh control, not a silent fallback. Set `physics=isaacsim_physx --frontend torch` explicitly to use the alternate physical backend; its results are separate from the Newton baseline.
