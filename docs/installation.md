# Installation

Use Linux x86-64, Python 3.12, a supported NVIDIA driver and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/sycamore-yu/Isaaclab-AerialNav-MBRL.git
cd Isaaclab-AerialNav-MBRL
uv sync --frozen --extra curobo
```

The lockfile pins Isaac Lab 3.0.0, Isaac Sim 6.2, Torch 2.11/CUDA 13, Newton 1.6.1 and cuRobo revision `78fd485fa82d9b9a063fb4985e371814587e666a`. Review NVIDIA's license terms when prompted at first simulator startup.

## Development checks

```bash
uv sync --frozen --extra curobo --group dev
uv run --frozen --no-sync ruff check src scripts tests
uv run --frozen --no-sync pytest -q -m 'not integration'
```

The default task requires CUDA and cuRobo. Use `--geometry_backend native` for analytic/mesh geometry, or `physics=isaacsim_physx --frontend torch` for the alternate physical runtime.
