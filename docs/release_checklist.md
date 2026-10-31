# Local release candidate checklist

| Area | Current state | Evidence or remaining gate |
|---|---|---|
| Independent source directory | Complete | Clean export into `opensource_projects/aerialnav-static` |
| Package / method / task naming | Complete | `aerialnav`, `diffnav`, task IDs and scripts; upstream license notices preserved |
| Static defaults | Complete in public entrypoints | S0, 300 static, 0 dynamic, PhysX/Torch |
| Inference weight export | Complete | `weights/best_static_s0.pt`; new and original CPU model outputs bitwise identical on three batches |
| Training benchmark evidence | Complete | `docs/data/static_training_summary.json` with process wall clock |
| Source checks | **Passed** | `ruff check`, `compileall`, public identifier scan and TOML/lock project-name consistency |
| Unit and contract tests | **Passed** | 364 passed, 13 skipped, 38 warnings, 85 subtests. Optional local upstream parity subset: 15 passed |
| Simulator smoke | **Blocked: NVIDIA Omniverse EULA not accepted** | First launch requested explicit NVIDIA EULA confirmation. No acceptance was performed on behalf of the user |
| Independent held-out evaluation | **Not verified** | At least 3 unseen map seeds × 1024 completed episodes; include all failure categories |
| Optional Newton/Warp parity | **Not verified** | Report separately from PhysX; no unverified speedup claim |
| External release on GitHub | Not performed | Manual owner review and authorized push |

Static task and baseline PPO share some low-level internal utilities with the upstream multi-obstacle environment. This repository intentionally exposes a static profile without claiming that every underlying helper is intrinsically static-only.
