# Changelog

## v0.1.0 (local release candidate, 2026-10-31)

- Created a static-only user-facing training and evaluation profile.
- Established independent `aerialnav` namespace and `diffnav` implementation.
- Preserved the optional PPO baseline with distinct task registration.
- Migrated the best static policy into an inference-only checkpoint.
- Published sanitized single-run wall-clock benchmark evidence.
- Removed internal experiment history, developer notes, runtime paths, and full training checkpoint metadata.
- Retained upstream legal notices and exact Hummingbird asset attribution.

**Not yet verified:** independent multi-seed static-navigation evaluation of this reorganized source tree, and Newton/Warp policy equivalence.
