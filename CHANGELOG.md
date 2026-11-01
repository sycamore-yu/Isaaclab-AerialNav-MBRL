# Changelog

## Unreleased

- Replace the static-only draft default with the direct randomized rectangular navigation baseline.
- Use Newton/Warp, cuRobo SDF, trapezoidal surrogate integration and pre-reset loss construction.
- Add the 10-second arrival-wait/final-state success protocol and complete failure reporting.
- Name the sampling protocol `terminal`; keep old inference-weight labels compatible.
- Preserve the legacy static weights and benchmark data separately from the new baseline.
- Replace the draft Static Gym IDs with neutral task IDs and pin the optional geometry dependency.
- Keep the public Git history independent of the development repository.
