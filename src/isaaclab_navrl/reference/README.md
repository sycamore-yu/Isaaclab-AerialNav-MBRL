# Historical DCOL comparator

This package is an explicit reference for the requested DCOL/SDF control, not the current Zhang collision implementation. Normal `ZhangTrainer` only calls the existing geometry SDF; only `scripts/train_dcol_reference.py` selects this subclass and its configuration.

The alpha solver and hybrid collision functions retain the same AST as `main@addcbff:src/isaaclab_navrl/zhang/dcol.py`, before its generic AHAC helpers. Only the namespace and trailing whitespace differ. Tests of those historical contracts import this package deliberately. Every other training setting, loss construction, optimizer, depth policy and integrator implementation is shared with the production trainer.
