"""Convenience recipe for the shared BPTT entry point and native PointNet preset."""

import sys

from train_diffnav import main

if __name__ == "__main__":
    if not any(token.startswith("presets=") for token in sys.argv[1:]):
        sys.argv.append("presets=pointnet")
    main()
