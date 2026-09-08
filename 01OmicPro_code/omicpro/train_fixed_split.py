"""Explicit validation/test fold training entry point.

This is the accurately named public entry point. ``evaluate_fixed_split`` is
kept as a compatibility alias for earlier releases.
"""

from .evaluate_fixed_split import main, run_one_trait_fixed_split

__all__ = ["main", "run_one_trait_fixed_split"]


if __name__ == "__main__":
    main()
