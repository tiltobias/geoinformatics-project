"""
Random-walk clock-offset generator.

All offsets are expressed in *nanoseconds* so that you can multiply by the
speed of light (c) later without unit confusion.

Usage
-----
>>> from random_walk import generate, to_csv, plot
>>> offsets = generate(n=1_000, step_ns=0.5, offset0=0.0, seed=42)
>>> to_csv(offsets, "clock_offset.csv")     # → writes CSV
>>> plot(offsets)                           # → quick diagnostic plot
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate(
    n: int = 1_000,
    step_ns: float = 0.5,
    offset0: float = 0.0,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Create a simple ±step random-walk sequence.

    Parameters
    ----------
    n
        Number of epochs.
    step_ns
        Step size *in nanoseconds* for each up/down move.
    offset0
        Initial offset at epoch 0 (nanoseconds).
    seed
        RNG seed.  Pass ``None`` for unpredictable results.

    Returns
    -------
    np.ndarray
        Shape (n,) array of offsets [ns].
    """
    rng = np.random.default_rng(seed)
    # first value is fixed (offset0); remaining n-1 steps are ±step_ns
    direction = rng.choice((-1.0, 1.0), size=n - 1)
    walk = np.empty(n, dtype=float)
    walk[0] = offset0
    walk[1:] = offset0 + step_ns * np.cumsum(direction)
    return walk


def to_csv(offsets: np.ndarray, path: str | Path = "clock_offset.csv") -> Path:
    """
    Write the offsets array to *path* with column name ``Clock Offset [ns]``.

    Returns the Path object for convenience.
    """
    path = Path(path)
    pd.DataFrame({"Clock Offset [ns]": offsets}).to_csv(path, index=False)
    return path


def plot(offsets: np.ndarray) -> None:
    """Quick-and-dirty line plot of the random walk."""
    plt.plot(offsets)
    plt.title("Clock offset random walk")
    plt.xlabel("Epoch")
    plt.ylabel("Offset [ns]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Optional CLI                                                                
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Generate a random-walk clock offset.")
    p.add_argument("-n",  "--epochs",   type=int,   default=1_000)
    p.add_argument("-s",  "--step-ns",  type=float, default=0.5)
    p.add_argument("-o",  "--offset0",  type=float, default=0.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--csv",             type=Path,  default=Path("clock_offset.csv"),
                   help="Write output CSV; omit to skip saving.")
    p.add_argument("--plot",            action="store_true",
                   help="Show a matplotlib preview.")
    args = p.parse_args()

    arr = generate(args.epochs, args.step_ns, args.offset0, args.seed)

    if args.csv:
        to_csv(arr, args.csv)
        print(f"Wrote {args.csv}", file=sys.stderr)

    if args.plot:
        plot(arr)
