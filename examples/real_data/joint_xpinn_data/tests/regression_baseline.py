"""Regression baseline for build_dataset's output on known-good configs.

This is a code-correctness check: does rebuilding a config still produce
the same output dict as a frozen known-good run? Distinct from
`checks/`'s data-consistency checks (see CONTEXT.md's "Consistency
check") — this doesn't judge whether the physics looks right, only
whether a change to shared pipeline code (domain.py in particular)
silently changed a shelf it wasn't supposed to touch. That failure mode
recurred throughout Ross East/Byrd development (see HANDOFF.md) and was
previously caught only by manual, ad hoc re-diffing of `.mat` field
counts after each change.

REFERENCE_CONFIGS are all "simple"-strategy (Amery-class) shelves —
build_dataset's output for them must never depend on any Byrd-class
(flow-restricted) code path. Byrd's own output isn't frozen here: its
geometry is still evolving (see HANDOFF.md's open items), so there's no
known-good state yet to regress against.

Usage (from /Users/jiapchen/Software/DIFFICE_jax/examples/real_data):
    PY -m joint_xpinn_data.tests.regression_baseline freeze   # (re-)establish the baseline
    PY -m joint_xpinn_data.tests.regression_baseline verify   # compare a fresh build against it

`freeze` overwrites the stored baseline — only run it after confirming a
change to build_dataset's output is intentional, not to make a failing
`verify` pass.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from joint_xpinn_data.build_dataset import build_dataset, config_stem
from joint_xpinn_data.config import load_config

BASELINE_DIR = Path(__file__).parent / "baselines"

REFERENCE_CONFIGS = [
    "joint_xpinn_data/data_build_configs/amery_lambert.yaml",
    "joint_xpinn_data/data_build_configs/amery_mellor.yaml",
    "joint_xpinn_data/data_build_configs/amery_fisher.yaml",
]

# shapely's MultiPoint.buffer() tessellates circles with some noise
# (confirmed in HANDOFF.md at ~0.3%), so geometry-derived coordinates
# aren't bit-exact across two runs even with zero logic change -
# compare with a tolerance, not exact equality, for float arrays.
RTOL = 1e-6
ATOL = 1e-6

_MAX_PRINTED_MISMATCHES = 20


def _diff(path: str, a, b, mismatches: list[str]) -> None:
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            mismatches.append(f"{path}: type differs {type(a).__name__} vs {type(b).__name__}")
            return
        keys_a, keys_b = set(a), set(b)
        if keys_a != keys_b:
            mismatches.append(f"{path}: keys differ, only-in-baseline={keys_a - keys_b}, only-in-current={keys_b - keys_a}")
        for k in sorted(keys_a & keys_b):
            _diff(f"{path}.{k}", a[k], b[k], mismatches)
        return

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            mismatches.append(f"{path}: type differs {type(a).__name__} vs {type(b).__name__}")
            return
        if len(a) != len(b):
            mismatches.append(f"{path}: length differs {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, mismatches)
        return

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            mismatches.append(f"{path}: shape differs {a.shape} vs {b.shape}")
            return
        if a.size == 0:
            return
        if a.dtype == object or b.dtype == object:
            # A MATLAB-style "cell" array (e.g. x_md/y_md: one cell per
            # interface, each holding its own point array) — np.array_equal
            # can't reduce this (each element is itself a non-scalar array,
            # so `(a1 == a2).all()` raises "ambiguous truth value"). Recurse
            # into each cell instead of comparing the container directly.
            for idx, (ai, bi) in enumerate(zip(a.flat, b.flat)):
                _diff(f"{path}[{idx}]", ai, bi, mismatches)
            return
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
                max_diff = np.nanmax(np.abs(a - b))
                mismatches.append(f"{path}: values differ, max abs diff {max_diff:.6g} (shape {a.shape})")
        elif not np.array_equal(a, b):
            n_diff = int(np.sum(a != b))
            mismatches.append(f"{path}: values differ in {n_diff}/{a.size} entries (dtype {a.dtype})")
        return

    if a != b:
        mismatches.append(f"{path}: {a!r} != {b!r}")


def freeze(config_paths: list[str]) -> None:
    BASELINE_DIR.mkdir(exist_ok=True)
    for cp in config_paths:
        config = load_config(cp)
        stem = config_stem(config)
        print(f"Building {stem} ...")
        output = build_dataset(config)
        out_path = BASELINE_DIR / f"{stem}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(output, f)
        print(f"  froze baseline -> {out_path}")


def verify(config_paths: list[str]) -> None:
    failed = []
    for cp in config_paths:
        config = load_config(cp)
        stem = config_stem(config)
        baseline_path = BASELINE_DIR / f"{stem}.pkl"
        if not baseline_path.exists():
            raise SystemExit(f"No baseline for {stem} - run `freeze` first ({baseline_path})")
        with open(baseline_path, "rb") as f:
            baseline = pickle.load(f)

        print(f"Rebuilding {stem} ...")
        current = build_dataset(config)

        mismatches: list[str] = []
        _diff(stem, baseline, current, mismatches)
        if mismatches:
            failed.append(stem)
            print(f"  MISMATCH ({len(mismatches)} field(s)):")
            for m in mismatches[:_MAX_PRINTED_MISMATCHES]:
                print(f"    {m}")
            if len(mismatches) > _MAX_PRINTED_MISMATCHES:
                print(f"    ... and {len(mismatches) - _MAX_PRINTED_MISMATCHES} more")
        else:
            print("  OK - matches baseline")

    if failed:
        raise SystemExit(f"Regression FAILED for: {', '.join(failed)}")
    print("All configs match their frozen baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["freeze", "verify"])
    parser.add_argument(
        "configs", nargs="*", default=REFERENCE_CONFIGS,
        help="Config YAML paths to include (default: all reference Amery configs)",
    )
    args = parser.parse_args()
    (freeze if args.mode == "freeze" else verify)(args.configs)


if __name__ == "__main__":
    main()
