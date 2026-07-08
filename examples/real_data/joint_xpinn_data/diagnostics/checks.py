"""CLI driver for the `joint_xpinn_data.checks` registry — one subcommand per
registered check, each with its own tuning knobs.

Usage:
    python -m joint_xpinn_data.diagnostics.checks velocity_vs_front --ice-shelf Amery --grounding-zone Lambert --search-radius-km 5
    python -m joint_xpinn_data.diagnostics.checks hydrostatic_equilibrium --ice-shelf Amery --grounding-zone Lambert
"""

import argparse
from pathlib import Path

import numpy as np

from joint_xpinn_data.build_dataset import config_output_dir
from joint_xpinn_data.checks import CHECKS
from joint_xpinn_data.config import PipelineConfig, load_config
from joint_xpinn_data.contracts import CheckResult
from joint_xpinn_data.utils.plot_utils import add_colorbar


def plot_check_result(result: CheckResult, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    if result.n_points:
        # 90th percentile, not a stricter one: a handful of extreme outliers
        # (e.g. rock-outcrop/margin sentinel artifacts) can otherwise blow
        # out the color scale and wash out the spatial pattern that
        # actually matters for this check.
        lim = np.percentile(np.abs(result.metric), 90)
        lim = lim if lim > 0 else 1.0
        sc = ax.scatter(result.x, result.y, s=6, c=result.metric, cmap="RdBu_r", vmin=-lim, vmax=lim)
        add_colorbar(fig, ax, sc, label=f"{result.name} [{result.unit}] (clipped at p90(|.|))")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    title = result.name + (f" ({result.region})" if result.region else "")
    ax.set_title(f"{title}, n={result.n_points}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default=None,
        help="Path to a saved config YAML (see configs/*.yaml) — loads the shelf's "
        "actual region_strategy/floating_region_source/etc. instead of PipelineConfig's "
        "bare defaults. Required for any shelf whose real config deviates from those "
        "defaults (e.g. Ross East/Byrd's region_strategy=flow_restricted) — otherwise "
        "this checks a different, unbuilt domain, not the one that was actually saved. "
        "Overrides --ice-shelf/--grounding-zone/--buffer-km.",
    )
    parser.add_argument("--ice-shelf", default="Amery")
    parser.add_argument("--grounding-zone", default="Lambert")
    parser.add_argument("--buffer-km", type=float, default=100.0)
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to save PNGs (default: joint_xpinn_data/output/<config>/figures)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="check", required=True)

    p_front = subparsers.add_parser("velocity_vs_front", help="Velocity vs. calving-front conformance")
    _add_shared_args(p_front)
    p_front.add_argument(
        "--search-radius-km", type=float, default=5.0,
        help="How far from the mapped front to pull in velocity points for evaluation "
        "(not a pass/fail tolerance — the check itself requires exact conformance)",
    )

    p_hydro = subparsers.add_parser("hydrostatic_equilibrium", help="Thickness vs. surface elevation flotation check")
    _add_shared_args(p_hydro)
    p_hydro.add_argument("--threshold-kgm2", type=float, default=None, help="Optional pass/fail tolerance on |delta|")

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig(ice_shelf=args.ice_shelf, grounding_zone=args.grounding_zone, buffer_km=args.buffer_km)
    out_dir = Path(args.out_dir) if args.out_dir else config_output_dir(config, "joint_xpinn_data/output") / "figures"

    if args.check == "velocity_vs_front":
        result = CHECKS["velocity_vs_front"](config, search_radius_km=args.search_radius_km)
    else:
        result = CHECKS["hydrostatic_equilibrium"](config, threshold=args.threshold_kgm2)

    print(result.summary())
    stem = f"{config.ice_shelf}_{config.grounding_zone}"
    plot_check_result(result, out_dir / f"{stem}_{args.check}.png")


if __name__ == "__main__":
    main()
