"""Compare alternative GL/calving-front sources against each other for a
given ice_shelf/grounding_zone — a sanity check before trusting a source
swap, and a way to quantify how much two products actually disagree.

Usage:
    python -m joint_xpinn_data.diagnostics.compare_sources --ice-shelf Amery --grounding-zone Lambert
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from joint_xpinn_data.build_dataset import config_output_dir
from joint_xpinn_data.config import PipelineConfig, load_config
from joint_xpinn_data.data_sources import boundaries, calving_front, grounding_line

GL_COMPARISON_SOURCES = ("measures_boundaries_2008", "bedmachine_mask")
FRONT_COMPARISON_SOURCES = ("bedmachine_mask", "antarctic_boundaries_mask", "velocity_mask")


def _nn_stats(a: np.ndarray, b: np.ndarray) -> dict:
    tree = cKDTree(b)
    d, _ = tree.query(a)
    return {"mean_m": float(d.mean()), "median_m": float(np.median(d)), "max_m": float(d.max())}


def compare_grounding_line(config: PipelineConfig, sources=GL_COMPARISON_SOURCES) -> dict:
    basin = boundaries.get_named_polygon(str(config.path("basins_refined")), config.grounding_zone)
    return {src: grounding_line.load_grounding_line(replace(config, grounding_line_source=src), basin) for src in sources}


def compare_calving_front(config: PipelineConfig, sources=FRONT_COMPARISON_SOURCES, gl_source: str | None = None) -> dict:
    basin = boundaries.get_named_polygon(str(config.path("basins_refined")), config.grounding_zone)
    gl_source = gl_source or config.grounding_line_source
    gl = grounding_line.load_grounding_line(replace(config, grounding_line_source=gl_source), basin)
    return {src: calving_front.load_calving_front(replace(config, calving_front_source=src), basin, gl) for src in sources}


def print_pairwise_stats(label: str, geoms: dict) -> None:
    names = list(geoms)
    print(f"--- {label} ---")
    for name in names:
        print(f"  {name}: {len(geoms[name].all_points())} points, product={geoms[name].product!r}, epoch={geoms[name].epoch!r}")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            a, b = geoms[names[i]].all_points(), geoms[names[j]].all_points()
            stats = _nn_stats(a, b)
            print(
                f"  {names[i]} -> nearest {names[j]}: "
                f"mean={stats['mean_m']:.0f}m median={stats['median_m']:.0f}m max={stats['max_m']:.0f}m"
            )


def plot_comparison(geoms: dict, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    markers = ["o", "^", "s", "D"]
    for (name, geo), marker in zip(geoms.items(), markers):
        pts = geo.all_points()
        ax.scatter(pts[:, 0], pts[:, 1], s=6, marker=marker, alpha=0.6, label=f"{name} (n={len(pts)})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None,
        help="Path to a saved config YAML (see configs/*.yaml) instead of bare "
        "--ice-shelf/--grounding-zone/--buffer-km — picks up the shelf's actual "
        "grounding_line_kwargs/calving_front_kwargs if it sets any. Overrides those "
        "three flags. Note: compare_grounding_line/compare_calving_front always compare "
        "raw sources against the named basin/shelf, independent of region_strategy or "
        "floating_region_source — those don't affect what this tool measures.",
    )
    parser.add_argument("--ice-shelf", default="Amery")
    parser.add_argument("--grounding-zone", default="Lambert")
    parser.add_argument("--buffer-km", type=float, default=100.0)
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to save PNGs (default: joint_xpinn_data/output/<config>/figures)",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig(ice_shelf=args.ice_shelf, grounding_zone=args.grounding_zone, buffer_km=args.buffer_km)
    out_dir = Path(args.out_dir) if args.out_dir else config_output_dir(config, "joint_xpinn_data/output") / "figures"
    stem = f"{config.ice_shelf}_{config.grounding_zone}"

    gl_geoms = compare_grounding_line(config)
    print_pairwise_stats("Grounding line", gl_geoms)
    plot_comparison(gl_geoms, f"Grounding line sources — {config.grounding_zone}", out_dir / f"{stem}_gl_comparison.png")

    front_geoms = compare_calving_front(config)
    print_pairwise_stats("Calving front", front_geoms)
    plot_comparison(front_geoms, f"Calving front sources — {config.ice_shelf}", out_dir / f"{stem}_front_comparison.png")


if __name__ == "__main__":
    main()
