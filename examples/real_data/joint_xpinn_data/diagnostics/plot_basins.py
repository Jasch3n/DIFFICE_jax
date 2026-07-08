"""Plot every named drainage basin, and check whether ice shelves are
partitioned along the same boundaries.

Usage:
    python -m joint_xpinn_data.diagnostics.plot_basins
"""

from pathlib import Path

import numpy as np

from joint_xpinn_data.utils.color_utils import distinct_colors
from joint_xpinn_data.config import DEFAULT_PATHS
from joint_xpinn_data.data_sources import boundaries


def _draw_polygon(ax, poly, **kwargs) -> None:
    """Draw a shapely (Multi)Polygon on an existing axis, holes and all.

    Uses a compound path (exterior + interior rings as separate subpaths)
    so matplotlib's nonzero winding fill rule renders holes correctly —
    relies on the same ESRI winding convention (exterior clockwise, holes
    counter-clockwise) that boundaries._rings_to_polygon already assumes.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        rings = [np.asarray(g.exterior.coords)] + [np.asarray(r.coords) for r in g.interiors]
        path = MplPath.make_compound_path(*[MplPath(r) for r in rings])
        ax.add_patch(PathPatch(path, **kwargs))


def plot_basins(out_path: Path, label: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat_path = str(DEFAULT_PATHS["basins_refined"])
    names = boundaries.list_names(mat_path)
    label_points = boundaries.get_label_points(mat_path)
    colors = distinct_colors(len(names))

    fig, ax = plt.subplots(figsize=(24, 24))
    all_bounds = []
    for color, name in zip(colors, names):
        poly = boundaries.get_named_polygon(mat_path, name)
        _draw_polygon(ax, poly, facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.65)
        all_bounds.append(poly.bounds)
        if label:
            xc, yc = label_points[name]
            ax.text(xc, yc, name, fontsize=3.5, ha="center", va="center", clip_on=True)

    bounds = np.array(all_bounds)
    ax.set_xlim(bounds[:, 0].min(), bounds[:, 2].max())
    ax.set_ylim(bounds[:, 1].min(), bounds[:, 3].max())
    ax.set_aspect("equal")
    ax.set_title(f"Antarctic drainage basins, refined (n={len(names)})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path} ({len(names)} basins labeled)")
    print(
        "Note: basins cluster tightly in some coastal regions (e.g. the LarsenA-G, "
        "Ross East1-5, Wilkins Island1-6 groups) — labels there will overlap at any "
        "readable font size on a continent-scale figure. Zoom into the saved PNG, or "
        "pull individual names via boundaries.list_names(...) for those areas."
    )


def plot_basins_vs_iceshelves(out_path: Path) -> None:
    """Overlay ice-shelf polygons (black outline, no fill) on the basin
    map — answers "are ice shelves partitioned by drainage basin" visually:
    they are not, since a single ice-shelf polygon's boundary cuts across
    several differently-colored basins rather than following one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basins_path = str(DEFAULT_PATHS["basins_refined"])
    shelves_path = str(DEFAULT_PATHS["iceshelves"])
    basin_names = boundaries.list_names(basins_path)
    shelf_names = boundaries.list_names(shelves_path)
    colors = distinct_colors(len(basin_names))

    fig, ax = plt.subplots(figsize=(24, 24))
    all_bounds = []
    for color, name in zip(colors, basin_names):
        poly = boundaries.get_named_polygon(basins_path, name)
        _draw_polygon(ax, poly, facecolor=color, edgecolor="none", alpha=0.5)
        all_bounds.append(poly.bounds)

    for name in shelf_names:
        poly = boundaries.get_named_polygon(shelves_path, name)
        _draw_polygon(ax, poly, facecolor="none", edgecolor="black", linewidth=0.8)
        all_bounds.append(poly.bounds)

    bounds = np.array(all_bounds)
    ax.set_xlim(bounds[:, 0].min(), bounds[:, 2].max())
    ax.set_ylim(bounds[:, 1].min(), bounds[:, 3].max())
    ax.set_aspect("equal")
    ax.set_title("Basins (colored fill) vs. ice-shelf outlines (black) — not the same partition")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path}")


def summarize_basin_iceshelf_overlap() -> None:
    """How many basins intersect each ice shelf — quantifies whether ice
    shelves respect basin boundaries (they don't)."""
    basins_path = str(DEFAULT_PATHS["basins_refined"])
    shelves_path = str(DEFAULT_PATHS["iceshelves"])
    basin_names = boundaries.list_names(basins_path)
    shelf_names = boundaries.list_names(shelves_path)

    basin_polys = {n: boundaries.get_named_polygon(basins_path, n) for n in basin_names}
    multi_fed = []
    zero_fed = []
    for shelf_name in shelf_names:
        shelf_poly = boundaries.get_named_polygon(shelves_path, shelf_name)
        sb0, sb1, sb2, sb3 = shelf_poly.bounds
        touching = []
        for basin_name, basin_poly in basin_polys.items():
            bb0, bb1, bb2, bb3 = basin_poly.bounds
            if bb2 < sb0 or bb0 > sb2 or bb3 < sb1 or bb1 > sb3:
                continue  # bbox prefilter — most of the 199 basins are nowhere near this shelf
            if basin_poly.intersects(shelf_poly):
                touching.append(basin_name)
        if len(touching) > 1:
            multi_fed.append((shelf_name, touching))
        elif len(touching) == 0:
            zero_fed.append(shelf_name)

    print(f"{len(shelf_names)} ice shelves, {len(basin_names)} basins.")
    print(f"{len(multi_fed)} ice shelves are fed by more than one basin, e.g.:")
    for shelf_name, touching in multi_fed[:10]:
        print(f"  {shelf_name}: {touching}")
    if len(multi_fed) > 10:
        print(f"  ... and {len(multi_fed) - 10} more")
    print(
        f"{len(zero_fed)} ice shelves have no bounding-box-adjacent basin at all "
        "(likely small/isolated shelves not the primary outlet of any refined basin)."
    )


def main() -> None:
    out_dir = Path("joint_xpinn_data/output/basins")
    plot_basins(out_dir / "all_basins.png")
    plot_basins_vs_iceshelves(out_dir / "basins_vs_iceshelves.png")
    summarize_basin_iceshelf_overlap()


if __name__ == "__main__":
    main()
