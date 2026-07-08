"""Visual validation plots for a preprocessed two-region .mat dataset.

Mirrors the three validation figures produced by PinningPointInversion's
+ProcessSyntheticData/plotValidationProcessing.m for the synthetic ISSM
pipeline (regions + interface + calving front, collocation libraries,
sparse-vs-dense thickness), so real-data output can be eyeballed the same
way the synthetic output already is. Adds two panels that don't apply to
synthetic data: velocity magnitude, and ols_d (signed distance to the
grounding line) — both are useful physical-plausibility checks that have
no synthetic-data analogue since those quantities are ISSM ground truth
there, not something worth re-plotting for a sanity check.

Usage:
    python -m joint_xpinn_data.diagnostics.plot_validation joint_xpinn_data/output/Amery_Lambert_10km/Amery_Lambert_10km.mat
    python -m joint_xpinn_data.diagnostics.plot_validation joint_xpinn_data/output/Amery_Lambert_10km/Amery_Lambert_10km.mat --show
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

from joint_xpinn_data.utils.plot_utils import add_colorbar

REGION_COLORS = ["#1a59d9", "#e64026"]
SECONDS_PER_YEAR = 365.25 * 86400.0


def _load(mat_path) -> dict:
    return sio.loadmat(mat_path, simplify_cells=True)


def _as_cells(value) -> list:
    """Normalize a MISMIP-schema "cell" field to a plain Python list.

    scipy.io.loadmat(..., simplify_cells=True) returns MATLAB cell arrays
    as object-dtype ndarrays, not Python lists — must check dtype, not
    just `isinstance(..., list)`, or a 2-region field gets treated as one
    opaque blob instead of two arrays.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value)
    return [value]


def _is_empty(x) -> bool:
    return np.size(x) == 0


def _concat(cells) -> np.ndarray:
    return np.concatenate([np.atleast_1d(c) for c in cells if not _is_empty(c)]) if cells else np.zeros(0)


def _region_labels_list(data: dict) -> list:
    """region_labels round-trips as a plain (non-object-dtype) string
    ndarray, unlike the per-region numeric fields — _as_cells's dtype
    check doesn't fire for it, so it needs its own normalization."""
    labels = data.get("region_labels", [])
    if isinstance(labels, np.ndarray):
        return [str(l) for l in np.atleast_1d(labels)]
    if isinstance(labels, list):
        return [str(l) for l in labels]
    return [str(labels)]


def _region_label(data: dict, i: int) -> str:
    labels = _region_labels_list(data)
    return labels[i] if i < len(labels) else f"region {i + 1}"


def plot_regions_and_interface(ax, data: dict) -> None:
    xd, yd = _as_cells(data["xd"]), _as_cells(data["yd"])
    for i, (x, y) in enumerate(zip(xd, yd)):
        if _is_empty(x):
            continue
        ax.scatter(
            x, y, s=0.5, color=REGION_COLORS[i % len(REGION_COLORS)], alpha=0.5,
            linewidths=0, label=f"velocity points ({_region_label(data, i)})",
        )

    for x, y in zip(_as_cells(data.get("x_md", [])), _as_cells(data.get("y_md", []))):
        if _is_empty(x):
            continue
        ax.plot(x, y, "k.", markersize=1.5, label="grounding line (x_md/y_md)")

    xct, yct = _as_cells(data.get("xct", [])), _as_cells(data.get("yct", []))
    nnct = _as_cells(data.get("nnct", []))
    for x, y, nn in zip(xct, yct, nnct):
        if _is_empty(x):
            continue
        ax.plot(x, y, "ms", markersize=1.5, label="calving front (xct/yct)")
        if not _is_empty(nn) and np.ndim(nn) == 2 and nn.shape[1] == 2:
            step = max(len(x) // 60, 1)
            ax.quiver(
                x[::step], y[::step], nn[::step, 0], nn[::step, 1],
                color="m", scale=1 / 3000.0, scale_units="xy", width=0.002,
            )

    for x, y in zip(_as_cells(data.get("xdir", [])), _as_cells(data.get("ydir", []))):
        if _is_empty(x):
            continue
        ax.plot(x, y, "g^", markersize=1.5, label="Dirichlet cut boundary (xdir/ydir)")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Regions, grounding line, calving front (+normals), cut boundary")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)


def plot_collocation(fig, data: dict) -> None:
    xcol, ycol = _as_cells(data["xcol"]), _as_cells(data["ycol"])
    axes = fig.subplots(1, len(xcol), squeeze=False)[0]
    for i, ax in enumerate(axes):
        x, y = xcol[i], ycol[i]
        label = _region_label(data, i)
        if _is_empty(x):
            ax.set_title(f"Collocation library, {label} (empty)")
            continue
        ax.scatter(x, y, s=0.5, color=REGION_COLORS[i % len(REGION_COLORS)], alpha=0.5, linewidths=0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Collocation library ({label})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")


def _shared_limits(*arrays, lo: float = 1.0, hi: float = 99.0):
    """(vmin, vmax) percentiles across every finite value in `arrays` pooled
    together — so panels/regions plotted against it share one color scale
    (same value -> same color everywhere), robust to a few outliers."""
    pooled = np.concatenate([np.atleast_1d(a).ravel() for a in arrays]) if arrays else np.zeros(0)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return None, None
    return float(np.percentile(pooled, lo)), float(np.percentile(pooled, hi))


def plot_thickness(fig, data: dict) -> None:
    axes = fig.subplots(1, 2)

    xd_h = _concat(_as_cells(data["xd_h"]))
    yd_h = _concat(_as_cells(data["yd_h"]))
    hd = _concat(_as_cells(data["hd"]))
    xd = _concat(_as_cells(data["xd"]))
    yd = _concat(_as_cells(data["yd"]))
    h_dense = _concat(_as_cells(data["h_dense"]))

    # One color scale across both panels (and, since each is a concatenation
    # over regions, across grounded and floating too), so sparse and dense
    # are directly comparable.
    vmin, vmax = _shared_limits(hd, h_dense)

    sc0 = axes[0].scatter(xd_h, yd_h, s=0.3, c=hd, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Sparse thickness observations (xd_h/yd_h/hd), n={len(xd_h)}")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    add_colorbar(fig, axes[0], sc0, label="Thickness [m]")

    sc1 = axes[1].scatter(xd, yd, s=0.3, c=h_dense, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Dense thickness reference (xd/yd/h_dense), n={len(xd)}")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    add_colorbar(fig, axes[1], sc1, label="Thickness [m]")


def plot_velocity_and_ols(fig, data: dict) -> None:
    from matplotlib.colors import LogNorm

    axes = fig.subplots(1, 2)

    xd, yd = _as_cells(data["xd"]), _as_cells(data["yd"])
    ud, vd = _as_cells(data["ud"]), _as_cells(data["vd"])
    x_all = _concat(xd)
    y_all = _concat(yd)
    speed = _concat([np.sqrt(np.atleast_1d(u) ** 2 + np.atleast_1d(v) ** 2) for u, v in zip(ud, vd)])
    speed_myr = speed * SECONDS_PER_YEAR

    # Log scale: speed spans ~1 to >1000 m/yr, dominated by the fast front, so
    # a linear scale washes out the slower grounded region and most of the
    # shelf. Floor vmin at the 1st percentile of positive speeds (>= 1 m/yr)
    # so LogNorm has a finite lower bound; the single scatter pools both
    # regions, so grounded and floating share this scale.
    positive = speed_myr[speed_myr > 0]
    vmin = max(1.0, float(np.percentile(positive, 1))) if positive.size else 1.0
    vmax = float(np.percentile(speed_myr, 99.5)) if speed_myr.size else vmin * 10
    sc0 = axes[0].scatter(
        x_all, y_all, s=0.5, c=np.clip(speed_myr, vmin, None), cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    axes[0].set_aspect("equal")
    axes[0].set_title("Velocity magnitude [m/yr] (log scale)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    add_colorbar(fig, axes[0], sc0, label="Speed [m/yr]")

    ols_all = _concat(_as_cells(data["ols_d"]))
    lim = np.percentile(np.abs(ols_all), 98) if not _is_empty(ols_all) else 1.0
    sc1 = axes[1].scatter(x_all, y_all, s=0.5, c=ols_all, cmap="RdBu", vmin=-lim, vmax=lim)
    axes[1].set_aspect("equal")
    axes[1].set_title("ols_d: signed distance to grounding line [m]\n(+grounded, -floating)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    add_colorbar(fig, axes[1], sc1, label="ols_d [m]")


def make_all_figures(mat_path, out_dir=None, show: bool = False) -> Path:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load(mat_path)
    mat_path = Path(mat_path)
    out_dir = Path(out_dir) if out_dir else mat_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mat_path.stem

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    plot_regions_and_interface(ax1, data)
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{stem}_regions_interface.png", dpi=150)

    fig2 = plt.figure(figsize=(11, 5.5))
    plot_collocation(fig2, data)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{stem}_collocation.png", dpi=150)

    fig3 = plt.figure(figsize=(11, 5.5))
    plot_thickness(fig3, data)
    fig3.tight_layout()
    fig3.savefig(out_dir / f"{stem}_thickness.png", dpi=150)

    fig4 = plt.figure(figsize=(11, 5.5))
    plot_velocity_and_ols(fig4, data)
    fig4.tight_layout()
    fig4.savefig(out_dir / f"{stem}_velocity_ols.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved validation figures to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", help="Path to a processed two-region .mat file")
    parser.add_argument("--out-dir", default=None, help="Directory to save PNGs (default: <mat dir>/figures)")
    parser.add_argument("--show", action="store_true", help="Also display figures interactively")
    args = parser.parse_args()
    make_all_figures(args.mat_path, out_dir=args.out_dir, show=args.show)


if __name__ == "__main__":
    main()
