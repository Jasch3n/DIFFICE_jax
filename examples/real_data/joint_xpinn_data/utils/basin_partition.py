"""Partition an ice shelf into provinces by which grounded drainage basin
feeds each part of it — the ice-shelf polygons in iceshelves_2008_v2.mat
are NOT partitioned along basin boundaries (see plot_basins.py), so this
derives a flow-informed partition instead.

Method (forward streamlines seeded on the real flux gate): for each basin
touching the shelf, find the actual grounded/floating interface pixels
(BedMachine mask adjacency — same technique the calving-front mask
providers use), keep only the ones with real inward flow (velocity
component along the outward normal > min_normal_velocity_myr), and use
those as seeds — they're "close to the grounding line but still
grounded," one seed per ~500m of genuine physical inflow. Integrate each
seed forward (with the flow, arc-length-parametrized RK4 — see
_rk4_step) until it leaves the shelf polygon, recording the whole path.
Every shelf point is then labeled by nearest-neighbor lookup against the
combined cloud of all basins' path points.

This replaced an earlier backward-tracing design (integrate every shelf
point backward until it re-enters some basin) for two reasons:
  - Seeding on the flux gate makes basin selection self-correcting. An
    earlier version filtered candidate basins by shared-boundary
    *length* (short arc = real tributary, long arc = lateral rock wall)
    — that heuristic was wrong for Amery: MacRobertson Land and American
    HighLand (long shared boundaries, assumed to be walls) turned out to
    have the 1st- and 3rd-largest real inward flux of any basin touching
    Amery (checked via BedMachine mask: 98.2% grounded ice, not rock),
    while Mawson Coast and Publications (short shared boundaries,
    assumed to be real tributaries) turned out to have *zero* actual
    grounded/floating interface pixels — no physical connection at all.
    Seeding directly on measured flux sidesteps needing that heuristic:
    a basin with no real interface contributes no seeds, automatically.
  - Backward tracing from arbitrary shelf points (particularly ones far
    downstream near the calving front) requires very long integration
    paths that accumulate error and often don't converge at all — ~45%
    of Amery's shelf never resolved even with an 8x larger step budget,
    confirming that was a fundamental accuracy problem, not a tuning
    one. Forward tracing from the grounding line is a shorter, more
    natural integration (with the flow, not against it), and every shelf
    point gets an answer immediately via nearest-neighbor — there's no
    unresolved state, only a `distance_m` confidence measure (how far
    from the nearest traced path).

Usage:
    python -m joint_xpinn_data.utils.basin_partition --ice-shelf Amery
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import shapely
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from joint_xpinn_data.utils.color_utils import distinct_colors
from joint_xpinn_data.config import DEFAULT_PATHS, PipelineConfig
from joint_xpinn_data.data_sources import boundaries, velocity
from joint_xpinn_data.data_sources.velocity import _index_slice
from joint_xpinn_data.utils.raster_utils import mask_boundary_points


@dataclass
class ShelfPartition:
    x: np.ndarray
    y: np.ndarray
    basin: np.ndarray  # string label, nearest traced path's basin
    distance_m: np.ndarray  # distance to that nearest path point — a
    # confidence measure, not a resolved/unresolved flag: every point
    # gets a label, but a large distance means it's far from any traced
    # streamline (e.g. downstream of a real gap between two basins'
    # diverging flow, or beyond where paths were allowed to run).


def all_touching_basins(shelf_name: str, paths: dict | None = None) -> list[str]:
    """Every basin whose polygon touches the shelf at all, with no
    filtering — real vs. spurious is decided later per-pixel by
    flux_gate_seeds, not here by basin-level geometry."""
    paths = paths if paths is not None else DEFAULT_PATHS
    basins_path = str(paths["basins_refined"])
    shelves_path = str(paths["iceshelves"])
    shelf_poly = boundaries.get_named_polygon(shelves_path, shelf_name)
    sb0, sb1, sb2, sb3 = shelf_poly.bounds

    touching = []
    for name in boundaries.list_names(basins_path):
        poly = boundaries.get_named_polygon(basins_path, name)
        bb0, bb1, bb2, bb3 = poly.bounds
        if bb2 < sb0 or bb0 > sb2 or bb3 < sb1 or bb1 > sb3:
            continue
        if poly.intersects(shelf_poly):
            touching.append(name)
    return touching


def _bedmachine_crop(bounds: tuple, pad_km: float = 20.0, paths: dict | None = None):
    paths = paths if paths is not None else DEFAULT_PATHS
    path = str(paths["bedmachine"])
    bx0, by0, bx1, by1 = bounds
    pad_m = pad_km * 1000.0
    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0 - pad_m, bx1 + pad_m)
        ys = _index_slice(y, by0 - pad_m, by1 + pad_m)
        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        mask = np.asarray(ds.variables["mask"][ys, xs])
    return x_sub, y_sub, mask


def _basin_shelf_interface(shelf_poly, basin_poly, pad_km: float, smooth_km: float, paths: dict | None = None):
    """Grounded(basin)/floating(shelf) interface pixels, with the
    velocity component along the outward (basin -> shelf) normal at
    each one (m/yr, positive = flowing into the shelf). Shared by
    basin_shelf_flux (aggregate stats) and flux_gate_seeds (per-pixel
    filtering for seeding).
    """
    x_sub, y_sub, mask = _bedmachine_crop(shelf_poly.bounds, pad_km=pad_km, paths=paths)
    xx, yy = np.meshgrid(x_sub, y_sub)
    pts_grid = shapely.points(xx.ravel(), yy.ravel())

    is_floating_shelf = (mask == 3) & shapely.contains(shelf_poly, pts_grid).reshape(mask.shape)
    is_grounded_basin = (mask == 2) & shapely.contains(basin_poly, pts_grid).reshape(mask.shape)

    points, normals = mask_boundary_points(x_sub, y_sub, is_grounded_basin, is_floating_shelf)
    if len(points) == 0:
        return points, np.zeros(0), abs(x_sub[1] - x_sub[0])

    velocity_at = velocity_field(shelf_poly.bounds, pad_km=pad_km, smooth_km=smooth_km, paths=paths)
    v = velocity_at(points)
    normal_velocity = np.einsum("ij,ij->i", v, normals)
    pixel_size_m = abs(x_sub[1] - x_sub[0])
    return points, normal_velocity, pixel_size_m


def basin_shelf_flux(
    shelf_name: str, basin_name: str, pad_km: float = 20.0, smooth_km: float = 0.5, paths: dict | None = None,
) -> dict:
    """Net ice flux crossing the grounded(basin)/floating(shelf)
    interface — a diagnostic for "is this really a contributing basin,"
    independent of any geometric proxy like shared-boundary length.
    `areal_flux` (mean normal velocity x interface length, m^2/yr) is
    the summary figure comparable across basins of different interface
    lengths. See the module docstring for what this caught for Amery.
    """
    paths = paths if paths is not None else DEFAULT_PATHS
    shelf_poly = boundaries.get_named_polygon(str(paths["iceshelves"]), shelf_name)
    basin_poly = boundaries.get_named_polygon(str(paths["basins_refined"]), basin_name)
    points, normal_velocity, pixel_size_m = _basin_shelf_interface(shelf_poly, basin_poly, pad_km, smooth_km, paths)
    if len(points) == 0:
        return {
            "n_interface_pixels": 0, "mean_normal_velocity_myr": 0.0,
            "median_normal_velocity_myr": 0.0, "areal_flux_m2yr": 0.0,
        }
    return {
        "n_interface_pixels": len(points),
        "mean_normal_velocity_myr": float(np.mean(normal_velocity)),
        "median_normal_velocity_myr": float(np.median(normal_velocity)),
        "areal_flux_m2yr": float(np.sum(normal_velocity) * pixel_size_m),
    }


def flux_gate_seeds(
    shelf_name: str, basin_name: str, pad_km: float = 20.0, smooth_km: float = 0.5,
    min_normal_velocity_myr: float = 0.0, paths: dict | None = None,
) -> np.ndarray:
    """Seed points for forward tracing: interface pixels between
    `basin_name` and `shelf_name` with real inward flow.

    Filtering per-pixel (not per-basin) matters: a basin's *net* flux can
    be small or negative (e.g. Amery's "Islands": -195,399 m^2/yr) while
    still having a handful of genuinely inflowing pixels mixed in with
    outflowing/noisy ones — keeping only pixels that individually pass
    the threshold is more honest than an all-or-nothing per-basin cutoff.
    """
    paths = paths if paths is not None else DEFAULT_PATHS
    shelf_poly = boundaries.get_named_polygon(str(paths["iceshelves"]), shelf_name)
    basin_poly = boundaries.get_named_polygon(str(paths["basins_refined"]), basin_name)
    points, normal_velocity, _ = _basin_shelf_interface(shelf_poly, basin_poly, pad_km, smooth_km, paths)
    if len(points) == 0:
        return points
    return points[normal_velocity > min_normal_velocity_myr]


def _fill_gaps(arr: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Replace `arr` values where `invalid` with the nearest valid pixel's
    value (same `invalid` mask must be used for both u and v, so a filled
    pixel gets a *paired* (u,v) from one real source pixel, not two
    different ones)."""
    if not invalid.any():
        return arr
    nearest = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return arr[tuple(nearest)]


def velocity_field(bounds: tuple, pad_km: float = 50.0, smooth_km: float = 0.5, paths: dict | None = None):
    """(u,v) interpolator over a cropped MEaSURES velocity grid.

    No-data pixels (CNT==0, ~5% of a typical crop) are filled from the
    nearest valid pixel rather than left as NaN, so a traced path
    doesn't die in a small data gap.

    `smooth_km` Gaussian-smooths u and v (after gap-filling) before
    building the interpolator — removes pixel-level noise that can fake
    small-scale flow direction changes, at the cost of blurring real
    velocity structure near tight confluences if set too large. 0.5km
    is a light default (~1 pixel); set to 0 to disable.
    """
    paths = paths if paths is not None else DEFAULT_PATHS
    path = str(paths["measures_velocity"])
    bx0, by0, bx1, by1 = bounds
    pad_m = pad_km * 1000.0

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0 - pad_m, bx1 + pad_m)
        ys = _index_slice(y, by0 - pad_m, by1 + pad_m)
        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        vx = np.asarray(ds.variables["VX"][ys, xs])
        vy = np.asarray(ds.variables["VY"][ys, xs])
        cnt = np.asarray(ds.variables["CNT"][ys, xs])

    invalid = cnt == 0
    vx = _fill_gaps(vx, invalid)
    vy = _fill_gaps(vy, invalid)

    if smooth_km > 0:
        pixel_size_m = abs(x_sub[1] - x_sub[0])
        sigma_px = (smooth_km * 1000.0) / pixel_size_m
        vx = gaussian_filter(vx, sigma=sigma_px)
        vy = gaussian_filter(vy, sigma=sigma_px)

    # y_sub is descending (shared convention across this package's raster
    # sources); RegularGridInterpolator requires ascending axes.
    order = np.argsort(y_sub)
    y_asc = y_sub[order]
    interp_u = RegularGridInterpolator((y_asc, x_sub), vx[order], bounds_error=False, fill_value=np.nan)
    interp_v = RegularGridInterpolator((y_asc, x_sub), vy[order], bounds_error=False, fill_value=np.nan)

    def velocity_at(points_xy: np.ndarray) -> np.ndarray:
        yx = points_xy[:, ::-1]
        return np.column_stack([interp_u(yx), interp_v(yx)])

    return velocity_at


def _unit_direction(velocity_at, pos: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray]:
    """sign*velocity/|velocity| at each point, plus a mask of where
    that's undefined (NaN from outside the loaded crop, or a true
    stagnation point — data gaps are already filled upstream in
    velocity_field)."""
    v = velocity_at(pos)
    speed = np.linalg.norm(v, axis=1)
    bad = ~np.isfinite(speed) | (speed == 0)
    direction = np.zeros_like(v)
    good = ~bad
    direction[good] = sign * v[good] / speed[good, None]
    return direction, bad


def _rk4_step(velocity_at, pos: np.ndarray, step_m: float, sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One arc-length-parametrized RK4 step along sign*velocity/|velocity|
    (sign=+1 forward with the flow, sign=-1 backward against it): treats
    it as a unit tangent field and integrates dx/ds = tangent(x) by
    step_m in s, using the standard 4-stage weighted average.

    This matters more than it might seem: simple Euler stepping (move
    step_m in the direction sampled once at the start) cuts corners in
    curved flow, which — in an earlier backward-tracing version of this
    module — routinely produced wrong basin attributions rather than
    just imprecise ones. RK4 samples the direction 4 times across the
    step and follows the curve far more faithfully.

    Falls back to plain Euler (k1 only) for a point if a later stage
    (k2/k3/k4) lands somewhere the velocity field is undefined, rather
    than discarding the whole step — only a point where even k1 fails is
    reported as stuck.
    """
    half = step_m / 2.0
    k1, bad1 = _unit_direction(velocity_at, pos, sign)
    k2, bad2 = _unit_direction(velocity_at, pos + half * k1, sign)
    k3, bad3 = _unit_direction(velocity_at, pos + half * k2, sign)
    k4, bad4 = _unit_direction(velocity_at, pos + step_m * k3, sign)

    rk4_direction = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    fall_back_to_euler = (bad2 | bad3 | bad4) & ~bad1
    direction = np.where(fall_back_to_euler[:, None], k1, rk4_direction)

    new_pos = pos + step_m * direction
    stuck = bad1  # only "can't even evaluate the field here" counts as stuck
    new_pos[stuck] = pos[stuck]
    return new_pos, stuck


def trace_forward_paths(shelf_poly, velocity_at, seeds: np.ndarray, step_m: float, max_steps: int) -> list[np.ndarray]:
    """Integrate every seed forward (with the flow) until it leaves
    `shelf_poly`, a step fails, or max_steps runs out. Returns one path
    (array of shape (Ni, 2), including the seed itself) per seed.
    """
    n = len(seeds)
    if n == 0:
        return []
    pos = seeds.copy().astype(float)
    active = np.ones(n, dtype=bool)
    path_points = [[seeds[i].copy()] for i in range(n)]

    for _ in range(max_steps):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break

        new_pos, stuck = _rk4_step(velocity_at, pos[idx], step_m, sign=1.0)
        pos[idx] = new_pos

        pts = shapely.points(new_pos[:, 0], new_pos[:, 1])
        inside = shapely.contains(shelf_poly, pts)

        for local_i, global_i in enumerate(idx):
            if not stuck[local_i]:
                path_points[global_i].append(new_pos[local_i].copy())

        left_shelf = ~inside
        done = stuck | left_shelf
        active[idx[done]] = False

    return [np.array(p) for p in path_points]


def build_streamline_atlas(
    ice_shelf: str, basin_names: list[str], step_m: float, max_steps: int,
    smooth_km: float, min_normal_velocity_myr: float, pad_km: float = 20.0, paths: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Forward-trace every basin's flux-gate seeds across the shelf.
    Returns (all path points stacked, matching basin label per point,
    {basin: n_seeds} for reporting).
    """
    paths = paths if paths is not None else DEFAULT_PATHS
    shelf_poly = boundaries.get_named_polygon(str(paths["iceshelves"]), ice_shelf)
    basin_polys = {n: boundaries.get_named_polygon(str(paths["basins_refined"]), n) for n in basin_names}
    all_bounds = np.array([shelf_poly.bounds] + [p.bounds for p in basin_polys.values()])
    combined_bounds = (
        all_bounds[:, 0].min(), all_bounds[:, 1].min(),
        all_bounds[:, 2].max(), all_bounds[:, 3].max(),
    )
    velocity_at = velocity_field(combined_bounds, pad_km=pad_km, smooth_km=smooth_km, paths=paths)

    all_points = []
    all_labels = []
    seed_counts = {}
    for name in basin_names:
        seeds = flux_gate_seeds(
            ice_shelf, name, pad_km=pad_km, smooth_km=smooth_km,
            min_normal_velocity_myr=min_normal_velocity_myr, paths=paths,
        )
        seed_counts[name] = len(seeds)
        if len(seeds) == 0:
            continue
        for path in trace_forward_paths(shelf_poly, velocity_at, seeds, step_m, max_steps):
            all_points.append(path)
            all_labels.extend([name] * len(path))

    if not all_points:
        raise ValueError(f"No basin produced any flux-gate seeds for {ice_shelf!r} — check min_normal_velocity_myr.")
    return np.concatenate(all_points, axis=0), np.array(all_labels, dtype=object), seed_counts


def partition_shelf(
    ice_shelf: str,
    step_m: float = 1500.0,
    max_steps: int = 1000,
    smooth_km: float = 0.5,
    min_normal_velocity_myr: float = 0.0,
    pad_km: float = 20.0,
    paths: dict | None = None,
) -> tuple[ShelfPartition, list[str], dict]:
    """Label every one of the shelf's velocity data points by nearest-
    neighbor lookup against the streamline atlas. Returns the partition,
    the basin names considered, and their seed counts (0 means that
    basin contributed nothing — no candidate-basin filtering step is
    needed separately, see module docstring).
    """
    paths = paths if paths is not None else DEFAULT_PATHS
    shelf_poly = boundaries.get_named_polygon(str(paths["iceshelves"]), ice_shelf)
    basin_names = all_touching_basins(ice_shelf, paths=paths)
    if not basin_names:
        raise ValueError(f"No basins found touching {ice_shelf!r}")

    atlas_points, atlas_labels, seed_counts = build_streamline_atlas(
        ice_shelf, basin_names, step_m, max_steps, smooth_km, min_normal_velocity_myr, pad_km, paths=paths
    )

    contributing = [n for n in basin_names if seed_counts.get(n, 0) > 0]
    # grounding_zone is required by PipelineConfig but irrelevant here —
    # velocity.load_velocity only uses it to resolve file paths.
    cfg = PipelineConfig(ice_shelf=ice_shelf, grounding_zone=contributing[0], buffer_km=1.0, paths=paths)
    vel = velocity.load_velocity(cfg, shelf_poly)

    tree = cKDTree(atlas_points)
    dist, idx = tree.query(np.column_stack([vel.x, vel.y]))

    return ShelfPartition(x=vel.x, y=vel.y, basin=atlas_labels[idx], distance_m=dist), basin_names, seed_counts


def print_summary(partition: ShelfPartition, basin_names: list[str], seed_counts: dict) -> None:
    n = len(partition.x)
    print(f"{n} shelf points classified (every point gets a label; distance_m is the confidence measure)")
    for name in basin_names:
        count = int((partition.basin == name).sum())
        seeds = seed_counts.get(name, 0)
        if count or seeds:
            print(f"  {name}: {seeds} flux-gate seeds -> {count} points ({100 * count / n:.1f}%)")
    far = partition.distance_m > 20_000.0
    print(f"distance to nearest streamline (km): mean={partition.distance_m.mean()/1000:.1f} median={np.median(partition.distance_m)/1000:.1f} max={partition.distance_m.max()/1000:.1f}")
    print(f"  {int(far.sum())} points ({100 * far.sum() / n:.1f}%) are >20km from any traced streamline — lower-confidence")


def plot_partition(partition: ShelfPartition, basin_names: list[str], ice_shelf: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    colors = distinct_colors(max(len(basin_names), 2))
    for color, name in zip(colors, basin_names):
        mask = partition.basin == name
        if not mask.any():
            continue
        ax.scatter(partition.x[mask], partition.y[mask], s=3, color=color, label=f"{name} (n={mask.sum()})")

    ax.set_aspect("equal")
    ax.set_title(f"{ice_shelf}: shelf provinces by feeding basin (forward streamlines from flux gate)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best", fontsize=8, markerscale=3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ice-shelf", default="Amery")
    parser.add_argument("--step-m", type=float, default=1500.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--smooth-km", type=float, default=0.5,
        help="Gaussian-smooth the velocity field by this length scale before integrating (0 to disable)",
    )
    parser.add_argument(
        "--min-normal-velocity-myr", type=float, default=0.0,
        help="Only seed at flux-gate pixels with at least this much inward flow (m/yr)",
    )
    parser.add_argument("--out-dir", default="joint_xpinn_data/output/basins")
    args = parser.parse_args()

    partition, basin_names, seed_counts = partition_shelf(
        args.ice_shelf, step_m=args.step_m, max_steps=args.max_steps,
        smooth_km=args.smooth_km, min_normal_velocity_myr=args.min_normal_velocity_myr,
    )
    print_summary(partition, basin_names, seed_counts)

    out_dir = Path(args.out_dir)
    plot_partition(partition, basin_names, args.ice_shelf, out_dir / f"{args.ice_shelf}_basin_partition.png")


if __name__ == "__main__":
    main()
