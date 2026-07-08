"""Assemble the two-region (grounded/floating) dataset in the same field
layout as DIFFICE_jax's synthetic MISMIP data (see
/Users/jiapchen/Software/DIFFICE_jax/examples/synthetic_data/data-README.md
and the generator at
PinningPointInversion/Scripts/Synthetic/+ProcessSyntheticData/process.m),
but built from real observational sources instead of an ISSM solution.

Differences from the synthetic convention, and why:
  - mud, alpha2d (viscosity, basal friction) are NaN-filled placeholders.
    In the synthetic case these come from the ISSM ground truth; for real
    data they're the inversion targets, not observations.
  - ols_d is still computed (positive=grounded, negative=floating) but as
    a geometric signed distance to the grounding line, not a levelset PDE
    solution — pure geometry, not a dynamical quantity.
  - xd_h/yd_h legitimately differ from xd/yd (sparse BEDMAP1 radar tracks
    vs. the velocity grid), per the data-README's note that this is
    exactly the "real data" case it anticipates.
  - xd_s/yd_s (surface elevation's own coordinates, paired with sd) are
    not part of DIFFICE_jax's original schema at all — surface elevation
    is sourced independently of thickness rather than resampled onto
    xd_h/yd_h, which means this pipeline's output isn't yet consumable by
    DIFFICE_jax's training code as-is. See
    docs/adr/0001-surface-elevation-independent-coordinates.md.
  - Floating region's xdir/ydir/udir/vdir are empty only when the
    floating region's *entire* boundary is either the grounding line
    (x_md/y_md) or the calving front (xct/yct/nnct). That's true by
    construction for the original synthetic rectangular domain, but isn't
    generally true here: Amery touches 6 basins (American HighLand,
    Fisher, Islands, Lambert, MacRobertson Land, Mellor — see
    utils/basin_partition.py), and every existing Amery config only
    models one of them as its grounded region — so most of the
    floating_region_source="whole_shelf" (default) boundary is neither
    the chosen zone's GL nor the true calving front, but a real
    grounded/floating interface with one of the other 5 basins this
    build isn't separately modeling. That arc now gets a Dirichlet BC
    from real velocity data, the same way the grounded region's own
    arbitrary buffer-radius truncation already does — not new physics,
    just no longer silently zeroed out. `floating_region_source`
    restricting the region to a corridor (see floating_region.py) adds a
    second, similar kind of cut: the corridor's own margin_km edge.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import shapely
from scipy.spatial import cKDTree

from joint_xpinn_data.config import PipelineConfig, load_config
from joint_xpinn_data.contracts import Geometry, PointObservations
from joint_xpinn_data.data_sources import surface, thickness, velocity
from joint_xpinn_data.domain import DomainRegions, build_regions
from joint_xpinn_data.resample import nearest_sample
from joint_xpinn_data.utils import collocation
from joint_xpinn_data.utils.raster_utils import boundary_indices, largest_connected_component

REGION_LABELS = ["grounded", "floating"]
BASAL_MASK = [1, 0]

_SECONDS_PER_YEAR = 365.25 * 86400.0

EXPECTED_SAVE_VARIABLES = [
    "xd", "yd", "ud", "vd", "xd_h", "yd_h", "hd", "xd_s", "yd_s", "sd",
    "h_dense", "s_dense", "xcol", "ycol", "xdir", "ydir", "udir", "vdir",
    "xct", "yct", "nnct", "x_md", "y_md", "Xe", "Ye", "Xe_h", "Ye_h",
    "idxcrop", "idxcrop_h", "basal_mask", "mud", "alpha2d", "ols_d",
]


def _signed_distance_to_gl(x: np.ndarray, y: np.ndarray, gl_points: np.ndarray, sign: float) -> np.ndarray:
    tree = cKDTree(gl_points)
    dist, _ = tree.query(np.column_stack([x, y]))
    return sign * dist


def _filter_by_speed(vel: PointObservations, min_speed_myr: float) -> PointObservations:
    """Drop points whose speed (u,v are stored in m/s, see velocity.py)
    is at or below `min_speed_myr`."""
    speed_myr = np.hypot(vel.values["u"], vel.values["v"]) * _SECONDS_PER_YEAR
    keep = speed_myr > min_speed_myr
    return PointObservations(
        x=vel.x[keep],
        y=vel.y[keep],
        values={k: v[keep] for k, v in vel.values.items()},
        weight=vel.weight[keep] if vel.weight is not None else None,
        product=vel.product,
        epoch=vel.epoch,
    )


def _seaward_of_front(
    x: np.ndarray, y: np.ndarray, front: Geometry, grounding_line: Geometry, band_m: float
) -> np.ndarray:
    """Boolean: True wherever (x,y) sits on the open-ocean side of the
    calving front *and* within `band_m` of it. Shared by `_filter_by_front`
    (a velocity point cloud) and `_accepted_velocity_mask` (a velocity
    grid's own pixel centers, for the raster Dirichlet mask) — same test,
    two different point sets.

    The land->sea direction at each front vertex is taken as the direction
    away from the nearest grounding-line point, NOT the front's own stored
    outward normal. The GL is unambiguously the landward interface, so this
    orientation is globally consistent; the per-vertex outward normal is not
    trustworthy here, because when the front sits several km landward of the
    floating polygon's own seaward edge (exactly the case that triggers this
    erosion — see domain._reconcile_front_with_velocity), many front points
    lie *inside* the polygon, where a local inside/outside probe can't orient
    a normal at all (confirmed for Amery/Lambert: front points up to ~7.9km
    inside, 94/839 normals flipped landward). `band_m` then bounds the drop
    to the strip actually between the front and the velocity-coverage edge,
    so any residual mis-orientation can't reach deep-interior points (the
    previous unbounded, normal-based test dropped 18,352 Amery points with a
    median distance of 30km from the front)."""
    front_points = front.all_points()
    gl_points = grounding_line.all_points()
    # unit land->sea direction at each front vertex (away from nearest GL)
    _, gidx = cKDTree(gl_points).query(front_points)
    sea_dir = front_points - gl_points[gidx]
    norm = np.linalg.norm(sea_dir, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    sea_dir = sea_dir / norm

    tree = cKDTree(front_points)
    query_xy = np.column_stack([x, y])
    dist, idx = tree.query(query_xy)
    signed = np.einsum("ij,ij->i", query_xy - front_points[idx], sea_dir[idx])
    # Seaward side (signed > 0) AND genuinely near the front (within band_m
    # by true distance, not just the seaward projection — a point far to the
    # *side* of a front vertex can still project to a small positive value).
    return (signed > 0.0) & (dist <= band_m)


def _filter_by_front(
    vel: PointObservations, front: Geometry, grounding_line: Geometry, band_m: float
) -> PointObservations:
    """Drop points on the open-ocean side of the calving front — used when
    domain.build_regions (`_reconcile_front_with_velocity`) found the front
    sitting landward of the velocity data's own coverage: the front is
    trusted as the true edge in that case, so velocity beyond it is
    excluded rather than kept as real floating-ice motion."""
    keep = ~_seaward_of_front(vel.x, vel.y, front, grounding_line, band_m)
    return PointObservations(
        x=vel.x[keep],
        y=vel.y[keep],
        values={k: v[keep] for k, v in vel.values.items()},
        weight=vel.weight[keep] if vel.weight is not None else None,
        product=vel.product,
        epoch=vel.epoch,
    )


def _accepted_velocity_mask(config: PipelineConfig, regions: DomainRegions, polygon, is_grounded: bool):
    """Boolean grid of exactly which MEaSURES velocity pixels are accepted
    into this region's xd/yd/ud/vd: CNT>0, inside `polygon`, and the same
    additional region-specific filter `_build_region` applies to the
    point cloud below (grounded_min_speed_myr's threshold for the grounded
    region; the front-seaward exclusion for the floating region, when
    `front_erosion_band_m` is set) — as a raster rather than a
    point cloud, so the TRUE pixel-adjacency boundary of that exact
    accepted set can be found (`_dirichlet_from_mask`), rather than a
    boundary derived from `polygon`'s own geometric shape.

    The two can disagree substantially. Confirmed for Byrd: the
    velocity-thresholded grounded corridor's own polygon edge sits up to
    `corridor_margin_km` beyond the real fast-flow contour it's traced
    from, in ice that's often genuinely slower than
    `grounded_min_speed_myr` — so a geometric-ring-based cut boundary
    (the polygon's own outline, searched for nearby real velocity within
    a fixed radius) found real *accepted* velocity near only ~11% of its
    candidate points (33/307), even though 100% had real *unfiltered*
    velocity nearby. The mask boundary computed here sits exactly where
    the accepted region truly ends, by construction.
    """
    x_sub, y_sub, u, v, cnt, inside = velocity.load_velocity_grid(config, polygon)
    accepted = (cnt > 0) & inside
    seaward_grid = None
    if is_grounded and config.grounded_min_speed_myr is not None:
        speed_myr = np.hypot(u, v) * _SECONDS_PER_YEAR
        accepted &= speed_myr > config.grounded_min_speed_myr
    elif not is_grounded and regions.front_erosion_band_m > 0.0:
        xx, yy = np.meshgrid(x_sub, y_sub)
        seaward_grid = _seaward_of_front(
            xx.ravel(), yy.ravel(), regions.calving_front, regions.grounding_line, regions.front_erosion_band_m
        ).reshape(xx.shape)
        accepted &= ~seaward_grid
    # The open-sea side, as the largest connected no-data (CNT==0) blob — so
    # _dirichlet_from_mask can tell an ocean-facing edge (the calving front,
    # which gets its own xct/yct BC and must NOT be a Dirichlet cut) from a
    # grounded-ice-facing lateral margin (a real unmodeled interface that
    # legitimately does). Grounded ice has CNT>0 so it's never in this blob;
    # interior data gaps (ice rises) are smaller components, so they keep
    # their own Dirichlet ring. The front-erosion band (points dropped just
    # seaward of the front — CNT>0, so not no-data) is folded in first, so it
    # doesn't sit between the accepted edge and the ocean and hide that the
    # edge is really front-facing.
    sea = cnt == 0
    if seaward_grid is not None:
        sea = sea | (seaward_grid & (cnt > 0) & inside)
    outer_ocean = largest_connected_component(sea)
    return x_sub, y_sub, u, v, accepted, outer_ocean


def _dirichlet_from_mask(
    x_sub: np.ndarray, y_sub: np.ndarray, u: np.ndarray, v: np.ndarray, accepted: np.ndarray,
    exclude_near: list[Geometry], outer_ocean: np.ndarray | None = None, gl_exclusion_km: float = 5.0,
):
    """Real (x,y,u,v) at the true pixel-adjacency boundary of `accepted`
    (an accepted pixel with at least one rejected-or-outside-the-crop
    neighbor — see `utils.raster_utils.boundary_indices`), excluding:

      - any boundary pixel adjacent to `outer_ocean` (the open sea) — that
        edge is the calving front (or the ice/ocean edge generally), a
        physical boundary reported as xct/yct, not an arbitrary truncation.
        This catches the whole ocean-facing edge even where the mapped front
        source has coverage gaps, while leaving lateral grounded-ice-facing
        margins (a real unmodeled grounded/floating interface) as Dirichlet;
      - any boundary pixel within `gl_exclusion_km` of a real interface in
        `exclude_near` (the grounding line, and the mapped calving front) —
        also reported in their own right (x_md/y_md, xct/yct/nnct).
    """
    rows, cols = boundary_indices(accepted, ~accepted)
    if outer_ocean is not None and len(rows):
        orows, ocols = boundary_indices(accepted, outer_ocean)
        ocean_facing = set(zip(orows.tolist(), ocols.tolist()))
        keep = np.array([(r, c) not in ocean_facing for r, c in zip(rows.tolist(), cols.tolist())], dtype=bool)
        rows, cols = rows[keep], cols[keep]
    if len(rows) == 0:
        z = np.zeros(0)
        return z, z, z, z
    bx, by = x_sub[cols], y_sub[rows]
    bu, bv = u[rows, cols], v[rows, cols]
    exclude_pts = np.concatenate([g.all_points() for g in exclude_near], axis=0)
    exclude_multi = shapely.multipoints(shapely.points(*exclude_pts.T))
    dist = shapely.distance(shapely.points(bx, by), exclude_multi)
    keep = dist > gl_exclusion_km * 1000.0
    return bx[keep], by[keep], bu[keep], bv[keep]


def _build_region(config: PipelineConfig, regions: DomainRegions, polygon, is_grounded: bool) -> dict:
    vel = velocity.load_velocity(config, polygon)
    if is_grounded and config.grounded_min_speed_myr is not None:
        vel = _filter_by_speed(vel, config.grounded_min_speed_myr)
    elif not is_grounded and regions.front_erosion_band_m > 0.0:
        vel = _filter_by_front(vel, regions.calving_front, regions.grounding_line, regions.front_erosion_band_m)
    dense_thick = thickness.load_dense_thickness(config, polygon)
    sparse_thick = thickness.load_sparse_thickness(config, polygon)
    dense_surf = surface.load_dense_surface(config, polygon)
    sparse_surf = surface.load_sparse_surface(config, polygon)

    xd, yd = vel.x, vel.y
    ud, vd = vel.values["u"], vel.values["v"]

    h_dense = nearest_sample(xd, yd, dense_thick)["thickness"]
    s_dense = nearest_sample(xd, yd, dense_surf)["surface"]

    # bedmachine_v3 is a dense grid, not a genuinely sparse survey — it has
    # no independent "native sparse points" distinct from its own dense
    # grid, unlike bedmap1_csv/bedmap2_csv's real, independently-located
    # radar tracks. Using it for the sparse role while keeping its own
    # native BedMachine-grid points (a different grid than xd/yd) made
    # dense/sparse visibly different even when both roles pointed at the
    # exact same source — resample onto (xd,yd) like the dense role
    # instead, so the two are consistent.
    if config.sparse_thickness_source == "bedmachine_v3":
        xd_h, yd_h = xd, yd
        hd = nearest_sample(xd, yd, sparse_thick)["thickness"]
    else:
        xd_h, yd_h, hd = sparse_thick.x, sparse_thick.y, sparse_thick.values["thickness"]

    # xd_s/yd_s are surface elevation's own native points, independent of
    # xd_h/yd_h — not resampled onto the thickness grid, see
    # docs/adr/0001-surface-elevation-independent-coordinates.md. Even when
    # sparse_thickness_source and sparse_surface_source are the same file
    # (e.g. both bedmap1_csv), xd_h and xd_s can differ in count and in
    # which rows survive: each column is sentinel-filtered independently
    # (data_sources/thickness.py's process_bedmap1_column), so a row with
    # valid thickness but a sentinel surface value (or vice versa) is kept
    # for one and dropped for the other. Same bedmachine_v3 special case as
    # thickness above, for the same reason.
    if config.sparse_surface_source == "bedmachine_v3":
        xd_s, yd_s = xd, yd
        sd = nearest_sample(xd, yd, sparse_surf)["surface"]
    else:
        xd_s, yd_s, sd = sparse_surf.x, sparse_surf.y, sparse_surf.values["surface"]

    # Collocation library (xcol/ycol). By default it is the velocity data
    # points (xd/yd). When the config sets a `collocation` block, sample an
    # independent, density-controlled two-tier Halton library inside this
    # region instead (denser near the GL/front) — see utils/collocation.py.
    coll_settings = collocation.collocation_settings(config)
    if coll_settings is None:
        xcol, ycol = xd, yd
    else:
        xcol, ycol = collocation.sample_collocation(regions, polygon, is_grounded, coll_settings)

    gl_points = regions.grounding_line.all_points()
    ols_d = _signed_distance_to_gl(xd, yd, gl_points, sign=1.0 if is_grounded else -1.0)

    mud = np.full(xd.shape, np.nan)
    alpha2d = np.full(xd.shape, np.nan)

    # Dirichlet BC from the true pixel-adjacency edge of the accepted-velocity
    # raster (_accepted_velocity_mask / _dirichlet_from_mask), for every
    # region strategy — it sits exactly where the accepted data ends, rather
    # than on the region polygon's own geometric ring, which can diverge from
    # the data mask by tens of km (confirmed for Amery/Lambert: the ring
    # boundary sat a median 17.5km from the fast-ice data it's meant to
    # bound). See docs/adr/0002-*.md.
    exclude = [regions.grounding_line] if is_grounded else [regions.grounding_line, regions.calving_front]
    x_sub, y_sub, gu, gv, accepted, outer_ocean = _accepted_velocity_mask(config, regions, polygon, is_grounded)
    xdir, ydir, udir, vdir = _dirichlet_from_mask(x_sub, y_sub, gu, gv, accepted, exclude, outer_ocean=outer_ocean)

    # xct/yct must be column vectors (N, 1), matching MISMIP's MATLAB-authored
    # convention: DIFFICE_jax's preprocessing hstacks them directly (unlike
    # xd/yd/etc., which it flattens before use), so a (1, N) row vector —
    # scipy.io.savemat's default orientation for a 1-D array — silently
    # produces a garbled (1, 2N) X_ct instead of the intended (N, 2).
    if is_grounded:
        xct, yct, nnct = np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 2))
    else:
        segs, normals = regions.calving_front.segments, regions.calving_front_normals
        xct = (np.concatenate([s[:, 0] for s in segs]) if segs else np.zeros(0)).reshape(-1, 1)
        yct = (np.concatenate([s[:, 1] for s in segs]) if segs else np.zeros(0)).reshape(-1, 1)
        nnct = np.concatenate(normals, axis=0) if normals else np.zeros((0, 2))

    return dict(
        xd=xd, yd=yd, ud=ud, vd=vd, xd_h=xd_h, yd_h=yd_h, hd=hd,
        xd_s=xd_s, yd_s=yd_s, sd=sd, h_dense=h_dense, s_dense=s_dense,
        xcol=xcol, ycol=ycol,
        xdir=xdir, ydir=ydir, udir=udir, vdir=vdir,
        xct=xct, yct=yct, nnct=nnct, mud=mud, alpha2d=alpha2d, ols_d=ols_d,
    )


def build_dataset(config: PipelineConfig) -> dict:
    regions = build_regions(config)
    per_region = [
        _build_region(config, regions, regions.grounded_polygon, True),
        _build_region(config, regions, regions.floating_polygon, False),
    ]

    cell_fields = [
        "xd", "yd", "ud", "vd", "xd_h", "yd_h", "hd", "xd_s", "yd_s", "sd",
        "h_dense", "s_dense", "xcol", "ycol", "xdir", "ydir", "udir", "vdir",
        "xct", "yct", "nnct", "mud", "alpha2d", "ols_d",
    ]
    data = {f: [r[f] for r in per_region] for f in cell_fields}

    n_dense = [len(r["xd"]) for r in per_region]
    n_thick = [len(r["xd_h"]) for r in per_region]
    cum_dense = np.concatenate([[0], np.cumsum(n_dense)])
    cum_thick = np.concatenate([[0], np.cumsum(n_thick)])

    Xe = np.concatenate([r["xd"] for r in per_region])
    Ye = np.concatenate([r["yd"] for r in per_region])
    Xe_h = np.concatenate([r["xd_h"] for r in per_region])
    Ye_h = np.concatenate([r["yd_h"] for r in per_region])

    idxcrop = [
        np.array([1, 1, cum_dense[i] + 1, cum_dense[i + 1]], dtype=np.uint16) for i in range(2)
    ]
    idxcrop_h = [
        np.array([1, 1, cum_thick[i] + 1, cum_thick[i + 1]], dtype=np.uint16) for i in range(2)
    ]

    gl_points = regions.grounding_line.all_points()
    # x_md/y_md must be object-dtype "cell" arrays (one cell per interface,
    # matching MISMIP's MATLAB-authored convention) so DIFFICE_jax's loader
    # indexes them as data['x_md'][0, idx]. A plain list of one array here
    # would let numpy silently stack it into a regular (1, N) float array
    # instead of a cell, since there's only ever one interface for a
    # 2-region config and numpy only falls back to dtype=object when a
    # list's elements have inconsistent shapes (true for every *other*
    # per-region cell field here, which lists one array per region).
    x_md = np.empty((1, 1), dtype=object)
    x_md[0, 0] = gl_points[:, 0].reshape(-1, 1)
    y_md = np.empty((1, 1), dtype=object)
    y_md[0, 0] = gl_points[:, 1].reshape(-1, 1)
    region_masks = []
    for i in range(2):
        m = np.zeros(len(Xe), dtype=bool)
        m[cum_dense[i]:cum_dense[i + 1]] = True
        region_masks.append(m)

    options = dict(
        ice_shelf=config.ice_shelf,
        grounding_zone=config.grounding_zone,
        buffer_km=config.buffer_km,
        velocity_source=config.velocity_source,
        dense_thickness_source=config.dense_thickness_source,
        sparse_thickness_source=config.sparse_thickness_source,
        dense_surface_source=config.dense_surface_source,
        sparse_surface_source=config.sparse_surface_source,
        grounding_line_source=config.grounding_line_source,
        calving_front_source=config.calving_front_source,
        floating_region_source=config.floating_region_source,
        paths={k: str(v) for k, v in config.paths.items()},
    )

    return dict(
        **data,
        x_md=x_md,
        y_md=y_md,
        Xe=Xe, Ye=Ye, Xe_h=Xe_h, Ye_h=Ye_h,
        idxcrop=idxcrop, idxcrop_h=idxcrop_h,
        basal_mask=np.array(BASAL_MASK, dtype=float),
        region_labels=REGION_LABELS,
        region_masks=region_masks,
        options=options,
        solution_source=(
            f"Real observational data: velocity={config.velocity_source}, "
            f"dense_thickness={config.dense_thickness_source}, "
            f"sparse_thickness={config.sparse_thickness_source}, "
            f"dense_surface={config.dense_surface_source}, "
            f"sparse_surface={config.sparse_surface_source}, "
            f"grounding_line={config.grounding_line_source}, "
            f"calving_front={config.calving_front_source}"
        ),
        thickness_sampling_pattern="real_radar",
        expected_save_variables=EXPECTED_SAVE_VARIABLES,
    )


def save_dataset(config: PipelineConfig, out_path: str) -> dict:
    output = build_dataset(config)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_path, output)
    return output


def config_stem(config: PipelineConfig) -> str:
    """Canonical name for one config's build: `<ice_shelf>_<grounding_zone>_<buffer_km>km`.

    Names the output `.mat`, its output folder, and its figures — the same
    stem regardless of whether the config came from a saved YAML or a
    one-off programmatic call.
    """
    buffer_str = f"{config.buffer_km:g}km"
    return f"{config.ice_shelf}_{config.grounding_zone}_{buffer_str}"


def config_output_dir(config: PipelineConfig, out_root: str) -> Path:
    """Per-config output folder: `<out_root>/<config_stem>/`, holding the
    built `.mat` plus a `figures/` subfolder for anything scoped to this
    (ice_shelf, grounding_zone, buffer_km) triple (plot_validation,
    compare_sources, checks)."""
    return Path(out_root) / config_stem(config)


def default_output_path(config: PipelineConfig, out_dir: str) -> Path:
    return config_output_dir(config, out_dir) / f"{config_stem(config)}.mat"


def build_from_config_file(config_path: Path, out_dir: str) -> Path:
    config = load_config(config_path)
    out_path = default_output_path(config, out_dir)
    print(f"Building {config.ice_shelf}/{config.grounding_zone} (buffer_km={config.buffer_km}) from {config_path} ...")
    save_dataset(config, str(out_path))
    print(f"  saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a two-region training dataset from one YAML config, "
        "or every *.yaml config in a directory."
    )
    parser.add_argument("config", help="Path to a config .yaml file, or a directory of them (see data_build_configs/TEMPLATE.yaml)")
    parser.add_argument("--out-dir", default="joint_xpinn_data/output", help="Directory to save the .mat file(s) in")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.is_dir():
        yaml_files = sorted(config_path.glob("*.yaml"))
        yaml_files = [p for p in yaml_files if p.name != "TEMPLATE.yaml"]
        if not yaml_files:
            raise SystemExit(f"No *.yaml configs found in {config_path} (TEMPLATE.yaml is skipped).")
        for p in yaml_files:
            build_from_config_file(p, args.out_dir)
    else:
        build_from_config_file(config_path, args.out_dir)


if __name__ == "__main__":
    main()
