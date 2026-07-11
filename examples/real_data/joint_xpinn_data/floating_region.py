"""Floating-region shaping: how far the floating region extends beyond
the grounded corridor.

`process_whole_shelf` is the default and matches every existing config's
behavior exactly: the floating region is the named ice shelf's full
extent minus the grounded corridor, regardless of `buffer_km` or which
grounding zone was chosen. Fine for a moderate shelf fed by a few
tributaries (Amery); breaks down for a shelf as large as Ross East, fed
by many outlet glaciers, where a single grounding zone's real flow only
occupies a small fraction of the whole shelf polygon (confirmed for
Byrd/Ross East: the fast confluence area is one corner of the shelf, the
rest is hundreds of km of plain fed by unrelated glaciers).

`process_flowline_corridor` and `process_basin_partition` restrict the
floating region to a corridor around `config.grounding_zone`'s own flow,
reusing `utils.basin_partition`'s forward-streamline-tracing machinery
(built for a different diagnostic purpose) unmodified — see their
docstrings for the tradeoff between the two.

Every registered function returns `(polygon, front_reference_points)`,
not just a polygon: `domain._trim_front_to_region` needs a tight
reference for where the *real* calving front should be near a restricted
corridor — the corridor's own boundary is only an approximation (buffered
traced streamlines, not a digitized coastline) and includes lateral
margin_km edges that a generic distance-to-boundary tolerance can bleed
through. `front_reference_points` is `None` for `whole_shelf` (the
generic, more generous polygon-boundary tolerance is used instead) and
for `process_basin_partition` (no natural per-point "distance from GL"
concept without deeper changes — a known gap, see HANDOFF.md).
"""

import warnings

import numpy as np
import shapely

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.utils import basin_partition

# A fixed margin_km bounds gaps *along* one traced streamline (point
# spacing is small relative to margin_km), but says nothing about gaps
# *across* diverging streamlines — a quantity that grows with downstream
# distance and how fast the flow fans out, and isn't bounded by anything
# in basin_partition.trace_forward_paths. This needs to be impossible to
# miss, same rationale as utils.geometry_utils's UNORDERED GEOMETRY banner.
_CORRIDOR_WARNING_BANNER = "!" * 78

# grounded_polygon and shelf_polygon (independently digitized —
# basins_refined_v2.mat vs iceshelves_2008_v2.mat) cross each other
# repeatedly over short spans near the coast rather than along one clean
# shared arc (see domain._GROUNDED_POLYGON_SMOOTHING_M, which closes this
# at the grounded_polygon level); a restricted corridor's own buffered-
# streamline boundary interacting with that same misalignment can still
# carve a small notch, near-pinch, or split off a sliver too thin to stay
# connected — confirmed for Byrd: a 4.2 km^2 island and a ~15km notch
# right at the near-GL end of the corridor, where flow-fan-out and the
# tightest taper radius meet the coastline detail. Closing removes
# sub-3km-scale artifacts like this; keeping only the dominant component
# drops slivers a fixed margin_km leaves disconnected. Real, large-scale
# fragmentation (this basin's flow genuinely splitting into separate
# lobes downstream) is still caught first by
# _warn_if_corridor_looks_broken, before this cleanup silently keeps only
# one of them.
_FLOATING_POLYGON_SMOOTHING_M = 3000.0


def _clean_floating_polygon(polygon):
    """Close small-scale boundary noise (see _FLOATING_POLYGON_SMOOTHING_M)
    and keep only the dominant connected component — applied uniformly to
    every floating_region_source's output, not just the restricted-corridor
    strategies, so whole_shelf gets the same treatment (a no-op for a
    already-single-piece, already-smooth shelf polygon like Amery's)."""
    closed = polygon.buffer(_FLOATING_POLYGON_SMOOTHING_M).buffer(-_FLOATING_POLYGON_SMOOTHING_M)
    parts = list(closed.geoms) if closed.geom_type == "MultiPolygon" else [closed]
    return max(parts, key=lambda p: p.area)


def process_whole_shelf(config: PipelineConfig, shelf_polygon, grounded_polygon, **_unused):
    """Today's default: the named shelf's full extent minus the grounded
    corridor. Independent of buffer_km and of which grounding_zone's
    corridor gets subtracted — see module docstring.

    Accepts and ignores arbitrary kwargs (`**_unused`) because
    `domain.process_flow_restricted` unconditionally injects
    `min_normal_velocity_myr` into every strategy's `floating_region_kwargs`
    regardless of which `floating_region_source` is selected — this has no
    seeding concept to apply that to, unlike `process_flowline_corridor`/
    `process_basin_partition`, but should still compose with
    `region_strategy: flow_restricted` rather than raising a TypeError."""
    return shelf_polygon.difference(grounded_polygon), None


def _warn_if_corridor_looks_broken(polygon, ice_shelf: str, grounding_zone: str) -> None:
    parts = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    has_hole = any(len(p.interiors) > 0 for p in parts)
    fragmented = False
    if len(parts) > 1:
        areas = sorted((p.area for p in parts), reverse=True)
        fragmented = areas[1] > 0.1 * areas[0]  # no single dominant component
    if not (has_hole or fragmented):
        return
    problems = []
    if has_hole:
        problems.append("interior holes")
    if fragmented:
        problems.append("multiple disconnected components with no single dominant one")
    warnings.warn(
        f"\n{_CORRIDOR_WARNING_BANNER}\n"
        f"{ice_shelf}/{grounding_zone}: the restricted floating-region corridor has "
        f"{' and '.join(problems)} — margin_km is likely too small for how far this "
        "basin's flow fans out downstream. Inspect with plot_validation before "
        "trusting this build.\n"
        f"{_CORRIDOR_WARNING_BANNER}",
        stacklevel=3,
    )


def _corridor_from_points(points: np.ndarray, margin_m, shelf_polygon, grounded_polygon, config: PipelineConfig):
    """`margin_m` (meters) may be a scalar (uniform buffer, equivalent to
    buffering the whole point cloud at once) or an array matching
    `points` (a per-point radius, e.g. a taper) — `shapely.buffer`
    broadcasts either way; the union of individually-buffered points is
    the same set as buffering the whole cloud at once when the radius is
    uniform, so this is a strict generalization, not a behavior change
    for the uniform case."""
    pts = shapely.points(points[:, 0], points[:, 1])
    corridor = shapely.union_all(shapely.buffer(pts, margin_m))
    restricted = shelf_polygon.difference(grounded_polygon).intersection(corridor)
    _warn_if_corridor_looks_broken(restricted, config.ice_shelf, config.grounding_zone)
    return restricted


def trace_grounding_zone_flow(
    config: PipelineConfig, shelf_polygon,
    step_m: float = 1500.0, max_steps: int = 1000, smooth_km: float = 0.5,
    min_normal_velocity_myr: float = 0.0, pad_km: float = 20.0,
):
    """The expensive half of `process_flowline_corridor`: flux-gate seeds
    (real inward flow at the grounded/floating interface —
    `basin_partition.flux_gate_seeds`, unmodified) + velocity field +
    forward-traced streamlines (`basin_partition.trace_forward_paths`)
    for `config.grounding_zone` only. Paths are integrated until they
    leave the shelf polygon, so they naturally reach the real calving
    front rather than stopping short of it.

    Separated out from the (cheap) taper/corridor-building step
    (`taper_margins_m`/`_corridor_from_points`) so parameter tuning
    (`margin_km`, `min_margin_km`, `taper_km`) can reuse one trace across
    many trials instead of re-paying this cost — dominated by
    `basin_partition.velocity_field`'s gap-filling/Gaussian-smoothing
    over a crop sized to the *whole shelf's* bounds, ~13-15 minutes for a
    shelf as large as Ross East — every time.

    Returns `(all_points, distance_from_gl_m, exit_points)`:
    - `all_points`: every recorded point across every seed's path, stacked.
    - `distance_from_gl_m`: each point's own downstream arc-length
      distance from its seed — every recorded step moves exactly `step_m`
      (`basin_partition._rk4_step`), so this is `index * step_m` within
      each path, not an approximation.
    - `exit_points`: each path's own last recorded point — exactly where
      that streamline left `shelf_polygon` — the tightest available
      reference for "where this corridor's flow actually reaches the real
      edge" (see `domain._trim_front_to_region`).
    """
    seeds = basin_partition.flux_gate_seeds(
        config.ice_shelf, config.grounding_zone, pad_km=pad_km, smooth_km=smooth_km,
        min_normal_velocity_myr=min_normal_velocity_myr, paths=config.paths,
    )
    if len(seeds) == 0:
        raise ValueError(
            f"No flux-gate seeds found for {config.grounding_zone!r} on {config.ice_shelf!r} "
            "— check min_normal_velocity_myr, or that this basin actually has real "
            "inflow at the grounded/floating interface (see basin_partition.py's module "
            "docstring: a basin can share a long boundary with a shelf's basin polygon "
            "without having any real inflow at all)."
        )
    velocity_at = basin_partition.velocity_field(
        shelf_polygon.bounds, pad_km=pad_km, smooth_km=smooth_km, paths=config.paths
    )
    paths = basin_partition.trace_forward_paths(shelf_polygon, velocity_at, seeds, step_m, max_steps)
    all_points = np.concatenate(paths, axis=0)
    distance_from_gl_m = np.concatenate([np.arange(len(p)) * step_m for p in paths])
    exit_points = np.array([p[-1] for p in paths if len(p) > 0])
    return all_points, distance_from_gl_m, exit_points


def taper_margins_m(
    distance_from_gl_m: np.ndarray, margin_km: float, min_margin_km: float = 2.0, taper_km: float | None = None,
) -> np.ndarray:
    """Per-point buffer radius (meters): a `min_margin_km`-wide core that
    sits strictly around the traced flow lines themselves, plus a band
    that ramps up to the full `margin_km` over `taper_km` of downstream
    distance and then holds there as a plateau, following a **smoothstep**
    `3t^2 - 2t^3` in `t = clip(distance/taper_km, 0, 1)`.

    Smoothstep has zero derivative at *both* ends (`t=0` and `t=1`) and
    reaches the full `margin_km` exactly at `distance=taper_km` (a genuine
    plateau), so the widening eases *in* near the grounding line (concave
    up, opening gradually from `min_margin_km` — matching a real confluence
    that is narrow at the GL) and eases *out* into the maximum width
    (concave down, reaching the plateau tangentially). Preferred over a
    linear ramp clipped at `taper_km` (a real derivative kink at both ends)
    and over `tanh(3*distance/taper_km)` (whose derivative is *maximal*
    right at the GL, so the corridor flares out fastest there, and which
    never fully reaches a hard plateau).

    NB the taper controls only the buffer *radius* over the first
    `taper_km` from the GL; it does not control the corridor's downstream
    width once the buffer is at plateau, nor where the buffer envelope
    intersects the shelf boundary. For Byrd, that intersection (not the
    taper) produced a sharp kink "at the end of the widening portion", and
    the fix was a uniform, narrower corridor (`taper_km=0`, small
    `margin_km`) so the buffer never reaches the shelf boundary at all —
    which also bypasses this taper entirely. See HANDOFF.md. `min_margin_km`
    keeps individually-buffered seeds' circles overlapping near the
    grounding line (flux-gate seeds are ~500m apart) so the corridor stays
    connected to `grounded_polygon` rather than tapering all the way to a
    gap. `taper_km=0` (or any value <= 0) disables the ramp: the radius is
    a flat `margin_km` everywhere, giving a constant-width corridor.
    """
    taper_m = (taper_km if taper_km is not None else margin_km) * 1000.0
    if taper_m <= 0:
        taper_fraction = np.ones_like(distance_from_gl_m)
    else:
        t = np.clip(distance_from_gl_m / taper_m, 0.0, 1.0)
        taper_fraction = t * t * (3.0 - 2.0 * t)
    return min_margin_km * 1000.0 + (margin_km - min_margin_km) * 1000.0 * taper_fraction


def _split_paths(points, distance_from_gl_m, *aligned):
    """Undo `trace_grounding_zone_flow`'s `np.concatenate`: split `points`
    (and any other per-point array of the same length, e.g. `margins_m`)
    back into one segment per traced streamline, using each streamline's
    own `distance_from_gl_m` restart at 0 as the path boundary."""
    starts = np.flatnonzero(distance_from_gl_m == 0.0)
    ends = np.append(starts[1:], len(points))
    return [(points[s:e], *(a[s:e] for a in aligned)) for s, e in zip(starts, ends)]


def _prune_streamlines_near_hole(restricted, paths_with_margins, prune_km, shelf_polygon, grounded_polygon, config):
    """Drop every whole streamline that comes within `prune_km` of
    `restricted`'s own interior hole(s) and rebuild the corridor from
    what's left, then return `(rebuilt_polygon, kept_paths)`.

    For Amery/Lambert, the corridor's interior hole is a real obstacle
    (confirmed against BedMachine's mask: a rock outcrop/ice rise, not open
    shelf) that the traced flow genuinely diverges around — but the hole's
    *shape* is a construction artifact: the streamlines that pass closest
    to the obstacle are exactly what buffers into its ragged inner rim,
    since `margin_km` isn't wide enough to bridge the gap between the two
    diverging lobes cleanly. Dropping those obstacle-hugging streamlines
    entirely (not just the points nearest the hole — a fixed `margin_km`
    buffer around a truncated streamline would just recreate the same
    ragged edge one step further out) and re-buffering only the remaining,
    farther-away streamlines pulls the boundary back to a cleaner, rounder
    exclusion. A no-op (returns the input unchanged) if the corridor has no
    interior hole, or nothing is within `prune_km` of one.
    """
    parts = list(restricted.geoms) if restricted.geom_type == "MultiPolygon" else [restricted]
    hole_rings = [ring for p in parts for ring in p.interiors]
    paths = [pts for pts, _ in paths_with_margins]
    if not hole_rings:
        return restricted, paths
    hole_zone = shapely.union_all([shapely.Polygon(r) for r in hole_rings])

    prune_m = prune_km * 1000.0
    kept = [
        (pts, m) for pts, m in paths_with_margins
        if shapely.distance(shapely.points(pts[:, 0], pts[:, 1]), hole_zone).min() >= prune_m
    ]
    if len(kept) == len(paths_with_margins):
        return restricted, paths

    kept_points = np.concatenate([pts for pts, _ in kept], axis=0)
    kept_margins = np.concatenate([m for _, m in kept], axis=0)
    rebuilt = _corridor_from_points(kept_points, kept_margins, shelf_polygon, grounded_polygon, config)
    return rebuilt, [pts for pts, _ in kept]


def process_flowline_corridor(
    config: PipelineConfig, shelf_polygon, grounded_polygon, margin_km: float,
    step_m: float = 1500.0, max_steps: int = 1000, smooth_km: float = 0.5,
    min_normal_velocity_myr: float = 0.0, pad_km: float = 20.0,
    min_margin_km: float = 2.0, taper_km: float | None = None,
    prune_near_hole_km: float | None = None,
):
    """Trace forward streamlines from only `config.grounding_zone`'s own
    flow (`trace_grounding_zone_flow`), taper the buffer radius with
    downstream distance (`taper_margins_m`), and intersect with the
    whole-shelf floating region.

    Cheaper than `process_basin_partition` (skips every other basin's
    seed-tracing, and — more importantly — its velocity-field crop is
    sized to this shelf alone, not the union of every touching basin's
    own bounding box) but less principled at the corridor's lateral
    edges: `margin_km` is a free parameter standing in for "how far this
    basin's real territory extends beyond its own traced centerline,"
    not derived from competition against neighboring basins' flow the
    way `process_basin_partition` is.

    `prune_near_hole_km` (None by default, so every other config's
    behavior is unchanged): an Amery/Lambert-specific escape hatch for an
    interior hole in the corridor — see `_prune_streamlines_near_hole`.
    """
    all_points, distance_from_gl_m, exit_points = trace_grounding_zone_flow(
        config, shelf_polygon, step_m=step_m, max_steps=max_steps, smooth_km=smooth_km,
        min_normal_velocity_myr=min_normal_velocity_myr, pad_km=pad_km,
    )
    margins_m = taper_margins_m(distance_from_gl_m, margin_km, min_margin_km, taper_km)
    restricted = _corridor_from_points(all_points, margins_m, shelf_polygon, grounded_polygon, config)

    if prune_near_hole_km is not None:
        paths_with_margins = _split_paths(all_points, distance_from_gl_m, margins_m)
        restricted, kept_paths = _prune_streamlines_near_hole(
            restricted, paths_with_margins, prune_near_hole_km, shelf_polygon, grounded_polygon, config,
        )
        if len(kept_paths) != len(paths_with_margins):
            exit_points = np.array([p[-1] for p in kept_paths if len(p) > 0])

    return restricted, exit_points


def process_basin_partition(
    config: PipelineConfig, shelf_polygon, grounded_polygon, margin_km: float, **partition_kwargs,
):
    """Run the full `basin_partition.partition_shelf` (every basin
    touching the shelf, competing by nearest-neighbor lookup against each
    other's traced flow), keep only points labeled `config.grounding_zone`,
    buffer by `margin_km`, and intersect with the whole-shelf floating
    region.

    More principled than `process_flowline_corridor` — a basin's
    territory is decided by competition against every other real
    contributing basin, not by an arbitrary margin around just this
    basin's own centerline — but expensive for a large, multi-tributary
    shelf: the velocity-field crop is sized to the union of the shelf's
    bounds and *every* touching basin's own bounds
    (`basin_partition.build_streamline_atlas`), and every basin's seeds
    get traced, not just this one's.
    """
    partition, basin_names, seed_counts = basin_partition.partition_shelf(
        config.ice_shelf, paths=config.paths, **partition_kwargs
    )
    if config.grounding_zone not in basin_names:
        raise ValueError(
            f"{config.grounding_zone!r} does not touch {config.ice_shelf!r} at all "
            f"(touching basins: {basin_names})."
        )
    mask = partition.basin == config.grounding_zone
    if not mask.any():
        raise ValueError(
            f"{config.grounding_zone!r} touches {config.ice_shelf!r} but has zero points "
            f"labeled in the partition (seed_counts={seed_counts}) — check "
            "min_normal_velocity_myr, or that this basin has real inflow at the "
            "grounded/floating interface, not just a shared basin-polygon boundary."
        )
    points = np.column_stack([partition.x[mask], partition.y[mask]])
    restricted = _corridor_from_points(points, margin_km, shelf_polygon, grounded_polygon, config)
    # No per-point "distance from GL" here (partition.x/y are shelf
    # velocity-grid points labeled by nearest-neighbor lookup against the
    # streamline atlas, not the traced paths themselves) — front trimming
    # falls back to the generic, more generous polygon-boundary
    # tolerance. See module docstring.
    return restricted, None


FLOATING_REGION_SOURCES = {
    "whole_shelf": process_whole_shelf,
    "flowline_corridor": process_flowline_corridor,
    "basin_partition": process_basin_partition,
}


def load_floating_region(config: PipelineConfig, shelf_polygon, grounded_polygon):
    try:
        fn = FLOATING_REGION_SOURCES[config.floating_region_source]
    except KeyError:
        raise KeyError(
            f"Unknown floating_region_source {config.floating_region_source!r}. "
            f"Available: {sorted(FLOATING_REGION_SOURCES)}"
        )
    polygon, front_reference_points = fn(config, shelf_polygon, grounded_polygon, **config.floating_region_kwargs)
    return _clean_floating_polygon(polygon), front_reference_points
