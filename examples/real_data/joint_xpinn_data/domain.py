"""Region geometry: combine extent lookup + GL/front providers into the two
polygons (grounded grounding-zone, floating shelf) the rest of the pipeline
operates on.
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
import shapely
from scipy.spatial import cKDTree

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import Geometry
from joint_xpinn_data.data_sources import boundaries, calving_front, grounding_line
from joint_xpinn_data import floating_region
from joint_xpinn_data.utils import basin_partition
from joint_xpinn_data.utils.geometry_utils import (
    contiguous_runs_within,
    mask_to_runs,
    resample_geometry,
)
from joint_xpinn_data.utils.raster_utils import polygon_from_level_set

# Below this, a front/velocity offset is measurement-grid noise (velocity
# grid spacing is 450m, BedMachine 500m) rather than a real epoch mismatch
# worth acting on — see _reconcile_front_with_velocity.
_FRONT_VELOCITY_TOLERANCE_M = 500.0

# Generous enough to absorb basin/shelf polygons being independently
# digitized products that don't align to the meter (empirically ~0m for
# the genuinely shared arc); tight enough to drop a GL source's
# tail points that trail off along a basin's lateral wall rather than its
# ice-shelf-facing edge (jumps straight to several km past the real
# interface — see HANDOFF.md's grounding-line-interface item) — see
# _trim_gl_to_interface.
_GL_INTERFACE_TOLERANCE_M = 1000.0

# Independently-digitized products describing "the same" seaward edge
# disagree by several km even for the *whole*, unrestricted shelf —
# confirmed empirically for Amery: bedmachine_mask's front points sit a
# median 1360m / max 5583m from floating_polygon's own boundary (from
# iceshelves_2008_v2.mat's coarser digitized polygon), across all three
# zones. Generalization/detail differences between the two independent
# products, not necessarily an epoch effect. Generous enough to keep
# every real front point for the whole-shelf case; still tight enough to
# drop points hundreds of km away along an unrelated stretch of a large,
# multi-tributary shelf's coastline once the floating region is
# restricted to one basin's corridor — see _trim_front_to_region. Used
# only when a strategy has no tighter reference_points to offer (e.g.
# process_basin_partition, or whole_shelf where it's the only option).
_FRONT_REGION_TOLERANCE_M = 8000.0

# Tighter tolerance for re-trimming a constant-width corridor's calving front
# to the FINAL corridor boundary (see build_regions): small enough to drop
# the wider discharge fan that sticks out past the uniform tube, generous
# enough to keep the terminus front despite the front product and the
# buffered-streamline terminus being independently derived (Amery/Lambert:
# front-to-corridor median ~250m, the fan starts past ~1km).
_FRONT_CORRIDOR_TOLERANCE_M = 2000.0

# How far around each traced streamline's own exit point (where it left
# shelf_polygon) to keep floating_polygon.boundary as a valid front match
# candidate — see _trim_front_to_region. Sized to comfortably span the
# real gaps *between* exit points along the front-facing arc (confirmed
# for Byrd: ~24 exit points spread across the corridor's downstream end,
# individual exit-point-to-real-front distances up to ~45km) while still
# excluding the corridor's long lateral sides entirely (confirmed: 0
# bleed-through at this radius, vs. the same tolerance applied to the
# *unrestricted* floating_polygon.boundary, which does bleed through).
_FRONT_EXIT_BUFFER_M = 20000.0

# grounded_polygon's boundary (basin_polygon, from basins_refined_v2.mat)
# and shelf_polygon's boundary (iceshelves_2008_v2.mat) are independently
# digitized products that cross each other repeatedly over short spans
# near the coast, rather than meeting along one clean shared arc —
# confirmed for Byrd: shelf_polygon.difference(grounded_polygon) picked
# up several new interior holes/notches right at the Byrd/Ross East
# interface that aren't present in shelf_polygon itself (its own real
# holes, e.g. ice rises, are all elsewhere) — these fed straight through
# into the floating corridor as a small disconnected island, a notch, and
# a near-pinch right where the flow is fastest and most important. A
# small morphological closing (buffer out then back in) on
# grounded_polygon removes sub-2km-scale digitization noise like this
# while leaving real, larger-scale shape essentially untouched (confirmed
# for Byrd: eliminates every near-GL hole at a 0.001% area change) — see
# _buffer_grounded.
_GROUNDED_POLYGON_SMOOTHING_M = 2000.0

# shelf_polygon's and basin_polygon's own boundaries (not just their
# interaction via grounded_polygon above) each carry the same basin/shelf
# digitization mismatch at a much larger scale: confirmed for Byrd, a
# genuine ~15-20km-deep V-shaped wedge (plus a second, smaller kink
# further along) in shelf_polygon's own boundary lines up almost exactly
# with a corner of basin_polygon's boundary — well beyond what
# _GROUNDED_POLYGON_SMOOTHING_M's few-km closing can smooth. The same
# mismatch shows up in basin_polygon's own boundary too once the grounded
# region's own shape (see _grounded_corridor) is smooth and wide enough
# to reach it. A closing radius large enough to fix this shelf/basin-wide
# would risk eating real, large-scale structure elsewhere (confirmed:
# shelf_polygon's own ~524 km^2 hole, likely a real named ice rise far
# from Byrd, visibly shrinks once an unrestricted closing radius reaches
# ~8-15km) — so _smooth_near_interface applies this only within
# _INTERFACE_EXTENT_M of the *other* polygon (comfortably larger than the
# ~15-20km wedge, nowhere near that distant hole), leaving the rest of
# each polygon untouched. Per explicit user direction: cutting off real
# shelf/basin territory here is an acceptable trade for a smooth grounded/
# floating transition, not an attempt to preserve every real detail of
# either polygon's own boundary near the interface.
#
# Applied to the *results* (grounded_polygon, floating_polygon), not to
# shelf_polygon/basin_polygon themselves before anything else uses them —
# confirmed the raw shapes have to survive through GL loading and
# flux-gate-seed/streamline tracing: smoothing basin_polygon before GL
# loading dropped every real GL point above grounding_line_min_speed_myr
# (a smoothed reference ring picks different points by real-distance
# selection); smoothing shelf_polygon before tracing changed which
# flux-gate seeds count as already inside it, giving a handful of
# borderline seeds inconsistent path lengths — a new sharp spike right at
# the seed line, not a fix. See build_regions for exactly where each
# smoothing happens.
_INTERFACE_SMOOTHING_M = 20000.0
_INTERFACE_EXTENT_M = 60000.0

# A second, more serious problem than the two above: even applied to the
# *results*, closing has no notion of "the real grounding line must stay
# exactly here" — it just smooths shape. Confirmed for Byrd: with no
# protection, the closing shifted grounded_polygon's and floating_polygon's
# shared boundary 13-26km away from the actual reported `gl` (x_md/y_md),
# so the two no longer met where the real interface is — the closing
# happened to smooth over a real concave feature of basin_polygon that
# the true GL sits right on top of. `_smooth_near_interface` excludes a
# `_GL_PROTECTION_RADIUS_M` buffer around the real GL points from the
# smoothing entirely, so that segment of the boundary is always returned
# byte-for-byte as it was, regardless of `radius_m`/`interface_extent_m`.
#
# 10km (not a larger value — 15km was tried first) is a deliberate
# trade-off, not just "big enough": alignment holds at every radius from
# 5-30km tested for grounded_polygon (gl-to-boundary distance ~2-123m
# throughout, matching real data resolution either way), but
# basin_polygon's raw, real embayment right at the interface is itself
# ~15-25km wide — a 15km protection radius fully revives that embayment
# (confirmed: visibly, as a deep concave notch cutting into
# grounded_polygon, which build_dataset._dirichlet_from_mask then reads
# as spurious Dirichlet boundary points along the real interface and in
# the shelf interior once the notch's own farthest excursion exceeds its
# `gl_exclusion_km`). At 5-10km, grounded_polygon's real interface
# segment survives exactly (protection still works — see above) but the
# embayment's own wider excursions fall *outside* the protected zone and
# get smoothed away with everything else, so it comes out as a single
# clean, notch-free boundary while `_dirichlet_from_mask`'s default 5km
# `gl_exclusion_km` still correctly distinguishes the (now narrow) real-
# interface segment from the (now fully present) lateral cut boundary on
# either side. 10km specifically (not 5km) because floating_polygon —
# using this same constant for its own post-hoc smoothing — needed the
# extra margin: at 5km, `floating_polygon`'s own alignment with `gl`
# stayed broken (~26km) even though grounded_polygon's was already fixed,
# since the two regions' boundaries near the interface don't have
# identical local shape.
_GL_PROTECTION_RADIUS_M = 10000.0

# Corner-rounding radius for the grounded region (see _round_corners). Large
# enough to smooth the corners left by intersecting the grounded masks
# (velocity contour / stadium / interface half-plane), small relative to the
# ~13-25km-wide fast trunk so it rounds corners without pinching the region.
_GROUNDED_CORNER_ROUNDING_M = 4000.0


def _smooth_near_interface(
    polygon, reference_polygon, radius_m: float, interface_extent_m: float,
    protect_points: np.ndarray | None = None, protect_radius_m: float = 0.0,
):
    """Close small-scale boundary noise in `polygon`, but only within
    `interface_extent_m` of `reference_polygon` — real structure far from
    the interface (see _INTERFACE_SMOOTHING_M) is returned untouched.
    `protect_points` (if given) carves a `protect_radius_m` buffer around
    each point out of the smoothing zone entirely, so the real boundary
    there survives exactly as-is — see _GL_PROTECTION_RADIUS_M.

    Returns `union(polygon, closed.intersection(near_interface))`, not
    `union(polygon.difference(near_interface), closed.intersection(near_interface))`
    — the two are equivalent in exact arithmetic (morphological closing is
    inflationary: `closed ⊇ polygon` always), but the "difference then
    union" form glues two independently-computed pieces together along a
    shared seam that must line up exactly. `closed`'s own buffer-out-then-
    in round-trip only *approximates* the true closing (shapely
    discretizes circular arcs), so `closed` isn't always numerically a
    perfect superset of `polygon` right at sharp real features like a
    wiggly, unsmoothed protected GL segment — confirmed for Byrd: the
    "difference then union" form left a thin, self-overlapping
    fold/sliver running parallel to the real GL, which
    build_dataset._dirichlet_from_mask then read as spurious Dirichlet
    boundary points along the GL and in the shelf interior. Adding the
    extra area directly onto `polygon` via a plain union has no seam to
    misalign — any near-touching or slight overlap is absorbed cleanly.
    """
    near_interface = reference_polygon.buffer(interface_extent_m)
    if protect_points is not None and len(protect_points) > 0 and protect_radius_m > 0:
        protected = shapely.multipoints(shapely.points(protect_points[:, 0], protect_points[:, 1])).buffer(
            protect_radius_m
        )
        near_interface = near_interface.difference(protected)
    closed = polygon.buffer(radius_m).buffer(-radius_m)
    return shapely.union(polygon, closed.intersection(near_interface))


def _round_corners(polygon, radius_m: float, protect_points: np.ndarray | None = None, protect_radius_m: float = 0.0):
    """Round sharp corners into smooth arcs via a morphological open-then-
    close (erode+dilate, then dilate+erode) with shapely's default round
    joins. Opening rounds convex corners / trims spikes; closing rounds
    concave corners / fills notches — together they turn the corners left by
    intersecting the grounded region's several masks (the velocity-threshold
    contour, the orienting stadium, the interface half-plane) into the smooth
    boundary the grounded region should physically have.

    `protect_points`/`protect_radius_m` keep the original boundary within a
    buffer of the given points (the real grounding-line interface), so the
    rounding softens the arbitrary outer/downstream corners without shifting
    the physical interface off the reported grounding line. Returns the
    dominant connected piece (a large-enough radius can pinch a thin neck)."""
    if radius_m <= 0:
        return polygon
    opened = polygon.buffer(-radius_m).buffer(radius_m)
    smoothed = _dominant_polygon(opened.buffer(radius_m).buffer(-radius_m))
    if protect_points is not None and len(protect_points) > 0 and protect_radius_m > 0:
        protected = shapely.multipoints(shapely.points(protect_points[:, 0], protect_points[:, 1])).buffer(
            protect_radius_m
        )
        smoothed = shapely.union(smoothed.difference(protected), polygon.intersection(protected))
    return _dominant_polygon(smoothed)


@dataclass
class DomainRegions:
    grounded_polygon: object  # shapely (Multi)Polygon
    floating_polygon: object
    grounding_line: Geometry
    calving_front: Geometry
    calving_front_normals: list[np.ndarray]  # one (Ni, 2) array per calving_front segment
    # Width (m) of the near-front band within which build_dataset should
    # drop floating-region velocity points that sit seaward of
    # `calving_front` once loaded — 0.0 means don't erode. Set to the
    # measured front/velocity-coverage offset by _reconcile_front_with_velocity
    # (see there for when/why). The band bounds the erosion to the strip
    # actually between the mapped front and the velocity-coverage edge, so a
    # seaward-side misclassification can't reach deep-interior points tens of
    # km away — see build_dataset._seaward_of_front.
    front_erosion_band_m: float = 0.0


def _basin_polygon(config: PipelineConfig):
    return boundaries.get_named_polygon(str(config.path("basins_refined")), config.grounding_zone)


def _shelf_polygon(config: PipelineConfig):
    return boundaries.get_named_polygon(str(config.path("iceshelves")), config.ice_shelf)


def _outward_normals(segment: np.ndarray, floating_polygon, step_m: float = 100.0) -> np.ndarray:
    """Unit outward-pointing normal at each vertex of a calving-front segment.

    Tangent at each vertex via central differences (forward/backward at the
    endpoints), rotated 90 degrees, then oriented so a small step along the
    normal leaves the floating polygon (i.e. points seaward).
    """
    tangent = np.zeros_like(segment)
    tangent[1:-1] = segment[2:] - segment[:-2]
    tangent[0] = segment[1] - segment[0]
    tangent[-1] = segment[-1] - segment[-2]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    tangent = tangent / norm

    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    probe = segment + normal * step_m
    inside = shapely.contains(floating_polygon, shapely.points(probe[:, 0], probe[:, 1]))
    normal[inside] *= -1
    return normal


def _trim_front_to_region(
    front: Geometry, floating_polygon, reference_points: np.ndarray | None = None, tol_m: float | None = None,
) -> Geometry:
    """Keep only calving-front points near the true seaward edge of
    `floating_polygon`. `calving_front.load_calving_front`'s providers
    derive their own crop from the *named shelf's* full extent regardless
    of what `floating_polygon` actually is (every provider accepts a
    `region_polygon` argument but none reference it in its body) — so
    once `floating_polygon` is restricted to a corridor
    (`floating_region.py`), front points from the rest of the shelf need
    to be dropped explicitly, or they'd corrupt `_outward_normals`'
    orientation test and `build_dataset._filter_by_front`'s nearest-point
    lookup right where the restriction matters most.

    `reference_points`, if provided (a restricted-region strategy's own
    per-streamline exit points — see `floating_region.py`), is used to
    restrict *which part* of `floating_polygon.boundary` counts, not
    matched against directly as a sparse point set: exit points are only
    ~1 per traced streamline (~24 for Byrd), far too sparse to cover a
    real front's own point density directly — confirmed empirically,
    point-to-nearest-exit-point distance for genuinely local front points
    has a median around 8km and a long tail past 40km, so a tolerance
    tight enough to avoid lateral bleed-through (see below) left the vast
    majority of the true local front classified as generic cut boundary
    instead, not just a corner. Buffering the exit points by
    `_FRONT_EXIT_BUFFER_M` and intersecting with `floating_polygon.
    boundary` instead gives a *continuous* restricted reference — dense
    coverage along the whole real front-facing arc between exit points,
    not just isolated circles around each one — while still excluding
    the corridor's long lateral sides (confirmed: matching against this
    at the standard `_FRONT_REGION_TOLERANCE_M` pulls in 0 points outside
    the true local front's own bounding area, vs. bleeding sideways
    through the lateral edges if `floating_polygon.boundary` were used
    unrestricted at that same tolerance).

    Plain pointwise filtering, not a contiguous-run trim like
    `_trim_gl_to_interface`: front sources have no meaningful walk order
    to preserve here (mask-based sources are pixel-scan-order scatters
    regardless of restriction), so "contiguous" isn't a concept worth
    protecting.
    """
    if reference_points is not None and len(reference_points) > 0:
        exit_buffer = shapely.multipoints(shapely.points(reference_points[:, 0], reference_points[:, 1])).buffer(
            _FRONT_EXIT_BUFFER_M
        )
        geom = floating_polygon.boundary.intersection(exit_buffer)
    else:
        geom = floating_polygon.boundary
    tol_m = tol_m if tol_m is not None else _FRONT_REGION_TOLERANCE_M

    kept_segments = []
    kept_normals = [] if front.normals is not None else None
    for i, seg in enumerate(front.segments):
        dist = shapely.distance(shapely.points(seg[:, 0], seg[:, 1]), geom)
        keep = dist <= tol_m
        kept_segments.append(seg[keep])
        if kept_normals is not None:
            kept_normals.append(front.normals[i][keep])
    if sum(len(s) for s in kept_segments) == 0:
        raise ValueError(
            f"No calving-front points found within {tol_m/1000:g}km of the floating "
            "region's boundary — check calving_front_source or "
            "floating_region_source/floating_region_kwargs."
        )
    return Geometry(
        segments=kept_segments, product=front.product, epoch=front.epoch,
        normals=kept_normals, ordered=front.ordered,
    )


def _trim_gl_to_interface(gl: Geometry, grounded_polygon, floating_polygon) -> Geometry:
    """Keep only the grounding-line points actually near where the
    grounded and floating regions touch. A GL source's own inclusion
    criterion doesn't know about the ice-shelf polygon at all — e.g.
    grounding_line.process_measures_gl keeps continental points within
    tol_km of the *basin's entire boundary ring*, which includes lateral
    basin walls (shared with neighboring basins/rock), not just the
    ice-shelf-facing arc. Confirmed empirically for Amery: raw per-point
    distance to the true grounded/floating boundary is ~0m through the
    middle of each returned segment, then jumps straight to several km at
    both tail ends — those tails are on a basin wall, not the interface.

    Ordered sources (Geometry.ordered=True) are trimmed by contiguous run
    (drop the tails, keep the middle) since position in the array is
    meaningful; unordered sources are filtered pointwise instead, since
    "contiguous" has no meaning for a scatter with no natural walk order.
    """
    shared = grounded_polygon.boundary.intersection(floating_polygon.boundary)
    if shared.is_empty:
        return gl

    if gl.ordered:
        kept_segments = []
        for seg in gl.segments:
            kept_segments.extend(contiguous_runs_within(seg, shared, _GL_INTERFACE_TOLERANCE_M))
        if not kept_segments:
            raise ValueError(
                f"No grounding-line points found within {_GL_INTERFACE_TOLERANCE_M/1000:g}km "
                "of the true grounded/floating boundary — check grounding_line_source "
                "or the region polygons."
            )
        return Geometry(segments=kept_segments, product=gl.product, epoch=gl.epoch, ordered=True)

    kept_segments, kept_normals = [], [] if gl.normals is not None else None
    for i, seg in enumerate(gl.segments):
        dist = shapely.distance(shapely.points(seg[:, 0], seg[:, 1]), shared)
        keep = dist <= _GL_INTERFACE_TOLERANCE_M
        kept_segments.append(seg[keep])
        if kept_normals is not None:
            kept_normals.append(gl.normals[i][keep])
    if sum(len(s) for s in kept_segments) == 0:
        raise ValueError(
            f"No grounding-line points found within {_GL_INTERFACE_TOLERANCE_M/1000:g}km "
            "of the true grounded/floating boundary — check grounding_line_source "
            "or the region polygons."
        )
    return Geometry(segments=kept_segments, product=gl.product, epoch=gl.epoch, normals=kept_normals, ordered=False)


def _filter_gl_by_flow(gl: Geometry, min_speed_myr: float, config: PipelineConfig) -> Geometry:
    """Keep only grounding-line points where real velocity data shows at
    least this much flow. A geometrically-correct GL point (already
    trimmed to the true grounded/floating interface by
    _trim_gl_to_interface) can still span more of a basin's downstream
    boundary than just the named glacier's own fast-flowing trunk — e.g.
    Byrd's basin polygon's real interface with Ross East is ~76km wide (a
    single contiguous run, not a leakage bug), including slower lateral
    margins/auxiliary outlets that share the same basin polygon in
    basins_refined_v2.mat's delineation. Mirrors
    build_dataset._filter_by_speed's existing real-velocity-magnitude
    filtering (grounded-region interior points), applied to the
    grounding line itself.

    basin_partition.velocity_field's raw VX/VY are already in m/yr (see
    basin_partition._basin_shelf_interface's docstring), unlike
    data_sources.velocity's m/s convention — no unit conversion needed
    here.

    Ordered sources are filtered by contiguous run (drop slow tails, keep
    the fast middle) for the same reason _trim_gl_to_interface is;
    unordered sources are filtered pointwise, since "contiguous" has no
    meaning for a scatter with no natural walk order.
    """
    all_points = gl.all_points()
    (bx0, by0), (bx1, by1) = all_points.min(axis=0), all_points.max(axis=0)
    velocity_at = basin_partition.velocity_field((bx0, by0, bx1, by1), pad_km=20.0, paths=config.paths)

    if gl.ordered:
        kept_segments = []
        for seg in gl.segments:
            speed_myr = np.linalg.norm(velocity_at(seg), axis=1)
            kept_segments.extend(mask_to_runs(seg, speed_myr > min_speed_myr))
        if not kept_segments:
            raise ValueError(
                f"No grounding-line points found with real velocity above "
                f"{min_speed_myr} m/yr for {config.ice_shelf}/{config.grounding_zone} "
                "— lower grounding_line_min_speed_myr."
            )
        return Geometry(segments=kept_segments, product=gl.product, epoch=gl.epoch, ordered=True)

    kept_segments, kept_normals = [], [] if gl.normals is not None else None
    for i, seg in enumerate(gl.segments):
        speed_myr = np.linalg.norm(velocity_at(seg), axis=1)
        keep = speed_myr > min_speed_myr
        kept_segments.append(seg[keep])
        if kept_normals is not None:
            kept_normals.append(gl.normals[i][keep])
    if sum(len(s) for s in kept_segments) == 0:
        raise ValueError(
            f"No grounding-line points found with real velocity above "
            f"{min_speed_myr} m/yr for {config.ice_shelf}/{config.grounding_zone} "
            "— lower grounding_line_min_speed_myr."
        )
    return Geometry(segments=kept_segments, product=gl.product, epoch=gl.epoch, normals=kept_normals, ordered=False)


def _grounded_corridor(
    gl_points: np.ndarray, basin_polygon, shelf_polygon, config: PipelineConfig,
    lateral_margin_km: float, length_km: float, min_speed_myr: float,
):
    """A "stadium" (line buffer) envelope, narrowed to real fast-flowing
    ice within it via velocity thresholding — constrains how wide the
    grounded region is allowed to be near the grounded/floating interface.
    The same "corridor" idea `floating_region.py` uses on the floating
    side, mirrored here, since `grounded_polygon = buffer(gl, buffer_km)
    ∩ basin_polygon` can be dominated by `basin_polygon`'s own width near
    the coast regardless of how narrow `gl` itself is. Confirmed for
    Byrd: even after re-buffering from a flow-filtered ~20km-wide `gl`
    (see `_filter_gl_by_flow`), the intersection with `basin_polygon`
    stayed ~100-140km wide — `buffer_km=100` is generous enough that
    `basin_polygon`'s own shape, not `gl`'s width, determines the
    result.

    The stadium alone (built from real velocity direction at `gl_points`,
    not a geometric fit to the points themselves — too short/curvy, ~20km
    for Byrd, to reliably estimate the true along-glacier orientation:
    mean flow direction is "downstream," its negation is upstream,
    perpendicular to it is the across-glacier width axis) fixed the
    original sharp corner from the width mismatch above, but intersecting
    its straight sides against `basin_polygon`'s own real, independently-
    digitized coastline detail can still produce new, smaller sharp
    corners wherever the two boundaries cross (the same kind of
    basin/shelf digitization mismatch `_GROUNDED_POLYGON_SMOOTHING_M`/
    `_smooth_near_interface` address on the shelf side). Rather than
    fight that mismatch geometrically again, replace the stadium's own
    straight, hard-edged sides with a contour of real (Gaussian-smoothed)
    velocity magnitude — `min_speed_myr` (matched to
    `grounding_line_min_speed_myr` for a consistent fast-trunk
    definition) traced via `utils.raster_utils.polygon_from_level_set`
    (marching squares — smooth by construction, unlike a rasterized pixel
    mask's own blocky outline). A real velocity field's contour is
    naturally smooth and data-driven, rather than a straight line
    arbitrarily cutting across whatever digitized detail happens to be
    there. The stadium still bounds the search: it sets where to trace
    the contour (its own bounds, generously padded) and provides the
    upstream/downstream length cutoff (`length_km`) and orientation, but
    no longer directly contributes a hard edge to the final shape's width.

    Orientation (which sign of the flow direction is actually upstream)
    is resolved empirically — build the stadium both ways, keep whichever
    overlaps `basin_polygon` more / `shelf_polygon` less — rather than
    trusted from the raw velocity sign, which flipped in testing. This
    remains a crude heuristic, not a physically-traced upstream catchment
    shape — no reliable backward flow-tracing is available (see
    `utils/basin_partition.py`'s docstring on why backward tracing was
    abandoned even for the gentler, more slowly-varying floating side).
    """
    bx0, by0 = gl_points.min(axis=0)
    bx1, by1 = gl_points.max(axis=0)
    velocity_at = basin_partition.velocity_field((bx0, by0, bx1, by1), pad_km=20.0, paths=config.paths)
    mean_flow = velocity_at(gl_points).mean(axis=0)
    mean_flow_unit = mean_flow / np.linalg.norm(mean_flow)
    width_axis = np.array([-mean_flow_unit[1], mean_flow_unit[0]])

    centroid = gl_points.mean(axis=0)
    proj = (gl_points - centroid) @ width_axis
    half_width_m = (proj.max() - proj.min()) / 2.0 + lateral_margin_km * 1000.0
    length_m = length_km * 1000.0

    candidates = [
        shapely.LineString([centroid, centroid + sign * mean_flow_unit * length_m]).buffer(half_width_m)
        for sign in (1.0, -1.0)
    ]
    stadium = max(candidates, key=lambda r: r.intersection(basin_polygon).area - r.intersection(shelf_polygon).area)

    pixel_size_m = 500.0
    sbx0, sby0, sbx1, sby1 = stadium.bounds
    x = np.arange(sbx0, sbx1, pixel_size_m)
    y = np.arange(sby0, sby1, pixel_size_m)
    velocity_at_stadium = basin_partition.velocity_field(stadium.bounds, pad_km=20.0, smooth_km=3.0, paths=config.paths)
    xx, yy = np.meshgrid(x, y)
    speed = np.linalg.norm(velocity_at_stadium(np.column_stack([xx.ravel(), yy.ravel()])), axis=1).reshape(xx.shape)
    fast_polygon = polygon_from_level_set(x, y, speed, min_speed_myr)
    if fast_polygon.is_empty:
        return stadium
    return fast_polygon.buffer(lateral_margin_km * 1000.0).intersection(stadium)


def _reconcile_front_with_velocity(
    config: PipelineConfig, basin_polygon, gl: Geometry, floating_polygon, front: Geometry,
):
    """Resolve a systematic disagreement between the configured calving
    front and the velocity data's own valid-coverage edge
    (`calving_front_source="velocity_mask"`) — front and velocity products
    are essentially never from the same epoch, and shelves calve/advance
    between epochs (confirmed for Amery/Lambert: fraction of near-front
    velocity points landing seaward of the mapped front is 26.5% for the
    2021-epoch `bedmachine_mask` default vs. 3.65% for the 2017-epoch
    `antarctic_boundaries_mask` — see HANDOFF.md's front-conformance-misfit
    item).

    If the configured front sits ocean-ward of real velocity coverage,
    the floating polygon is claiming shelf area with no velocity support
    behind it — erode the polygon inward to stop at the real data extent.
    If it sits landward of real velocity coverage instead (the case
    actually observed for Amery/Lambert), the front is trusted as the true
    edge — leave the polygon alone and tell build_dataset to drop velocity
    points beyond it once loaded (domain.py never touches velocity data
    itself).

    Returns (floating_polygon, front_erosion_band_m) — the second is 0.0
    unless the front is landward of coverage, in which case it's the width
    of the strip (95th-percentile landward offset) within which
    build_dataset should drop seaward-of-front velocity points.
    """
    if config.calving_front_source == "velocity_mask":
        return floating_polygon, 0.0  # nothing independent to compare against

    ref_front = calving_front.load_calving_front(
        replace(config, calving_front_source="velocity_mask"), basin_polygon, gl
    )
    ref_points = ref_front.all_points()
    if len(ref_points) == 0:
        return floating_polygon, 0.0

    ref_normals = np.concatenate(ref_front.normals, axis=0)
    front_points = front.all_points()

    tree = cKDTree(ref_points)
    _, idx = tree.query(front_points)
    offset = front_points - ref_points[idx]
    # positive = configured front lies seaward of real velocity coverage
    signed = np.einsum("ij,ij->i", offset, ref_normals[idx])

    typical_offset = np.median(signed)
    if abs(typical_offset) <= _FRONT_VELOCITY_TOLERANCE_M:
        return floating_polygon, 0.0

    shelf_zone = f"{config.ice_shelf}/{config.grounding_zone}"
    if typical_offset > 0:
        erosion_m = float(np.percentile(signed[signed > 0], 95))
        warnings.warn(
            f"{shelf_zone}: calving_front_source={config.calving_front_source!r} lies "
            f"~{typical_offset:.0f}m ocean-ward of the velocity data's own coverage "
            f"edge — eroding the floating polygon inward by {erosion_m:.0f}m so "
            "region extent doesn't outrun real velocity data.",
            stacklevel=2,
        )
        return floating_polygon.buffer(-erosion_m), 0.0

    # The strip of seaward-of-front velocity points to drop is at most as
    # wide as how far landward the front sits from the coverage edge (past
    # that edge there's no velocity anyway) — bound the erosion band to the
    # 95th-percentile landward offset so build_dataset can't drop interior
    # points far from the front (mirrors the ocean-ward erosion_m above).
    band_m = float(np.percentile(-signed[signed < 0], 95))
    warnings.warn(
        f"{shelf_zone}: calving_front_source={config.calving_front_source!r} lies "
        f"~{-typical_offset:.0f}m landward of the velocity data's own coverage edge "
        f"— dropping velocity points within {band_m:.0f}m seaward of the mapped front "
        "instead of trusting them as real floating-ice motion.",
        stacklevel=2,
    )
    return floating_polygon, band_m


def _buffer_grounded(gl_geometry: Geometry, basin_polygon, config: PipelineConfig):
    """Buffer the grounding line's point cloud directly (not an ordered
    LineString): some GL sources (bedmachine_mask) return an unordered
    pixel scatter rather than a connected polyline. A dense-enough point
    buffer is equivalent to a line buffer as long as point spacing <<
    buffer_km — true for both the ~500m mask-pixel spacing and the
    ~km-scale vertex spacing of the measures_boundaries_2008 polyline.
    Intersects with `basin_polygon` to get the grounded corridor, then
    closes small digitization-noise notches/holes against it (see
    _GROUNDED_POLYGON_SMOOTHING_M) before anything downstream (the
    interface trim, the floating region) can react to them. Called once
    from build_regions' shared prologue (bootstrap, against the raw
    basin) and again from process_flow_restricted (re-buffer from a
    narrowed gl and a smoothed basin) — same function either way, not
    duplicated logic.
    """
    gl_points_geom = shapely.multipoints(shapely.points(*gl_geometry.all_points().T))
    corridor = gl_points_geom.buffer(config.buffer_km * 1000.0)
    grounded = corridor.intersection(basin_polygon)
    if grounded.is_empty:
        raise ValueError(
            f"{config.buffer_km} km buffer around the grounding line does not "
            f"intersect the {config.grounding_zone!r} basin polygon — check "
            "that the grounding line was clipped to the right basin."
        )
    return grounded.buffer(_GROUNDED_POLYGON_SMOOTHING_M).buffer(-_GROUNDED_POLYGON_SMOOTHING_M)


def _dominant_polygon(geom):
    """Largest connected component of a possibly-Multi polygon, so a clip
    that slivers off small fragments returns one clean Polygon."""
    if geom.is_empty:
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def _shelf_holes(shelf_polygon):
    """The shelf polygon's own interior rings (ice rises / rock outcrops,
    encoded in iceshelves_2008_v2.mat via ESRI ring winding) as a single
    (Multi)Polygon — for re-excluding from the floating region after the
    hole-filling closings. Empty polygon if there are none."""
    parts = list(shelf_polygon.geoms) if shelf_polygon.geom_type == "MultiPolygon" else [shelf_polygon]
    holes = [shapely.Polygon(r) for p in parts for r in p.interiors]
    return shapely.union_all(holes) if holes else shapely.Polygon()


def _interface_halfplane(gl: Geometry, config: PipelineConfig, depth_m: float = 800_000.0, span_m: float = 500_000.0):
    """A half-plane polygon covering the *downstream* (floating) side of
    the grounding line, used to clip grounded/floating cleanly to opposite
    sides of a single shared interface (`grounded.difference(halfplane)`,
    `floating.intersection(halfplane)`).

    Both `buffer(gl, buffer_km) ∩ basin` (grounded) and a `flowline_corridor`
    floating region spill across the grounding line — the grounded capsule
    extends downstream past it, the floating corridor protrudes upstream
    into it (confirmed for Byrd: a ~216 km^2 overlap lens straddling the
    GL). Resolving that overlap by `grounded.difference(floating_polygon)`
    carved a U-shaped bite out of the grounded region and left the
    floating corridor's raw landward edge (its own sharp corners) exposed.
    Clipping both regions at the grounding line instead removes the
    overlap by construction: the grounding line *is* the physical
    grounded/floating boundary, so grounded ends there and floating
    begins there.

    The half-plane's straight edge follows the (flow-filtered, resampled)
    GL points, extended `span_m` past both ends along the across-flow axis
    so it fully spans the regions' width, then offset `depth_m` downstream
    (both large relative to the ~450km domain). Only `process_flow_restricted`
    builds one — the `simple` strategy has no single flow-defined interface
    (its regions meet along the whole-shelf boundary) and leaves
    `RegionShape.interface_halfplane` None.
    """
    pts = gl.all_points()
    (bx0, by0), (bx1, by1) = pts.min(axis=0), pts.max(axis=0)
    velocity_at = basin_partition.velocity_field((bx0, by0, bx1, by1), pad_km=20.0, paths=config.paths)
    flow = velocity_at(pts).mean(axis=0)
    flow = flow / np.linalg.norm(flow)  # downstream unit vector
    across = np.array([-flow[1], flow[0]])

    spine = pts[np.argsort(pts @ across)]  # order GL points across the glacier width
    spine = np.vstack([spine[0] - across * span_m, spine, spine[-1] + across * span_m])
    downstream_edge = spine + flow * depth_m
    return shapely.Polygon(np.vstack([spine, downstream_edge[::-1]]))


@dataclass
class RegionShape:
    """What a `region_strategy` function returns — the finalized
    grounded/floating geometry, consumed unconditionally by
    build_regions' shared epilogue (front load/trim, normals, velocity
    reconciliation, cut-boundary computation, DomainRegions assembly).
    `basin_polygon` is whichever basin the epilogue's calving-front load
    and velocity reconciliation should see — raw for a strategy that
    never smooths it (process_simple), the smoothed working copy for one
    that does (process_flow_restricted). See docs/adr/0002-*.md.
    """
    grounded_polygon: object  # single Polygon, disjoint from floating_polygon
    floating_polygon: object  # single Polygon
    gl: Geometry  # final grounding line (post flow-filter/resample, if the strategy does either)
    basin_polygon: object
    front_reference_points: np.ndarray | None = None
    # Downstream half-plane for the clean grounded/floating interface clip
    # (see _interface_halfplane). Set by process_flow_restricted; None for
    # process_simple, whose regions meet along the whole-shelf boundary and
    # use the plain overlap-subtraction epilogue instead.
    interface_halfplane: object | None = None


def process_simple(config: PipelineConfig, basin_raw, shelf_raw, gl: Geometry, grounded_polygon, **_) -> RegionShape:
    """Amery-class region geometry: no GL flow-filtering, no velocity-
    thresholded grounded corridor, no near-interface smoothing — the
    floating region is whatever `floating_region_source` (default
    `whole_shelf`) says it is. This is `region_strategy`'s default
    ("simple") and the common case: three of today's four configs
    (Lambert/Mellor/Fisher) use it. Its body references none of
    `_filter_gl_by_flow`/`_smooth_near_interface`/`_grounded_corridor` —
    a shelf on this path structurally cannot execute Byrd-class geometry.
    Accepts and ignores `**_` so REGION_STRATEGIES can call every
    strategy the same way, forwarding `config.region_strategy_kwargs`
    regardless of which entries a given strategy actually uses. See
    docs/adr/0002-*.md.
    """
    if config.grounding_line_resample_m is not None:
        gl = resample_geometry(gl, config.grounding_line_resample_m)

    floating_polygon, front_reference_points = floating_region.load_floating_region(
        config, shelf_raw, grounded_polygon
    )
    return RegionShape(
        grounded_polygon=grounded_polygon,
        floating_polygon=floating_polygon,
        gl=gl,
        basin_polygon=basin_raw,
        front_reference_points=front_reference_points,
    )


def process_flow_restricted(
    config: PipelineConfig, basin_raw, shelf_raw, gl: Geometry, grounded_polygon,
    *, min_speed_myr: float, corridor_margin_km: float | None = None, **_,
) -> RegionShape:
    """Byrd-class region geometry: narrows the grounding line to real
    fast-flowing ice, constrains the grounded region to a velocity-
    thresholded corridor around it, and smooths both regions' near-
    interface boundary against the basin/shelf digitization mismatch
    this class of shelf exposes — see HANDOFF.md's Ross East/Byrd
    investigation and docs/adr/0002-*.md. `min_speed_myr` and
    `corridor_margin_km` come from `config.region_strategy_kwargs`;
    `min_speed_myr` is required — opting into `region_strategy=
    "flow_restricted"` without it is a config error, not a silent no-op
    (REGION_STRATEGIES' dispatch raises TypeError on the missing kwarg).
    """
    gl = _filter_gl_by_flow(gl, min_speed_myr, config)
    # Flow-filtering can narrow gl a lot (confirmed for Byrd: ~76km
    # basin-defined interface -> ~20km fast trunk) — re-buffer
    # grounded_polygon from the corrected, narrower gl so it stays
    # consistent with it. Otherwise grounded_polygon (built from the
    # *original*, unfiltered, much wider interface) can end up far
    # wider than the flow-consistent floating corridor it meets at
    # the interface, and its own boundary creates a sharp corner
    # where it cuts into the shelf — confirmed for Byrd via
    # utils.raster_utils.boundary_sharpness: a grid search over every
    # floating-region taper parameter (margin_km, min_margin_km,
    # taper_km) left this specific corner's sharpness completely
    # unchanged, proving it wasn't a taper issue at all — see
    # HANDOFF.md.
    #
    # basin_polygon is smoothed near shelf_polygon from here on
    # (safe now — gl's real fast-trunk interface was already
    # identified above using the raw one): otherwise basin_polygon's
    # own unsmoothed real coastline detail becomes the final
    # grounded_polygon's own visible, jagged edge wherever
    # _grounded_corridor's smooth velocity-thresholded shape is wide
    # enough to reach it (confirmed for Byrd).
    basin_polygon = _smooth_near_interface(
        basin_raw, shelf_raw, _INTERFACE_SMOOTHING_M, _INTERFACE_EXTENT_M,
        protect_points=gl.all_points(), protect_radius_m=_GL_PROTECTION_RADIUS_M,
    )
    grounded_polygon = _buffer_grounded(gl, basin_polygon, config)
    if corridor_margin_km is not None:
        grounded_polygon = grounded_polygon.intersection(
            _grounded_corridor(
                gl.all_points(), basin_polygon, shelf_raw, config,
                lateral_margin_km=corridor_margin_km,
                length_km=config.buffer_km * 1.5,
                min_speed_myr=min_speed_myr,
            )
        )

    if config.grounding_line_resample_m is not None:
        gl = resample_geometry(gl, config.grounding_line_resample_m)

    # Seed the floating corridor from the same fast-trunk threshold used
    # to filter the grounding line above — a mismatched, hand-set
    # min_normal_velocity_myr let the corridor's own seeds span the
    # basin's whole (slower, wider) interface regardless of the GL
    # filter (confirmed for Byrd: the corridor started just as wide
    # immediately, no taper fixes a lateral-spread problem). `setdefault`
    # so an explicit override in floating_region_kwargs still wins.
    floating_kwargs = dict(config.floating_region_kwargs)
    floating_kwargs.setdefault("min_normal_velocity_myr", min_speed_myr)
    floating_polygon, front_reference_points = floating_region.load_floating_region(
        replace(config, floating_region_kwargs=floating_kwargs), shelf_raw, grounded_polygon
    )
    # Near-interface smoothing of floating_polygon is NOT specific to this
    # strategy — build_regions' shared epilogue applies it to every
    # shelf-class's output (confirmed non-trivial for Amery too, not just
    # Byrd — see docs/adr/0002-*.md). The interface_halfplane, by contrast,
    # IS specific: it needs this strategy's single flow-defined grounding
    # line, so the epilogue uses it (when present) to clip grounded/floating
    # to opposite sides of the GL instead of the plain overlap subtraction.
    return RegionShape(
        grounded_polygon=grounded_polygon,
        floating_polygon=floating_polygon,
        gl=gl,
        basin_polygon=basin_polygon,
        front_reference_points=front_reference_points,
        interface_halfplane=_interface_halfplane(gl, config),
    )


# Registry of shelf-class region-geometry strategies, mirroring
# floating_region.FLOATING_REGION_SOURCES — see docs/adr/0002-*.md for
# why the grounded/GL half of shelf-class variation needed the same
# contract+registry treatment the floating half already had.
REGION_STRATEGIES = {
    "simple": process_simple,
    "flow_restricted": process_flow_restricted,
}


def build_regions(config: PipelineConfig) -> DomainRegions:
    """Shared skeleton: raw polygon load -> GL load -> bootstrap grounded
    buffer -> GL-interface trim are common to every shelf-class and never
    vary; front load/trim, normals, velocity reconciliation, cut-boundary
    computation, and DomainRegions assembly likewise never vary. Only the
    geometry in between — how far the grounded/floating regions actually
    extend — differs per shelf-class, dispatched to one of
    REGION_STRATEGIES via `config.region_strategy` (default "simple").
    See docs/adr/0002-*.md.

    basin_raw/shelf_raw stay raw through GL loading and the bootstrap
    buffer/trim below — anything that depends on the *true* physical
    boundary (GL sources selecting points by real distance to
    basin_raw's own ring; a strategy's own flux-gate seeds/traced
    streamlines terminating against shelf_raw's real edge) needs the
    unmodified shape; see process_flow_restricted for why smoothing must
    happen only after the fact, inside the strategy that needs it.
    """
    basin_raw = _basin_polygon(config)
    shelf_raw = _shelf_polygon(config)

    gl = grounding_line.load_grounding_line(config, basin_raw)
    grounded_polygon = _buffer_grounded(gl, basin_raw, config)

    # Cheap bootstrap for the interface trim below (shelf_raw minus the
    # whole-shelf grounded corridor, not a floating_region_source
    # restriction) — the true grounded/floating interface doesn't depend
    # on how far the floating region ends up extending downstream, so
    # there's no need to pay for an expensive corridor trace just for
    # this trim.
    gl = _trim_gl_to_interface(gl, grounded_polygon, shelf_raw.difference(grounded_polygon))

    try:
        strategy = REGION_STRATEGIES[config.region_strategy]
    except KeyError:
        raise KeyError(
            f"Unknown region_strategy {config.region_strategy!r}. "
            f"Available: {sorted(REGION_STRATEGIES)}"
        )
    shape = strategy(config, basin_raw, shelf_raw, gl, grounded_polygon, **config.region_strategy_kwargs)

    # Smooth the strategy's floating result against basin_raw (not
    # shelf_raw itself beforehand — the raw shape has to survive through
    # GL loading and any strategy's own tracing, see the docstring above)
    # — basin/shelf digitization mismatch can carve a large-scale
    # (~15-20km) notch into the final shape near the interface, on top of
    # the smaller (sub-3km) scale floating_region._clean_floating_polygon
    # already addresses. Applies to every shelf-class uniformly —
    # confirmed non-trivial for Amery's "simple" strategy too, not just
    # "flow_restricted" (see docs/adr/0002-*.md).
    #
    # The GL-protection radius (carving a buffer around the real GL out of
    # the smoothing zone) exists only to keep floating aligned to the real
    # GL through this inflationary closing (bug 1). When the strategy
    # supplies an interface_halfplane, the downstream clip below re-imposes
    # that alignment exactly (GL-to-boundary ~11m mean either way), so the
    # protection is redundant — and its carve-out actively introduces a
    # thin (~2km) notch where the protected raw zone abuts the smoothed
    # zone, ~8km from the GL (confirmed for Byrd). Drop it when a clip
    # follows; keep it for `simple`, which has no clip to realign the GL.
    floating_protect_m = 0.0 if shape.interface_halfplane is not None else _GL_PROTECTION_RADIUS_M
    floating_polygon = _smooth_near_interface(
        shape.floating_polygon, basin_raw, _INTERFACE_SMOOTHING_M, _INTERFACE_EXTENT_M,
        protect_points=shape.gl.all_points(), protect_radius_m=floating_protect_m,
    )
    # Resolve the grounded/floating overlap. Both regions can spill across
    # the grounding line (the grounded capsule extends downstream of it;
    # _smooth_near_interface, being inflationary, regrows floating upstream
    # into it), so build_dataset would otherwise report the same location
    # as both grounded and floating velocity — confirmed for Byrd, a
    # ~216 km^2 overlap lens straddling the GL.
    if shape.interface_halfplane is not None:
        # flow_restricted: the strategy defined a single flow-aligned
        # grounding line, so clip both regions to opposite sides of it —
        # the GL is the physical grounded/floating interface. This gives a
        # clean, smooth shared boundary and removes the overlap by
        # construction, unlike the plain subtraction below, which carved a
        # U-shaped bite out of the grounded region and left the floating
        # corridor's raw landward corners exposed (see _interface_halfplane).
        grounded_polygon = _dominant_polygon(shape.grounded_polygon.difference(shape.interface_halfplane))
        floating_polygon = _dominant_polygon(floating_polygon.intersection(shape.interface_halfplane))
        # Round the corners the mask intersections leave in the grounded
        # region (see _round_corners), protecting the real GL interface so
        # only the arbitrary outer/downstream corners soften; then re-clip to
        # the half-plane so the closing step can't regrow grounded back across
        # the interface into the floating side (reintroducing the overlap
        # resolved above).
        grounded_polygon = _round_corners(
            grounded_polygon, _GROUNDED_CORNER_ROUNDING_M,
            protect_points=shape.gl.all_points(), protect_radius_m=_GL_PROTECTION_RADIUS_M,
        )
        grounded_polygon = _dominant_polygon(grounded_polygon.difference(shape.interface_halfplane))
    else:
        # simple: no single flow-defined interface (the regions meet along
        # the whole-shelf boundary). Treat floating_polygon as authoritative
        # in any overlap and shrink grounded_polygon — floating's smoothing
        # above specifically fixed a basin/shelf digitization mismatch and
        # is GL-aligned, whereas grounded has no comparable reason to extend
        # into that area. A near-no-op for Amery (negligible overlap),
        # verified byte-identical by tests/regression_baseline.py.
        grounded_polygon = _dominant_polygon(
            _round_corners(shape.grounded_polygon, _GROUNDED_CORNER_ROUNDING_M).difference(floating_polygon)
        )

    front = calving_front.load_calving_front(config, shape.basin_polygon, shape.gl)
    # Trim before computing normals/reconciling with velocity below —
    # both compare against floating_polygon's shape, and an untrimmed
    # shelf-wide front would corrupt both once floating_polygon is a
    # restricted corridor (see _trim_front_to_region).
    front = _trim_front_to_region(front, floating_polygon, reference_points=shape.front_reference_points)

    # Front-extension: union a small buffer around the (already-trimmed)
    # front points into floating_polygon so the region actually reaches every
    # front point it reports (otherwise _trim_front_to_region's tolerance can
    # leave the polygon edge up to _FRONT_REGION_TOLERANCE_M short of the
    # front markers — a visible gap for a whole-shelf region whose own
    # digitized seaward edge disagrees with the front by ~1.4km).
    #
    # NOT applied to a restricted corridor (flowline_corridor / basin_partition):
    # a corridor's real calving front fans wider than its constant-width
    # buffered-streamline terminus, so reaching every front point would bulge
    # the corridor out near the front — the "sudden widening" that defeats the
    # point of a constant-width corridor. For a corridor the front is instead
    # left trimmed to the part within _FRONT_REGION_TOLERANCE_M of the
    # constant-width terminus (a small, acceptable gap), keeping the corridor
    # uniform. Also skipped for `simple` (no interface_halfplane), which keeps
    # floating_polygon exactly as the overlap resolution left it.
    _is_corridor = config.floating_region_source in ("flowline_corridor", "basin_partition")
    if shape.interface_halfplane is not None and not _is_corridor:
        front_pts = front.all_points()
        front_reach = shapely.multipoints(shapely.points(front_pts[:, 0], front_pts[:, 1])).buffer(
            _FRONT_REGION_TOLERANCE_M
        )
        floating_polygon = _dominant_polygon(shapely.union(floating_polygon, front_reach))

    floating_polygon, erosion_band_m = _reconcile_front_with_velocity(
        config, shape.basin_polygon, shape.gl, floating_polygon, front
    )

    # Clip the floating region's seaward extent to the calving-front source's
    # own ice/valid region, so the ocean-facing edge coincides with the front
    # by construction (calving_front.load_front_extent — velocity_mask ->
    # valid-velocity blob, bedmachine_mask -> floating-ice blob, etc.). The
    # base floating region comes from the ~2008 iceshelves polygon, which
    # overruns ice that has since grounded or lies past the current front at
    # the margins (confirmed for Amery: NE/SW corners are grounded in
    # BedMachine 2021); that stale edge, far from any mapped front point, was
    # being mislabeled as a Dirichlet cut boundary. A GL buffer is unioned
    # into the clip region so it only trims the seaward side and never pulls
    # the floating region back from its modeled grounding-line interface.
    # Line-based front sources (measures_boundaries_2008, custom_xy) return
    # None -> no clip. A near-no-op for a restricted corridor already well
    # inside the ice extent (e.g. Byrd's flowline_corridor).
    extent = calving_front.load_front_extent(config, shape.basin_polygon, shape.gl)
    if extent is not None and not extent.is_empty:
        gl_pts = shape.gl.all_points()
        gl_guard = shapely.multipoints(shapely.points(gl_pts[:, 0], gl_pts[:, 1])).buffer(_GL_PROTECTION_RADIUS_M)
        floating_polygon = _dominant_polygon(floating_polygon.intersection(shapely.union(extent, gl_guard)))

    # Restore the shelf's genuine interior holes (ice rises / rock outcrops).
    # The closings above (floating_region._clean_floating_polygon,
    # _smooth_near_interface) fill holes to erase the sub-2km-to-20km
    # basin/shelf digitization-mismatch notches, but that also fills these
    # real islands, which must stay excluded from the floating region — they
    # aren't floating shelf. The mismatch artifacts aren't present in
    # shelf_raw, so re-punching only shelf_raw's own holes brings back the
    # real ones without the spurious ones. A no-op wherever a hole doesn't
    # overlap floating_polygon (e.g. a restricted corridor away from any ice
    # rise).
    holes = _shelf_holes(shelf_raw)
    if not holes.is_empty:
        floating_polygon = _dominant_polygon(floating_polygon.difference(holes))

    # For a constant-width corridor, re-trim the calving front to the FINAL
    # corridor so xct/yct conform to it: the earlier trim ran against the
    # pre-reconcile/pre-extent-clip corridor and, with no front-extension to
    # pull the corridor out to the wider discharge fan, left front points
    # sticking out past the uniform tube (confirmed for Amery/Lambert: 57 of
    # 203 points up to ~7.9km beyond the corridor). Tight tolerance against
    # the final boundary directly (no exit-point reference) — the front is
    # already terminus-local from the first trim, so there's no lateral-bleed
    # risk, and dropping the fan makes the front match the corridor's own
    # seaward edge. whole_shelf keeps its front_reach-extended front.
    if _is_corridor:
        front = _trim_front_to_region(front, floating_polygon, tol_m=_FRONT_CORRIDOR_TOLERANCE_M)

    if front.normals is not None:
        normals = front.normals
    else:
        normals = [_outward_normals(seg, floating_polygon) for seg in front.segments]

    return DomainRegions(
        grounded_polygon=grounded_polygon,
        floating_polygon=floating_polygon,
        grounding_line=shape.gl,
        calving_front=front,
        calving_front_normals=normals,
        front_erosion_band_m=erosion_band_m,
    )
