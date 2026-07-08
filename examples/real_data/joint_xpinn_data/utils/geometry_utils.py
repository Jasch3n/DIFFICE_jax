"""Small generic geometry helpers shared by domain.py and the boundary/
calving-front providers. Nothing here is source-specific."""

import warnings

import numpy as np
import shapely

from joint_xpinn_data.contracts import Geometry

# Wrapped in a banner rather than a plain sentence — resampling an
# unordered scatter as if it were a walkable line is a real correctness
# risk (see order_counterclockwise's docstring), not a routine notice, so
# this needs to be impossible to miss in a log full of other warnings.
_UNORDERED_WARNING_BANNER = "!" * 78


def largest_ring(polygon) -> np.ndarray:
    """Exterior ring coordinates of a Polygon, or of the largest part of a
    MultiPolygon."""
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    return np.asarray(polygon.exterior.coords)


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Evenly-spaced resample of an ordered (N,2) polyline via linear
    interpolation along cumulative arc length — artificially controls
    point resolution independent of whatever spacing the source vertices
    happened to come at. Always keeps both endpoints exactly.

    Only meaningful for a genuinely *ordered* polyline (true for
    measures_boundaries_2008's grounding line/front segments); do not use
    on an unordered pixel scatter (e.g. a mask-based provider's output) —
    interpolating between arbitrarily-adjacent points would connect
    unrelated neighbors into a meaningless zigzag.
    """
    if len(points) < 2:
        return points
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumdist[-1]
    if total == 0:
        return points[:1]
    n_segments = max(int(np.ceil(total / spacing_m)), 1)
    targets = np.linspace(0.0, total, n_segments + 1)
    x = np.interp(targets, cumdist, points[:, 0])
    y = np.interp(targets, cumdist, points[:, 1])
    return np.column_stack([x, y])


def order_counterclockwise(points: np.ndarray) -> np.ndarray:
    """Impose an ordering on an unordered (N,2) point cloud by sorting
    counterclockwise around its centroid (ascending polar angle). A
    heuristic, not a real reconstruction of the original boundary — exact
    only for a star-shaped cloud (every point visible from the centroid
    without crossing the boundary); a long, curvy, non-loop grounding-line
    arc can violate that, so treat the result as approximate."""
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def resample_geometry(geometry: Geometry, spacing_m: float) -> Geometry:
    """Apply `resample_polyline` to every segment of a Geometry, so its
    resolution (and downstream point counts like x_md/y_md) can be
    controlled independent of the source's native vertex spacing. Drops
    `normals` (would need recomputing against the new vertices, and
    callers that need them already recompute from local tangents when
    absent — see domain._outward_normals).

    If `geometry.ordered` is False (e.g. a mask-pixel-adjacency scatter),
    there is no real polyline to interpolate along — warns loudly and
    falls back to `order_counterclockwise` per segment before resampling,
    rather than silently treating raster scan order as if it were
    boundary order.
    """
    segments = geometry.segments
    if not geometry.ordered:
        warnings.warn(
            f"\n{_UNORDERED_WARNING_BANNER}\n"
            f"UNORDERED GEOMETRY ({geometry.product!r}) IS BEING RESAMPLED AS IF IT "
            "WERE A POLYLINE.\n"
            "Its points have no natural walking order (e.g. mask-pixel-adjacency "
            "output comes out in raster scan order, not boundary order) — "
            "falling back to sorting each segment counterclockwise around its "
            "centroid before interpolating. This is a heuristic, not a real "
            "boundary reconstruction: it is only exact for a star-shaped point "
            "cloud, and can silently produce a nonsensical zigzag for a long, "
            "curvy, non-loop arc. VERIFY THE RESULT (e.g. plot it) BEFORE "
            "TRUSTING IT.\n"
            f"{_UNORDERED_WARNING_BANNER}",
            stacklevel=2,
        )
        segments = [order_counterclockwise(seg) for seg in segments]

    return Geometry(
        segments=[resample_polyline(seg, spacing_m) for seg in segments],
        product=f"{geometry.product}_resampled_{spacing_m:g}m",
        epoch=geometry.epoch,
        normals=None,
        ordered=True,
    )


def mask_to_runs(segment: np.ndarray, keep_mask: np.ndarray, min_len: int = 2) -> list[np.ndarray]:
    """Split `segment` into contiguous runs of `keep_mask`, dropping runs
    shorter than `min_len`. `segment` is an open (non-wrapping) polyline —
    see `circular_runs` for the closed-ring equivalent. The criterion for
    `keep_mask` is the caller's — distance to some other geometry
    (`contiguous_runs_within`), real velocity magnitude
    (`domain._filter_gl_by_flow`), or anything else."""
    runs = []
    start = None
    for i, keep in enumerate(keep_mask):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            if i - start >= min_len:
                runs.append(segment[start:i])
            start = None
    if start is not None and len(segment) - start >= min_len:
        runs.append(segment[start:])
    return runs


def contiguous_runs_within(segment: np.ndarray, geom, tol_m: float, min_len: int = 2) -> list[np.ndarray]:
    """Split `segment` into contiguous point runs within `tol_m` of `geom`.

    A single raw polyline (e.g. one continuous coastline trace, or a
    basin's own boundary ring) may only pass near the target geometry for
    a short stretch — keep just that stretch, not the whole original.
    Assumes `segment` is ordered (see Geometry.ordered) — for an unordered
    scatter, "contiguous" has no meaning; filter pointwise instead.
    """
    points = shapely.points(segment[:, 0], segment[:, 1])
    dist = shapely.distance(points, geom)
    return mask_to_runs(segment, dist <= tol_m, min_len)


def circular_runs(ring: np.ndarray, keep_mask: np.ndarray, min_len: int = 2) -> list[np.ndarray]:
    """Contiguous runs of True in `keep_mask` around a closed ring
    (ring[0] == ring[-1]), handling wraparound across the closure point.

    Operates on the `m = len(ring) - 1` unique circular points (`ring[-1]`
    duplicates `ring[0]`) by rotating the array to start right after a
    False point first, so no run ever needs to "wrap past the end" during
    the scan — a previous version instead let a run that reached the end
    of the array assume it simply continued for another `len(ring)`
    points without checking, which silently pulled in an unrelated False
    stretch elsewhere in the array (confirmed for Byrd: a trailing "far"
    run reaching the ring's last index swallowed the entire rest of the
    ring, including a real near-GL excluded stretch nowhere near it,
    misclassifying points ~3-150m from the grounding line as cut
    boundary). Rotating first means a True run can never cross the seam,
    since the seam is deliberately placed at a False point.
    """
    m = len(ring) - 1
    unique_mask = keep_mask[:m]
    if unique_mask.all():
        return [ring.copy()] if m >= min_len else []
    if not unique_mask.any():
        return []

    rot = np.flatnonzero(~unique_mask)[0] + 1
    rotated_mask = np.concatenate([unique_mask[rot:], unique_mask[:rot]])
    rotated_ring = np.concatenate([ring[rot:m], ring[:rot]])

    runs = []
    start = None
    for i in range(m):
        keep = rotated_mask[i]
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            if i - start >= min_len:
                runs.append(rotated_ring[start:i])
            start = None
    if start is not None and m - start >= min_len:
        runs.append(rotated_ring[start:m])
    return runs
