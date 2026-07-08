"""Grounding-line position providers.

Each provider returns a Geometry clipped to the vicinity of the requested
region polygon (the chosen grounding-zone basin). Providers are looked up
by name via GL_SOURCES so a future product (ICESat-2-derived, etc.) plugs
in without touching the pipeline.

`process_measures_gl` is the default; `process_bedmachine_mask` reads the
grounded_ice/floating_ice transition directly from BedMachine's mask
instead, as an independent cross-check — useful for seeing how much the
two products disagree on where the grounding line actually is.
"""

from functools import lru_cache

import netCDF4
import numpy as np
import scipy.io as sio
import shapely

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import Geometry, split_on_nan
from joint_xpinn_data.data_sources.velocity import _index_slice
from joint_xpinn_data.utils.geometry_utils import contiguous_runs_within
from joint_xpinn_data.utils.raster_utils import mask_boundary_points


@lru_cache(maxsize=None)
def _load_continental_gl(mat_path: str) -> list[np.ndarray]:
    d = sio.loadmat(mat_path, simplify_cells=True)
    x = np.asarray(d["x"], dtype=float).flatten()
    y = np.asarray(d["y"], dtype=float).flatten()
    return split_on_nan(x, y)


def process_measures_gl(
    config: PipelineConfig, region_polygon, tol_km: float = 10.0
) -> Geometry:
    """MEaSURES Antarctic Boundaries 2007-2009 grounding line, clipped to
    the segments running along `region_polygon`'s boundary.

    The basin's downstream edge *is* the grounding line by construction in
    this dataset, so segments within `tol_km` of the basin boundary are
    kept; everything else (other basins' grounding lines, thousands of km
    away) is dropped.
    """
    segments = _load_continental_gl(str(config.path("groundingline")))
    boundary = region_polygon.boundary
    tol_m = tol_km * 1000.0
    bx0, by0, bx1, by1 = region_polygon.bounds

    kept = []
    for seg in segments:
        # cheap bbox pre-filter before the pointwise distance check — most
        # of the 657 continental segments are nowhere near this region
        xmin, ymin = seg.min(axis=0)
        xmax, ymax = seg.max(axis=0)
        if xmax < bx0 - tol_m or xmin > bx1 + tol_m:
            continue
        if ymax < by0 - tol_m or ymin > by1 + tol_m:
            continue
        kept.extend(contiguous_runs_within(seg, boundary, tol_m))

    if not kept:
        raise ValueError(
            f"No grounding-line segments found within {tol_km} km of the "
            f"region polygon (bounds={region_polygon.bounds}). Check "
            "region_polygon or increase tol_km."
        )
    return Geometry(segments=kept, product="measures_boundaries_2008", epoch="2007-2009")


def process_bedmachine_mask(
    config: PipelineConfig, region_polygon, pad_km: float = 20.0, tol_km: float = 10.0
) -> Geometry:
    """Grounded_ice/floating_ice transition read directly from BedMachine's
    mask, filtered to within `tol_km` of `region_polygon`'s boundary.

    The tol_km filter matters here for the same reason it does in
    process_measures_gl: the crop window is sized to the basin's own
    bounding box, which for a large basin can span hundreds of km and
    pick up unrelated nearby glaciers' grounding lines too.
    """
    path = str(config.path("bedmachine"))
    bx0, by0, bx1, by1 = region_polygon.bounds
    pad_m = pad_km * 1000.0

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0 - pad_m, bx1 + pad_m)
        ys = _index_slice(y, by0 - pad_m, by1 + pad_m)
        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        mask = np.asarray(ds.variables["mask"][ys, xs])

    is_grounded = mask == 2
    is_floating = mask == 3
    points, normals = mask_boundary_points(x_sub, y_sub, is_grounded, is_floating)
    if len(points) == 0:
        raise ValueError(
            f"No grounded_ice/floating_ice transition found near "
            f"{config.grounding_zone!r} in the BedMachine mask — check "
            "grounding_zone and pad_km."
        )

    tol_m = tol_km * 1000.0
    dist = shapely.distance(shapely.points(points[:, 0], points[:, 1]), region_polygon.boundary)
    keep = dist <= tol_m
    points, normals = points[keep], normals[keep]
    if len(points) == 0:
        raise ValueError(
            f"BedMachine mask grounding-line pixels exist near "
            f"{config.grounding_zone!r} but none are within {tol_km} km of "
            "the region polygon boundary — increase tol_km or check the "
            "basin polygon."
        )

    return Geometry(
        segments=[points],
        product="bedmachine_v3_mask",
        epoch="2021 composite",
        normals=[normals],
        # Raster scan order (row-major over mask pixels), not boundary
        # order — see Geometry.ordered's docstring.
        ordered=False,
    )


def process_custom_geometry(config: PipelineConfig, region_polygon, **kwargs) -> Geometry:
    """Escape hatch for any future GL/front product.

    kwargs may supply either:
      - `xy`: an (N,2) array or list of (Ni,2) arrays (already split into
        segments), or
      - `xy_path`: a path to a .csv/.npy file with two columns x,y (NaN
        rows are treated as segment separators, matching the convention
        used elsewhere in this package).
    `product`/`epoch` should be supplied so provenance is recorded.
    """
    xy = kwargs.get("xy")
    xy_path = kwargs.get("xy_path")
    if xy is None and xy_path is None:
        raise ValueError("custom_xy source requires 'xy' or 'xy_path' in kwargs")

    if xy is not None:
        if isinstance(xy, np.ndarray):
            segments = split_on_nan(xy[:, 0], xy[:, 1])
        else:
            segments = [np.asarray(seg, dtype=float) for seg in xy]
    else:
        arr = np.loadtxt(xy_path, delimiter=",")
        segments = split_on_nan(arr[:, 0], arr[:, 1])

    return Geometry(
        segments=segments,
        product=kwargs.get("product", "custom_xy"),
        epoch=kwargs.get("epoch"),
    )


GL_SOURCES = {
    "measures_boundaries_2008": process_measures_gl,
    "bedmachine_mask": process_bedmachine_mask,
    "custom_xy": process_custom_geometry,
}


def load_grounding_line(config: PipelineConfig, region_polygon) -> Geometry:
    try:
        fn = GL_SOURCES[config.grounding_line_source]
    except KeyError:
        raise KeyError(
            f"Unknown grounding_line_source {config.grounding_line_source!r}. "
            f"Available: {sorted(GL_SOURCES)}"
        )
    return fn(config, region_polygon, **config.grounding_line_kwargs)
