"""Ice velocity providers.

MEaSURES InSAR-derived velocity (`_FillValue=0` for VX/VY, which collides
with real slow-moving grounded ice — CNT, the per-pixel observation count,
is the correct no-data indicator and is what we filter on instead).
"""

import netCDF4
import numpy as np
import shapely

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import PointObservations


def _index_slice(coord: np.ndarray, lo: float, hi: float, pad: int = 2) -> slice:
    """Index range covering [lo, hi] in a monotonic (ascending or
    descending) 1-D coordinate array, with a small pad."""
    ascending = coord[1] > coord[0]
    key = coord if ascending else -coord
    a, b = (lo, hi) if ascending else (-hi, -lo)
    i0, i1 = np.searchsorted(key, [a, b])
    i0 = max(i0 - pad, 0)
    i1 = min(i1 + pad, len(coord))
    return slice(i0, i1)


def process_measures_velocity(config: PipelineConfig, region_polygon) -> PointObservations:
    path = str(config.path("measures_velocity"))
    bx0, by0, bx1, by1 = region_polygon.bounds

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0, bx1)
        ys = _index_slice(y, by0, by1)

        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        vx = np.asarray(ds.variables["VX"][ys, xs])
        vy = np.asarray(ds.variables["VY"][ys, xs])
        stdx = np.asarray(ds.variables["STDX"][ys, xs])
        stdy = np.asarray(ds.variables["STDY"][ys, xs])
        cnt = np.asarray(ds.variables["CNT"][ys, xs])

    xx, yy = np.meshgrid(x_sub, y_sub)
    valid = cnt > 0

    pts_in_bbox = shapely.points(xx[valid], yy[valid])
    inside = shapely.contains(region_polygon, pts_in_bbox)

    x_flat = xx[valid][inside]
    y_flat = yy[valid][inside]
    u = vx[valid][inside] / 365.25 / 86400.0  # m/yr -> m/s, matching SSA-residual convention in DIFFICE_jax
    v = vy[valid][inside] / 365.25 / 86400.0
    sigma = np.sqrt(stdx[valid][inside] ** 2 + stdy[valid][inside] ** 2) / 365.25 / 86400.0

    return PointObservations(
        x=x_flat,
        y=y_flat,
        values={"u": u, "v": v},
        weight=sigma,
        product="measures_v2",
        epoch="1995-2018 InSAR composite",
    )


def load_velocity_grid(config: PipelineConfig, region_polygon):
    """Raw MEaSURES VX/VY/CNT grid cropped to `region_polygon`'s bounds —
    NOT filtered to CNT>0 or clipped to the polygon shape — plus a
    boolean `inside` mask for polygon containment. The raster counterpart
    to `load_velocity`'s point cloud: needed wherever a true
    pixel-adjacency mask boundary is required (see
    `build_dataset._accepted_velocity_mask`/`_dirichlet_from_mask`) rather
    than a point cloud to nearest-neighbor-match against.

    Grid-specific: only meaningful for `measures_v2` (the only registered
    velocity source that's a real grid) — raises for anything else, since
    a `custom_xy` point list has no grid to build a raster mask from.
    """
    if config.velocity_source != "measures_v2":
        raise ValueError(
            f"load_velocity_grid needs a real grid, but velocity_source={config.velocity_source!r} "
            "has none (only 'measures_v2' does)."
        )
    path = str(config.path("measures_velocity"))
    bx0, by0, bx1, by1 = region_polygon.bounds

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0, bx1)
        ys = _index_slice(y, by0, by1)
        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        vx = np.asarray(ds.variables["VX"][ys, xs])
        vy = np.asarray(ds.variables["VY"][ys, xs])
        cnt = np.asarray(ds.variables["CNT"][ys, xs])

    xx, yy = np.meshgrid(x_sub, y_sub)
    inside = shapely.contains(region_polygon, shapely.points(xx.ravel(), yy.ravel())).reshape(xx.shape)
    u = vx / 365.25 / 86400.0  # m/yr -> m/s, matching load_velocity's convention
    v = vy / 365.25 / 86400.0
    return x_sub, y_sub, u, v, cnt, inside


def process_custom_points(config: PipelineConfig, region_polygon, **kwargs) -> PointObservations:
    """Escape hatch: kwargs must supply x, y, and a `values` dict directly
    (already in EPSG:3031 meters and m/s). Points are still cropped to
    `region_polygon` so callers don't have to pre-clip."""
    x = np.asarray(kwargs["x"], dtype=float)
    y = np.asarray(kwargs["y"], dtype=float)
    values = {k: np.asarray(v, dtype=float) for k, v in kwargs["values"].items()}
    weight = kwargs.get("weight")
    if weight is not None:
        weight = np.asarray(weight, dtype=float)

    inside = shapely.contains(region_polygon, shapely.points(x, y))
    return PointObservations(
        x=x[inside],
        y=y[inside],
        values={k: v[inside] for k, v in values.items()},
        weight=weight[inside] if weight is not None else None,
        product=kwargs.get("product", "custom_xy"),
        epoch=kwargs.get("epoch"),
    )


VELOCITY_SOURCES = {
    "measures_v2": process_measures_velocity,
    "custom_xy": process_custom_points,
}


def load_velocity(config: PipelineConfig, region_polygon) -> PointObservations:
    try:
        fn = VELOCITY_SOURCES[config.velocity_source]
    except KeyError:
        raise KeyError(
            f"Unknown velocity_source {config.velocity_source!r}. "
            f"Available: {sorted(VELOCITY_SOURCES)}"
        )
    return fn(config, region_polygon, **config.velocity_kwargs)
