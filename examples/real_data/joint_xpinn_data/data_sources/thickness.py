"""Ice thickness (and surface/bed) providers.

BedMachine and BEDMAP1 need genuinely different processing, not just
different file readers:
  - BedMachine is a dense, gridded, multi-field product (thickness,
    surface, bed all share one grid and one ice/ocean mask) — QC is a mask
    lookup and an index-range crop.
  - BEDMAP1 is sparse scattered radar-flight-line data in lon/lat with
    per-column -9999 sentinels and no gridded mask — QC is sentinel
    filtering, and it additionally needs reprojection (lon/lat -> EPSG:3031)
    that BedMachine doesn't.
"""

from functools import lru_cache

import netCDF4
import numpy as np
import pandas as pd
import pyproj
import shapely

from joint_xpinn_data.config import BEDMAP1_CRS, CRS, PipelineConfig
from joint_xpinn_data.contracts import PointObservations
from joint_xpinn_data.data_sources.velocity import _index_slice

# mask flag_values: 0 ocean, 1 ice_free_land, 2 grounded_ice, 3 floating_ice, 4 lake_vostok
_BEDMACHINE_ICE_MASK_VALUES = (2, 3)


def process_bedmachine_grid(
    config: PipelineConfig,
    region_polygon,
    fields: tuple[str, ...] = ("thickness", "surface", "bed"),
) -> PointObservations:
    path = str(config.path("bedmachine"))
    bx0, by0, bx1, by1 = region_polygon.bounds

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0, bx1)
        ys = _index_slice(y, by0, by1)

        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        mask = np.asarray(ds.variables["mask"][ys, xs])
        field_arrays = {f: np.asarray(ds.variables[f][ys, xs]) for f in fields}

    xx, yy = np.meshgrid(x_sub, y_sub)
    is_ice = np.isin(mask, _BEDMACHINE_ICE_MASK_VALUES)

    pts = shapely.points(xx[is_ice], yy[is_ice])
    inside = shapely.contains(region_polygon, pts)

    x_flat = xx[is_ice][inside]
    y_flat = yy[is_ice][inside]
    values = {f: arr[is_ice][inside] for f, arr in field_arrays.items()}

    return PointObservations(
        x=x_flat,
        y=y_flat,
        values=values,
        weight=None,
        product="bedmachine_v3",
        epoch="2021 composite",
    )


# Column names and -9999 sentinel are shared across the whole BEDMAP
# series' CSV exports (BEDMAP1 and BEDMAP2 both use them), not unique to
# BEDMAP1 — hence the schema-generic naming here.
_BEDMAP_LON_COL = "longitude (degree_east)"
_BEDMAP_LAT_COL = "latitude (degree_north)"
_BEDMAP_THICKNESS_COL = "land_ice_thickness (m)"
_BEDMAP_SURFACE_COL = "surface_altitude (m)"
_BEDMAP_BED_COL = "bedrock_altitude (m)"
_BEDMAP_SENTINEL = -9999
_BEDMAP_VALUE_COLS = (_BEDMAP_THICKNESS_COL, _BEDMAP_SURFACE_COL, _BEDMAP_BED_COL)


@lru_cache(maxsize=4)
def _read_bedmap_csv(path: str) -> pd.DataFrame:
    """Cached raw read of one BEDMAP-family CSV's value columns (e.g.
    BEDMAP1's ~1.9M rows). Thickness and surface elevation are independent
    data kinds (see data_sources/surface.py) that can both default to the
    same file, each read once per region — without this cache, a
    two-region build_dataset call would re-parse the full CSV from disk up
    to 4 times."""
    cols = [_BEDMAP_LON_COL, _BEDMAP_LAT_COL, *_BEDMAP_VALUE_COLS]
    return pd.read_csv(path, comment="#", usecols=cols)


def process_bedmap_column(
    config: PipelineConfig,
    region_polygon,
    value_col: str,
    value_key: str,
    path_key: str,
    product: str,
    epoch: str,
) -> PointObservations:
    """Read one BEDMAP-family scalar column (thickness, surface, or bed)
    at its own native lon/lat rows, filtered and reprojected the same way
    regardless of which column or which BEDMAP source file is being read
    — thickness and surface elevation are independent data kinds that
    just happen to share a file (and, incidentally, its rows) for a given
    BEDMAP source."""
    path = str(config.path(path_key))
    df = _read_bedmap_csv(path)

    valid = (
        (df[_BEDMAP_LON_COL] != _BEDMAP_SENTINEL)
        & (df[_BEDMAP_LAT_COL] != _BEDMAP_SENTINEL)
        & (df[value_col] != _BEDMAP_SENTINEL)
    )
    df = df.loc[valid]

    # Cheap lon/lat bbox pre-filter before reprojecting — BEDMAP sources
    # cover the whole continent, the target region is a tiny fraction of it.
    transformer = pyproj.Transformer.from_crs(BEDMAP1_CRS, CRS, always_xy=True)
    bx0, by0, bx1, by1 = region_polygon.bounds
    pad_deg = 2.0
    lon_lo, lat_lo = transformer.transform(bx0, by0, direction="INVERSE")
    lon_hi, lat_hi = transformer.transform(bx1, by1, direction="INVERSE")
    lat_min = min(lat_lo, lat_hi) - pad_deg
    lat_max = max(lat_lo, lat_hi) + pad_deg
    df = df.loc[(df[_BEDMAP_LAT_COL] >= lat_min) & (df[_BEDMAP_LAT_COL] <= lat_max)]

    x, y = transformer.transform(df[_BEDMAP_LON_COL].to_numpy(), df[_BEDMAP_LAT_COL].to_numpy())
    pts = shapely.points(x, y)
    inside = shapely.contains(region_polygon, pts)

    return PointObservations(
        x=x[inside],
        y=y[inside],
        values={value_key: df[value_col].to_numpy()[inside]},
        weight=None,
        product=product,
        epoch=epoch,
    )


def process_bedmap1_csv(config: PipelineConfig, region_polygon) -> PointObservations:
    return process_bedmap_column(
        config, region_polygon, _BEDMAP_THICKNESS_COL, "thickness",
        path_key="bedmap1_csv", product="bedmap1_1966-2000", epoch="1966-2000",
    )


def process_bedmap2_csv(config: PipelineConfig, region_polygon) -> PointObservations:
    """BGR's 2002-2003 Prince Charles Mountains / Lambert Glacier airborne
    radar survey (PCMEGA, part of Bedmap2) — same column schema as
    BEDMAP1's CSV, covering the Lambert/Mellor/Fisher grounding-zone area
    more recently, and in places more densely, than BEDMAP1's 1966-2000
    flights."""
    return process_bedmap_column(
        config, region_polygon, _BEDMAP_THICKNESS_COL, "thickness",
        path_key="bedmap2_csv", product="bedmap2_pcmega_2002-2003", epoch="2002-2003",
    )


def process_concat(config: PipelineConfig, region_polygon, sources: list[dict]) -> PointObservations:
    """Concatenate multiple named sources of this kind — e.g. combining
    BEDMAP1 and BEDMAP2 sparse thickness surveys so real coverage gaps in
    one are filled by the other, rather than being limited to whichever
    single source a role is bound to. Each entry in `sources` is
    {"source": <name>, "kwargs": {...}} (kwargs optional)."""
    parts = [_load(config, region_polygon, s["source"], s.get("kwargs", {})) for s in sources]
    keys = set(parts[0].values)
    if any(set(p.values) != keys for p in parts[1:]):
        raise ValueError(f"concat sources have mismatched value keys: {[sorted(p.values) for p in parts]}")
    weights = [p.weight for p in parts]
    weight = None if any(w is None for w in weights) else np.concatenate(weights)
    return PointObservations(
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        values={k: np.concatenate([p.values[k] for p in parts]) for k in keys},
        weight=weight,
        product="+".join(p.product for p in parts),
        epoch="+".join(dict.fromkeys(p.epoch for p in parts if p.epoch)) or None,
    )


def process_bedmachine_thickness(config: PipelineConfig, region_polygon, **kwargs) -> PointObservations:
    """Thickness role only — `surface` is its own data kind, see
    data_sources/surface.py, even though both happen to come from the same
    BedMachine grid read."""
    return process_bedmachine_grid(config, region_polygon, fields=("thickness", "bed"), **kwargs)


def process_custom_points(config: PipelineConfig, region_polygon, **kwargs) -> PointObservations:
    from joint_xpinn_data.data_sources.velocity import process_custom_points as _custom

    return _custom(config, region_polygon, **kwargs)


THICKNESS_SOURCES = {
    "bedmachine_v3": process_bedmachine_thickness,
    "bedmap1_csv": process_bedmap1_csv,
    "bedmap2_csv": process_bedmap2_csv,
    "concat": process_concat,
    "custom_xy": process_custom_points,
}


def _load(config: PipelineConfig, region_polygon, source: str, kwargs: dict) -> PointObservations:
    try:
        fn = THICKNESS_SOURCES[source]
    except KeyError:
        raise KeyError(
            f"Unknown thickness source {source!r}. Available: {sorted(THICKNESS_SOURCES)}"
        )
    return fn(config, region_polygon, **kwargs)


def load_dense_thickness(config: PipelineConfig, region_polygon) -> PointObservations:
    """h_dense role — a dense gridded product, later resampled onto the
    velocity (xd,yd) locations by the caller. Surface elevation's dense
    role (s_dense) is a separate, independent fetch — see
    data_sources/surface.py."""
    return _load(config, region_polygon, config.dense_thickness_source, config.dense_thickness_kwargs)


def load_sparse_thickness(config: PipelineConfig, region_polygon) -> PointObservations:
    """hd/xd_h/yd_h role — kept at this source's own native locations."""
    return _load(config, region_polygon, config.sparse_thickness_source, config.sparse_thickness_kwargs)
