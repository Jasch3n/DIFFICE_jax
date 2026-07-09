"""Ice surface elevation providers.

Surface elevation is its own data kind, independent of thickness, even
though a given provider may read it from the same underlying file as
thickness (e.g. BEDMAP1's `surface_altitude` column, same rows as its
`land_ice_thickness` column). It is never derived *from* thickness — see
docs/adr/0001-surface-elevation-independent-coordinates.md for why it
keeps its own native x/y rather than being resampled onto a thickness
role's grid.
"""

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import PointObservations
from joint_xpinn_data.data_sources.thickness import (
    _BEDMAP_SURFACE_COL,
    process_bedmachine_grid,
    process_bedmap_column,
)


def process_bedmachine_surface(config: PipelineConfig, region_polygon, **kwargs) -> PointObservations:
    return process_bedmachine_grid(config, region_polygon, fields=("surface",), **kwargs)


def process_bedmap1_surface(config: PipelineConfig, region_polygon) -> PointObservations:
    return process_bedmap_column(
        config, region_polygon, _BEDMAP_SURFACE_COL, "surface",
        path_key="bedmap1_csv", product="bedmap1_1966-2000", epoch="1966-2000",
    )


def process_custom_points(config: PipelineConfig, region_polygon, **kwargs) -> PointObservations:
    from joint_xpinn_data.data_sources.velocity import process_custom_points as _custom

    return _custom(config, region_polygon, **kwargs)


SURFACE_SOURCES = {
    "bedmachine_v3": process_bedmachine_surface,
    "bedmap1_csv": process_bedmap1_surface,
    "custom_xy": process_custom_points,
}


def _load(config: PipelineConfig, region_polygon, source: str, kwargs: dict) -> PointObservations:
    try:
        fn = SURFACE_SOURCES[source]
    except KeyError:
        raise KeyError(
            f"Unknown surface source {source!r}. Available: {sorted(SURFACE_SOURCES)}"
        )
    return fn(config, region_polygon, **kwargs)


def load_surface(config: PipelineConfig, region_polygon) -> PointObservations:
    """The configured surface source. build_dataset._build_region derives
    BOTH surface output roles from this single PointObservations: s_dense
    (resampled onto the velocity (xd,yd) grid) and sd/xd_s/yd_s (the
    source's own native points, independent of the thickness grid — see
    docs/adr/0001-surface-elevation-independent-coordinates.md)."""
    return _load(config, region_polygon, config.surface_source, config.surface_kwargs)
