"""Calving-front (ice-shelf seaward edge) position providers.

`process_bedmachine_mask` is the default and the one to reach for: it
finds the transition between floating_ice and ocean directly in
BedMachine's mask, so it's correct by construction regardless of how many
different tributary glaciers' grounding lines touch the shelf.

`process_antarctic_boundaries_mask` does the same lookup against a
different, independent mask (Antarctic-boundaries/data/Mask_Antactica_v2.tif,
values 0=ocean/125=isiceshelf/255=isgrounded, dated 2017 per its TIFF
tag — i.e. an older product than BedMachine v3's ~2021 composite, and
notably from *before* Amery's 2019 D28 calving event). Useful for
cross-checking `process_bedmachine_mask` and for seeing how much a real
ice front moved between the two products' epochs.

`process_velocity_mask` is a third, independent cross-check that doesn't
use any boundary/mask product at all: it finds where MEaSURES velocity
coverage itself transitions from valid (CNT>0) to no-data. Coherent InSAR
phase generally can't be tracked over open water or drifting sea ice, so
this transition tends to trace the same ice edge — but it can also pick
up unrelated no-data gaps (poor coherence over rugged terrain, swath
edges), so treat it as a check on the other two, not a default.

`process_measures_front` is kept as an alternative but has a real
limitation worth knowing about: it derives the front by subtracting only
the *one* grounding line passed in (whichever grounding zone the config
selected) from the shelf polygon boundary. If the shelf is fed by other
tributaries too (e.g. Amery is fed by Lambert, Mellor, *and* Fisher), the
boundary arcs adjacent to those other grounding lines are never excluded
and get misclassified as open-ocean calving front. Use it only when you
know the shelf has a single grounding zone, or don't have gridded
ice/ocean mask data for it.
"""

import netCDF4
import numpy as np
import shapely

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import Geometry
from joint_xpinn_data.data_sources import boundaries
from joint_xpinn_data.data_sources.velocity import _index_slice
from joint_xpinn_data.utils.geometry_utils import circular_runs, largest_ring
from joint_xpinn_data.utils.raster_utils import (
    largest_connected_component,
    mask_boundary_points,
    polygon_from_level_set,
    read_geotiff_crop,
)


def _bedmachine_masks(config: PipelineConfig, pad_km: float = 20.0):
    """(x_sub, y_sub, is_shelf, is_ocean) for the named shelf's largest
    connected floating-ice blob in the BedMachine mask crop, and the ocean
    mask. Shared by the calving-front source (its floating/ocean transition)
    and the front-extent source (the floating region the front bounds)."""
    path = str(config.path("bedmachine"))
    shelf_poly = boundaries.get_named_polygon(str(config.path("iceshelves")), config.ice_shelf)
    bx0, by0, bx1, by1 = shelf_poly.bounds
    pad_m = pad_km * 1000.0

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0 - pad_m, bx1 + pad_m)
        ys = _index_slice(y, by0 - pad_m, by1 + pad_m)
        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        mask = np.asarray(ds.variables["mask"][ys, xs])

    # flag_meanings: ocean, ice_free_land, grounded_ice, floating_ice, lake_vostok
    # Restricted to the largest connected floating_ice blob in the crop —
    # the bbox-based crop can also catch unrelated floating ice (a
    # neighboring shelf/tongue, sea ice) that isn't part of this named shelf
    # at all; see largest_connected_component's docstring.
    is_shelf = largest_connected_component(mask == 3)
    is_ocean = mask == 0
    return x_sub, y_sub, is_shelf, is_ocean


def process_bedmachine_mask(
    config: PipelineConfig,
    region_polygon,
    grounding_line: Geometry,
    pad_km: float = 20.0,
) -> Geometry:
    """Floating_ice pixels adjacent to an ocean pixel in the BedMachine mask.

    Normals are computed directly from which neighbor pixel is ocean (not
    from a tangent estimate), since the mask already tells us exactly
    which side is open water — that's more accurate and doesn't need an
    ordered polyline, so the result is one unordered point set.
    """
    x_sub, y_sub, is_shelf, is_ocean = _bedmachine_masks(config, pad_km)

    # Pad with False so the crop edge itself is never mistaken for a real
    # ocean transition (pad_km is generous specifically so real ice never
    # reaches the crop boundary).
    points, normals = mask_boundary_points(x_sub, y_sub, is_shelf, is_ocean)
    if len(points) == 0:
        raise ValueError(
            f"No floating_ice/ocean transition found for {config.ice_shelf!r} "
            "in the BedMachine mask — check ice_shelf name and pad_km."
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


def _antarctic_boundaries_masks(config: PipelineConfig, pad_km: float = 20.0):
    """(x_sub, y_sub, is_shelf, is_ocean) for the named shelf's largest
    connected isiceshelf blob in Mask_Antactica_v2.tif and the isopenocean
    mask. Shared by the calving-front source and the front-extent source."""
    path = str(config.path("antarctic_boundaries_mask"))
    shelf_poly = boundaries.get_named_polygon(str(config.path("iceshelves")), config.ice_shelf)
    bx0, by0, bx1, by1 = shelf_poly.bounds
    pad_m = pad_km * 1000.0

    x_sub, y_sub, arr = read_geotiff_crop(path, bx0, by0, bx1, by1, pad_m)
    # Same largest-connected-blob restriction as process_bedmachine_mask,
    # for the same reason.
    is_shelf = largest_connected_component(arr == 125)
    is_ocean = arr == 0
    return x_sub, y_sub, is_shelf, is_ocean


def process_antarctic_boundaries_mask(
    config: PipelineConfig,
    region_polygon,
    grounding_line: Geometry,
    pad_km: float = 20.0,
) -> Geometry:
    """Isiceshelf pixels adjacent to an isopenocean pixel in the
    Antarctic-boundaries raster mask (see module docstring for provenance).
    """
    x_sub, y_sub, is_shelf, is_ocean = _antarctic_boundaries_masks(config, pad_km)
    points, normals = mask_boundary_points(x_sub, y_sub, is_shelf, is_ocean)
    if len(points) == 0:
        raise ValueError(
            f"No isiceshelf/isopenocean transition found for {config.ice_shelf!r} "
            "in Mask_Antactica_v2.tif — check ice_shelf name and pad_km."
        )

    return Geometry(
        segments=[points],
        product="antarctic_boundaries_mask",
        epoch="2017",
        normals=[normals],
        ordered=False,
    )


def _velocity_valid_masks(config: PipelineConfig, pad_km: float = 20.0):
    """(x_sub, y_sub, is_valid, is_nodata) for the named shelf's largest
    connected valid-velocity (CNT>0) blob and the outer no-data (CNT==0)
    region. Shared by the calving-front source and the front-extent source.

    Unlike process_bedmachine_mask's `mask == 3` (floating ice, naturally
    confined to the named shelf), CNT>0 velocity coverage is connected
    across the whole continent — so an unrestricted largest_connected_component
    grabs all of Antarctica's coverage inside the (potentially huge) shelf
    bounding box, and its valid/no-data transitions trace far-away coastlines
    and interior data-gap edges, not the shelf front. Confirmed for Ross
    East: ~76% of the transition points landed nowhere near the front, a
    ~625km median disagreement with the other front sources (see HANDOFF.md).
    Two restrictions fix this without needing a shelf-specific mask category:
      (1) keep only CNT>0 pixels inside the named shelf polygon (dilated by
          pad_m so the front, which can sit just seaward of the digitized
          polygon, isn't clipped), then take the largest blob — the shelf's
          own coverage, not the continent's;
      (2) count only the OUTER ocean (largest CNT==0 component) as the
          no-data side, so interior coverage gaps inside the shelf don't each
          become a spurious front point.
    The GL side produces no points either way: there the shelf-local valid
    blob abuts grounded ice (CNT>0, not no-data), not ocean.
    """
    path = str(config.path("measures_velocity"))
    shelf_poly = boundaries.get_named_polygon(str(config.path("iceshelves")), config.ice_shelf)
    bx0, by0, bx1, by1 = shelf_poly.bounds
    pad_m = pad_km * 1000.0

    with netCDF4.Dataset(path) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        xs = _index_slice(x, bx0 - pad_m, bx1 + pad_m)
        ys = _index_slice(y, by0 - pad_m, by1 + pad_m)
        x_sub = np.asarray(x[xs])
        y_sub = np.asarray(y[ys])
        cnt = np.asarray(ds.variables["CNT"][ys, xs])

    yi, xi = np.nonzero(cnt > 0)
    in_shelf = shapely.contains(shelf_poly.buffer(pad_m), shapely.points(x_sub[xi], y_sub[yi]))
    shelf_valid = np.zeros(cnt.shape, dtype=bool)
    shelf_valid[yi[in_shelf], xi[in_shelf]] = True
    is_valid = largest_connected_component(shelf_valid)
    is_nodata = largest_connected_component(cnt == 0)
    return x_sub, y_sub, is_valid, is_nodata


def process_velocity_mask(
    config: PipelineConfig,
    region_polygon,
    grounding_line: Geometry,
    pad_km: float = 20.0,
) -> Geometry:
    """Valid (CNT>0) velocity pixels adjacent to a no-data (CNT==0) pixel —
    the ice edge as implied by the velocity data's own coverage boundary,
    independent of any boundary/mask product. See _velocity_valid_masks.

    Same CNT>0 validity convention as velocity.process_measures_velocity
    (VX/VY's own _FillValue=0 collides with real slow ice, CNT doesn't).
    """
    x_sub, y_sub, is_valid, is_nodata = _velocity_valid_masks(config, pad_km)
    points, normals = mask_boundary_points(x_sub, y_sub, is_valid, is_nodata)
    if len(points) == 0:
        raise ValueError(
            f"No valid/no-data velocity transition found for {config.ice_shelf!r} "
            "in the MEaSURES velocity CNT field — check ice_shelf name and pad_km."
        )

    return Geometry(
        segments=[points],
        product="measures_v2_velocity_mask",
        epoch="1995-2018 InSAR composite",
        normals=[normals],
        ordered=False,
    )


def process_measures_front(
    config: PipelineConfig,
    region_polygon,
    grounding_line: Geometry,
    gl_exclusion_km: float = 5.0,
) -> Geometry:
    """Ice-shelf polygon boundary, minus the arc adjacent to `grounding_line`
    only — see the module docstring for why this misses other tributaries."""
    shelf_poly = boundaries.get_named_polygon(
        str(config.path("iceshelves")), config.ice_shelf
    )
    ring = largest_ring(shelf_poly)
    gl_points = shapely.points(*grounding_line.all_points().T)
    gl_multi = shapely.multipoints(gl_points)

    tol_m = gl_exclusion_km * 1000.0
    ring_points = shapely.points(ring[:, 0], ring[:, 1])
    dist_to_gl = shapely.distance(ring_points, gl_multi)
    far_from_gl = dist_to_gl > tol_m

    runs = circular_runs(ring, far_from_gl)

    if not runs:
        raise ValueError(
            f"No calving-front points found for {config.ice_shelf!r} after "
            f"excluding the {gl_exclusion_km} km grounding-line buffer — "
            "the whole shelf boundary might be near the grounding line, or "
            "gl_exclusion_km is too large."
        )
    return Geometry(segments=runs, product="measures_boundaries_2008", epoch="2007-2009")


def process_custom_geometry(
    config: PipelineConfig, region_polygon, grounding_line: Geometry, **kwargs
) -> Geometry:
    from joint_xpinn_data.data_sources.grounding_line import (
        process_custom_geometry as _custom,
    )

    return _custom(config, region_polygon, **kwargs)


FRONT_SOURCES = {
    "bedmachine_mask": process_bedmachine_mask,
    "antarctic_boundaries_mask": process_antarctic_boundaries_mask,
    "velocity_mask": process_velocity_mask,
    "measures_boundaries_2008": process_measures_front,
    "custom_xy": process_custom_geometry,
}


def load_calving_front(config: PipelineConfig, region_polygon, grounding_line: Geometry) -> Geometry:
    try:
        fn = FRONT_SOURCES[config.calving_front_source]
    except KeyError:
        raise KeyError(
            f"Unknown calving_front_source {config.calving_front_source!r}. "
            f"Available: {sorted(FRONT_SOURCES)}"
        )
    return fn(config, region_polygon, grounding_line, **config.calving_front_kwargs)


def _extent_from_mask(x_sub, y_sub, is_valid):
    """Outer boundary of a boolean ice/valid mask as a solid shapely Polygon
    (largest marching-squares loop — smooth, not pixel-blocky; see
    polygon_from_level_set). Interior holes are deliberately NOT subtracted
    here: this extent is used only to trim the floating region's *seaward*
    margin to the front source, while genuine interior ice-rise holes are
    restored separately in domain.build_regions. Empty polygon if the mask
    has no closed contour."""
    return polygon_from_level_set(x_sub, y_sub, is_valid.astype(float), 0.5)


def bedmachine_extent(config: PipelineConfig, region_polygon, grounding_line: Geometry, pad_km: float = 20.0, **_):
    x_sub, y_sub, is_shelf, _ocean = _bedmachine_masks(config, pad_km)
    return _extent_from_mask(x_sub, y_sub, is_shelf)


def antarctic_boundaries_extent(config: PipelineConfig, region_polygon, grounding_line: Geometry, pad_km: float = 20.0, **_):
    x_sub, y_sub, is_shelf, _ocean = _antarctic_boundaries_masks(config, pad_km)
    return _extent_from_mask(x_sub, y_sub, is_shelf)


def velocity_extent(config: PipelineConfig, region_polygon, grounding_line: Geometry, pad_km: float = 20.0, **_):
    x_sub, y_sub, is_valid, _nodata = _velocity_valid_masks(config, pad_km)
    return _extent_from_mask(x_sub, y_sub, is_valid)


# Each mask-based front source has an associated ice/valid *region* (the
# floating ice or valid-velocity blob the front bounds); the line-based
# sources (measures_boundaries_2008, custom_xy) have none and are absent
# here, so their floating region isn't clipped (its extent stays the 2008
# shelf polygon). Mirrors FRONT_SOURCES so a config's calving_front_source
# choice drives the clip extent too — see domain.build_regions.
FRONT_EXTENT_SOURCES = {
    "bedmachine_mask": bedmachine_extent,
    "antarctic_boundaries_mask": antarctic_boundaries_extent,
    "velocity_mask": velocity_extent,
}


def load_front_extent(config: PipelineConfig, region_polygon, grounding_line: Geometry):
    """The floating/valid-ice region the configured calving front bounds, as
    a shapely Polygon, or None if the source has no associated raster extent
    (line-based fronts). domain.build_regions intersects the floating region
    with this so its ocean-facing edge coincides with the front source's own
    extent — resolving the 2008-shelf-polygon-vs-current-mask margin mismatch
    extensibly, per whichever calving_front_source the config selects."""
    fn = FRONT_EXTENT_SOURCES.get(config.calving_front_source)
    if fn is None:
        return None
    return fn(config, region_polygon, grounding_line, **config.calving_front_kwargs)
