"""Pipeline configuration.

Everything shelf/zone-specific lives in `PipelineConfig`. Everything
file-path/infrastructure-specific lives in `DEFAULT_PATHS`. Adding a new
ice shelf or grounding zone never requires touching source code — only
these values (or ones passed at call time).
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

# Root of the raw observational data (MEaSURES velocity, BedMachine,
# BEDMAP, Antarctic-boundaries, ...). This directory is NOT part of the
# git repo (see CLAUDE.md). Override per machine with the
# DIFFICE_DATA_ROOT environment variable; the fallback is the original
# author's local layout so nothing changes for that machine.
DATA_ROOT = Path(os.environ.get("DIFFICE_DATA_ROOT", "/Users/jiapchen/Research/Data"))

DEFAULT_PATHS = {
    "measures_velocity": DATA_ROOT / "MEaSURES-ice-vel" / "insar_antarctica_ice_velocity_450m_v2.nc",
    "bedmachine": DATA_ROOT / "BedMachineAntarctica-v3.nc",
    "bedmap1_csv": DATA_ROOT / "BEDMAP1_1966-2000_AIR_BM1.csv",
    "bedmap2_csv": DATA_ROOT / "BEDMAP2" / "BGR_2002_PCMEGA_AIR_BM2.csv",
    "basins_refined": DATA_ROOT / "Antarctic-boundaries" / "data" / "basins_refined_v2.mat",
    "iceshelves": DATA_ROOT / "Antarctic-boundaries" / "data" / "iceshelves_2008_v2.mat",
    "groundingline": DATA_ROOT / "Antarctic-boundaries" / "data" / "groundingline_2008_v2.mat",
    "antarctic_boundaries_mask": DATA_ROOT / "Antarctic-boundaries" / "data" / "Mask_Antactica_v2.tif",
}

# EPSG:3031 — Antarctic Polar Stereographic, true scale at 71S. Shared by
# MEaSURES velocity, BedMachine, and the Antarctic-boundaries .mat files.
CRS = "EPSG:3031"

# BEDMAP1 stores lon/lat (WGS84).
BEDMAP1_CRS = "EPSG:4326"


@dataclass
class PipelineConfig:
    ice_shelf: str
    grounding_zone: str
    buffer_km: float

    velocity_source: str = "measures_v2"
    # One source per data kind. thickness_source feeds BOTH thickness
    # output roles: h_dense (resampled onto the velocity (xd,yd) grid) and
    # hd/xd_h/yd_h (the source's own native points — e.g. real BEDMAP radar
    # tracks, which legitimately live at different points than velocity
    # data; see data-README.md in DIFFICE_jax's synthetic_data). A dense
    # gridded source (bedmachine_v3) has no native sparse points distinct
    # from its own grid, so for it BOTH roles are resampled onto (xd,yd) —
    # see build_dataset._build_region.
    thickness_source: str = "bedmachine_v3"
    # surface_source feeds BOTH surface output roles the same way: s_dense
    # (resampled onto (xd,yd)) and sd/xd_s/yd_s (native points). Surface is
    # a fully independent data kind, never derived from thickness, even when
    # both point at the same file — see data_sources/surface.py and
    # docs/adr/0001-surface-elevation-independent-coordinates.md.
    surface_source: str = "bedmachine_v3"
    # Drop grounded-region velocity (and everything resampled/derived from
    # it: xd/yd/ud/vd, xcol/ycol, h_dense/s_dense, ols_d) below this speed.
    # None = no filtering (keep the pre-existing behavior). Only applies to
    # the grounded region — the floating region has no such knob, since
    # slow-moving shelf ice isn't the same kind of noise/near-divide
    # concern this is meant to address.
    grounded_min_speed_myr: float | None = None

    # Which domain.REGION_STRATEGIES entry builds the grounded/floating
    # geometry. "simple" (default) = Amery-class: no GL flow-filtering, no
    # velocity-thresholded grounded corridor, no near-interface smoothing
    # beyond what build_regions' shared epilogue always applies.
    # "flow_restricted" = Byrd-class: GL flow-filtering plus, optionally,
    # a velocity-thresholded grounded corridor — see region_strategy_kwargs
    # below and docs/adr/0002-shelf-class-region-strategy-registry.md.
    region_strategy: str = "simple"
    # Extra kwargs forwarded to the chosen region_strategy function.
    # process_flow_restricted requires `min_speed_myr` (m/yr; a basin's
    # own geometric interface with a shelf can be wider than just the
    # named glacier's fast trunk — e.g. Byrd's is ~76km wide, real speeds
    # 2-832 m/yr median 170; 500.0 isolates a single ~20km segment
    # matching its known confluence width — see domain._filter_gl_by_flow)
    # and accepts optional `corridor_margin_km` (also constrains the
    # grounded region's width near the interface to roughly this many km
    # beyond the flow-filtered grounding line's own extent — confirmed for
    # Byrd that grounded_polygon = buffer(gl, buffer_km) ∩ basin_polygon
    # stays ~100-140km wide near the coast even after min_speed_myr
    # narrows gl, since buffer_km=100 is generous enough that
    # basin_polygon's own shape dominates regardless; omitted = no
    # constraint — see domain._grounded_corridor). The strategy also uses
    # `min_speed_myr` as floating_region_kwargs' min_normal_velocity_myr
    # default, so the two never need to be set separately.
    region_strategy_kwargs: dict = field(default_factory=dict)

    grounding_line_source: str = "measures_boundaries_2008"
    # Artificially resample the grounding line to this arc-length spacing
    # (meters) before it's used anywhere (corridor buffer, cut boundary,
    # ols_d, x_md/y_md) — None (default) = keep the source's native vertex
    # spacing. Only meaningful for an ordered-polyline source
    # (measures_boundaries_2008); do not set this for a mask-based
    # unordered pixel-scatter source (bedmachine_mask) — see
    # utils/geometry_utils.resample_polyline's docstring.
    grounding_line_resample_m: float | None = None
    # bedmachine_mask finds the true floating_ice/ocean transition, so it
    # correctly excludes shelf-boundary arcs adjacent to *any* tributary
    # glacier's grounding line, not just the one grounding_zone selected —
    # see data_sources/calving_front.py's module docstring.
    calving_front_source: str = "bedmachine_mask"

    # whole_shelf (default) = the named ice shelf's full extent minus the
    # grounded corridor, regardless of buffer_km/grounding_zone. Fine for
    # a moderate shelf fed by a few tributaries (Amery); for a shelf as
    # large as Ross East, fed by many outlet glaciers, restrict to just
    # grounding_zone's own flow with flowline_corridor/basin_partition
    # instead — see floating_region.py.
    floating_region_source: str = "whole_shelf"

    # Collocation library (xcol/ycol) sampling. Empty (default) keeps the
    # historical behavior: xcol/ycol = the velocity data points (xd/yd). To
    # decouple the collocation library from data density, set:
    #   collocation:
    #     density: 0.5           # pts/km^2 over the whole region (base tier)
    #     interface_density: 2.0 # pts/km^2 in a band around the interfaces
    #     interface_band_km: 10  # half-width (km) of that band (GL + front)
    # Two-tier low-discrepancy (Halton) sampling, partitioned so each
    # subregion has exactly its stated density — see utils/collocation.py.
    # Only `density` is required when the block is present; interface_density
    # defaults to density (no boost) and interface_band_km defaults to 10.
    collocation: dict = field(default_factory=dict)

    paths: dict = field(default_factory=lambda: dict(DEFAULT_PATHS))

    # Extra kwargs forwarded verbatim to the chosen source-processing
    # functions (e.g. a `custom_xy` source's array/path).
    velocity_kwargs: dict = field(default_factory=dict)
    thickness_kwargs: dict = field(default_factory=dict)
    surface_kwargs: dict = field(default_factory=dict)
    grounding_line_kwargs: dict = field(default_factory=dict)
    calving_front_kwargs: dict = field(default_factory=dict)
    floating_region_kwargs: dict = field(default_factory=dict)

    def path(self, key: str) -> Path:
        return Path(self.paths[key])


def load_config(path: str | Path) -> PipelineConfig:
    """Build a PipelineConfig from a YAML file — see data_build_configs/TEMPLATE.yaml
    for every available field.

    `ice_shelf`, `grounding_zone`, and `buffer_km` are required; everything
    else falls back to PipelineConfig's own defaults if omitted. A
    `paths` block in the YAML is merged into DEFAULT_PATHS (only the keys
    it names are overridden) rather than replacing it wholesale, so a
    config that only wants to point at a different BedMachine version
    doesn't have to repeat every other path.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    valid_fields = {f.name for f in fields(PipelineConfig)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ValueError(
            f"{path}: unknown config field(s) {sorted(unknown)}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    if "paths" in raw:
        merged_paths = dict(DEFAULT_PATHS)
        merged_paths.update(raw["paths"])
        raw["paths"] = merged_paths

    return PipelineConfig(**raw)
