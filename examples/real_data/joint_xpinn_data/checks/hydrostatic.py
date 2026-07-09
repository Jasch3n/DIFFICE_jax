"""Hydrostatic equilibrium (flotation) consistency check between ice
thickness and surface elevation.

Only meaningful for the floating region — grounded ice is supported by
bedrock, not buoyancy, so a large delta there is expected, not a data
error (see CONTEXT.md's "Hydrostatic equilibrium (flotation) delta").
"""

import numpy as np

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import CheckResult
from joint_xpinn_data.data_sources import surface, thickness
from joint_xpinn_data.domain import build_regions
from joint_xpinn_data.resample import nearest_sample

RHO_ICE = 917.0
RHO_SEAWATER = 1023.0


def check_hydrostatic_equilibrium(
    config: PipelineConfig,
    regions=None,
    rho_ice: float = RHO_ICE,
    rho_seawater: float = RHO_SEAWATER,
    threshold: float | None = None,
) -> CheckResult:
    """metric = delta = rho_ice*thickness + rho_seawater*(surface-thickness),
    in kg/m^2 — zero means the ice column is exactly floating. Evaluated at
    the thickness source's own points (xd_h/yd_h); surface elevation (its
    own independent data kind, see data_sources/surface.py) is
    nearest-neighbor resampled onto those points purely for this
    computation, with no effect on the persisted .mat schema — see
    docs/adr/0001-surface-elevation-independent-coordinates.md.

    Purely descriptive by default (threshold=None, passed=None): delta is
    a continuous physical residual that's never exactly zero even for
    genuinely floating ice (measurement noise, and the two-density formula
    here ignores firn's lower density near the surface), so a strict
    "any nonzero delta fails" rule would be uninformative. Pass an
    explicit threshold once you've decided what an acceptable residual
    looks like empirically for this data.
    """
    regions = regions or build_regions(config)
    floating_polygon = regions.floating_polygon

    thick = thickness.load_thickness(config, floating_polygon)
    surf = surface.load_surface(config, floating_polygon)

    surf_at_thick = nearest_sample(thick.x, thick.y, surf)["surface"]
    h = thick.values["thickness"]
    delta = rho_ice * h + rho_seawater * (surf_at_thick - h)

    passed = None if threshold is None else bool((np.abs(delta) <= threshold).all())

    return CheckResult(
        name="hydrostatic_equilibrium",
        region="floating",
        x=thick.x,
        y=thick.y,
        metric=delta,
        unit="kg/m^2",
        threshold=threshold,
        passed=passed,
    )
