"""Velocity vs. calving-front conformance check.

Velocity and calving-front data are essentially never from the same
epoch, and ice fronts move — so this doesn't assume perfect agreement. It
quantifies the misfit precisely so you can judge whether it's negligible
noise or a real front/velocity epoch mismatch worth accounting for before
training on this domain.
"""

import numpy as np
import shapely
from scipy.spatial import cKDTree

from joint_xpinn_data.config import PipelineConfig
from joint_xpinn_data.contracts import CheckResult
from joint_xpinn_data.data_sources import velocity
from joint_xpinn_data.domain import build_regions


def check_velocity_vs_front(config: PipelineConfig, search_radius_km: float = 5.0, regions=None) -> CheckResult:
    """metric = signed distance to the calving front, at velocity points
    within search_radius_km (positive = seaward = misfit, negative =
    landward = expected). threshold=0.0: any velocity measurement at all
    on the open-ocean side of the mapped front fails the check — no
    tolerance for stray points, since each one signals either a stale
    front or a front/velocity epoch mismatch that could feed the PINN a
    bogus "ice exists here" data point past the true edge.

    `search_radius_km` is *not* that tolerance — it only controls which
    velocity points get pulled in for evaluation at all (a disk around the
    mapped front). Widening it doesn't make the check more lenient; past
    the distance the front geometry was actually tracked to, it can start
    matching points to the nearest mapped front vertex by raw Euclidean
    distance even when that vertex's normal no longer describes the
    unrelated coastline the point actually sits on — which can turn a
    PASS into a FAIL. Keep it small (comparable to how far the mapped
    front's own endpoints extend past real data) rather than treating it
    as "how much slack to give."
    """
    regions = regions or build_regions(config)
    front_points = regions.calving_front.all_points()
    front_normals = np.concatenate(regions.calving_front_normals, axis=0)
    if len(front_points) == 0:
        raise ValueError("No calving-front points to check against.")

    radius_m = search_radius_km * 1000.0
    # Query a disk around the front on BOTH sides — velocity.load_velocity
    # clips to whatever polygon it's given, so this must extend past the
    # mapped ice extent, or seaward measurements would be invisible by
    # construction.
    front_zone = shapely.multipoints(shapely.points(*front_points.T)).buffer(radius_m)
    vel = velocity.load_velocity(config, front_zone)

    tree = cKDTree(front_points)
    query_xy = np.column_stack([vel.x, vel.y])
    dist, idx = tree.query(query_xy)
    near = dist <= radius_m

    nearest_normal = front_normals[idx[near]]
    offset = query_xy[near] - front_points[idx[near]]
    signed = np.einsum("ij,ij->i", offset, nearest_normal)

    return CheckResult(
        name="velocity_vs_front",
        region=None,
        x=vel.x[near],
        y=vel.y[near],
        metric=signed,
        unit="m",
        threshold=0.0,
        passed=bool((signed > 0.0).sum() == 0),
    )
