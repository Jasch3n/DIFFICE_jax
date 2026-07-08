"""Nearest-neighbor resampling of one source's PointObservations onto another
source's (x, y) locations.

Needed because different sources live on different native grids/point
sets (MEaSURES velocity ~450m grid vs BedMachine ~500m grid vs BEDMAP1
scattered flight lines) — "dense" reference fields like h_dense/s_dense
must be reported at the velocity data locations, not at BedMachine's own
grid nodes.
"""

import numpy as np
from scipy.spatial import cKDTree

from joint_xpinn_data.contracts import PointObservations


def nearest_sample(query_x: np.ndarray, query_y: np.ndarray, source: PointObservations) -> dict[str, np.ndarray]:
    if source.x.shape[0] == 0:
        raise ValueError(
            f"Cannot resample onto {len(query_x)} query point(s): source "
            f"{source.product!r} has no points in this region."
        )
    tree = cKDTree(np.column_stack([source.x, source.y]))
    _, idx = tree.query(np.column_stack([query_x, query_y]))
    return {field: values[idx] for field, values in source.values.items()}
