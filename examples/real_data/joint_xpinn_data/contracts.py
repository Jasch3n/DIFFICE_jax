"""Shared data contracts.

Every data-source processing function returns one of these, regardless of
its input file format. Downstream pipeline code (domain construction,
dataset assembly) only ever depends on these shapes, never on a specific
source's file format.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PointObservations:
    """Scattered or gridded point data, already cropped to a region.

    x, y are always in the shared projection (EPSG:3031 polar stereographic,
    meters) regardless of what the source file used natively.
    """

    x: np.ndarray
    y: np.ndarray
    values: dict[str, np.ndarray]
    # Per-point 1-sigma measurement uncertainty, in the same units as the
    # relevant `values` entry — smaller means more confident. None if the
    # source provides no uncertainty estimate (e.g. BEDMAP1).
    weight: np.ndarray | None
    product: str
    epoch: str | None

    def __post_init__(self):
        n = self.x.shape[0]
        if self.y.shape[0] != n:
            raise ValueError(f"x has {n} points but y has {self.y.shape[0]}")
        for key, arr in self.values.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"values[{key!r}] has {arr.shape[0]} points, expected {n}"
                )
        if self.weight is not None and self.weight.shape[0] != n:
            raise ValueError(
                f"weight has {self.weight.shape[0]} points, expected {n}"
            )


def split_on_nan(x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Split a NaN-separated polyline/ring array into segments of shape (Ni, 2)."""
    breaks = np.where(np.isnan(x))[0]
    segments = []
    start = 0
    for b in list(breaks) + [len(x)]:
        if b > start:
            seg = np.column_stack([x[start:b], y[start:b]])
            if len(seg) >= 2:
                segments.append(seg)
        start = b + 1
    return segments


@dataclass
class Geometry:
    """One or more disconnected polylines (grounding line, calving front, ...).

    segments[i] has shape (Ni, 2) in EPSG:3031 meters.

    normals, if provided, is a per-segment (Ni, 2) array of unit outward
    normals computed directly by the source (e.g. a mask-based provider
    that already knows exactly which side is ocean). When absent,
    downstream code falls back to estimating normals from local tangents,
    which is less accurate than a source that can determine "outward"
    directly.

    ordered records whether each segment's points are a genuine, walkable
    polyline (true for e.g. measures_boundaries_2008's shapefile-derived
    line) vs. an unordered scatter (true for mask-pixel-adjacency
    providers, whose points come out in raster scan order, not boundary
    order). Code that needs to walk the boundary (e.g.
    utils.geometry_utils.resample_geometry) must check this rather than
    assume every Geometry is a real line.
    """

    segments: list[np.ndarray]
    product: str
    epoch: str | None = None
    normals: list[np.ndarray] | None = None
    ordered: bool = True

    def as_array(self) -> np.ndarray:
        """NaN-separated (N, 2) array, matching the groundingline_2008_v2.mat convention."""
        if not self.segments:
            return np.zeros((0, 2))
        parts = []
        for i, seg in enumerate(self.segments):
            if i > 0:
                parts.append(np.full((1, 2), np.nan))
            parts.append(seg)
        return np.concatenate(parts, axis=0)

    def all_points(self) -> np.ndarray:
        """All vertices stacked, no NaN separators — for spatial queries."""
        if not self.segments:
            return np.zeros((0, 2))
        return np.concatenate(self.segments, axis=0)


@dataclass
class CheckResult:
    """Output of a consistency check — a quantitative comparison between
    two independently-observed quantities that should agree if the region
    model and source data are both correct (e.g. velocity vs. mapped
    calving front, thickness vs. surface elevation via flotation).

    Every check function in the `checks` registry returns this same
    shape, regardless of what it compares — mirroring how every provider
    in `data_sources` returns `PointObservations`/`Geometry` regardless of
    file format.
    """

    name: str
    region: str | None  # "grounded" / "floating" / None if not region-scoped
    x: np.ndarray
    y: np.ndarray
    metric: np.ndarray  # per-point signed quantitative well-fitness value
    unit: str
    threshold: float | None = None  # tolerance defining a "violation", if set
    passed: bool | None = None  # None if purely descriptive (no threshold judgement)

    def __post_init__(self):
        n = self.x.shape[0]
        if self.y.shape[0] != n:
            raise ValueError(f"x has {n} points but y has {self.y.shape[0]}")
        if self.metric.shape[0] != n:
            raise ValueError(f"metric has {self.metric.shape[0]} points, expected {n}")

    @property
    def n_points(self) -> int:
        return len(self.x)

    def summary(self) -> str:
        """Reports the metric's distribution only — not a recomputed
        violation count, since what counts as a "violation" (one-sided vs.
        symmetric around zero, etc.) is check-specific. `passed`, if set,
        is always computed by the check itself, not derived here."""
        header = self.name + (f" ({self.region})" if self.region else "")
        if self.n_points == 0:
            return f"{header}: no points evaluated"
        p10, p90 = np.percentile(self.metric, [10, 90])
        lines = [
            f"{header}: {self.n_points} points",
            f"  metric [{self.unit}]: mean={self.metric.mean():.3g} median={np.median(self.metric):.3g} "
            f"p10={p10:.3g} p90={p90:.3g} min={self.metric.min():.3g} max={self.metric.max():.3g}",
        ]
        if self.threshold is not None:
            lines.append(f"  threshold={self.threshold:.3g} {self.unit}")
        if self.passed is not None:
            lines.append(f"  CHECK: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)
