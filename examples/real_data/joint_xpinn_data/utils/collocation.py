"""Collocation-point sampling for the PINN residual library.

Decouples the collocation library (``xcol``/``ycol``) from the velocity data
grid (``xd``/``yd``). This matters because, for XPINN, the ``.mat``'s
``xcol``/``ycol`` *are* each region's collocation library: DIFFICE_jax loads
them verbatim and samples the physics-residual points from that pool at train
time (no re-expansion, unlike its PINN path). With ``xcol = xd`` the library is
just the data points; this module lets a config set the library's size and
spatial distribution independently.

Method — two-tier, low-discrepancy (Halton), configured by ``PipelineConfig``'s
``collocation`` block::

    collocation:
      density: 0.5            # pts/km^2, base — over the whole region
      interface_density: 2.0  # pts/km^2, inside the near-interface band
      interface_band_km: 10   # half-width (km) of that band

``density``/``interface_density`` each accept either a scalar (same value for
both regions, e.g. the Amery builds) or a ``[grounded, floating]`` pair, for
shelves whose two regions have noticeably different velocity-point densities
(e.g. Byrd's grounded corridor is flow-filtered to a much sparser trunk than
its floating region). The two tiers are *partitioned* (disjoint): the band
around the region's interfaces holds exactly ``interface_density``, and the
core (region minus band) holds ``density`` — so each subregion's point density
is precisely what the config states. "Interfaces" are the grounding line for
the grounded region and the grounding line + calving front for the floating
region. Points are low-discrepancy (Halton, taken in sequence and kept where
inside the polygon), so builds are deterministic and reproducible without a
seed.

When the ``collocation`` block is absent (or ``density`` is None), this module
is not invoked and the caller keeps ``xcol/ycol = xd/yd`` — existing outputs
are unchanged (this is a purely additive, opt-in feature).

Note: sampling covers the whole region polygon, including any near-stagnant
area that ``grounded_min_speed_myr`` drops from the *data* (xd/yd). The PDE
residual is valid there regardless; if a future use wants collocation confined
to the fast-flow / data-supported area, that would be a further refinement.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.stats import qmc
from shapely.geometry import MultiPoint, Polygon

from joint_xpinn_data.domain import DomainRegions

_VALID_KEYS = {"density", "interface_density", "interface_band_km"}


@dataclass(frozen=True)
class CollocationSettings:
    density: tuple[float, float]  # pts/km^2 over the whole region (core tier), (grounded, floating)
    interface_density: tuple[float, float]  # pts/km^2 inside the near-interface band, (grounded, floating)
    interface_band_km: float  # half-width (km) of that band


def _as_region_pair(value) -> tuple[float, float]:
    """A scalar broadcasts to both regions; a 2-element sequence is (grounded, floating)."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"collocation density must be a scalar or a [grounded, floating] pair, got {value!r}."
            )
        return float(value[0]), float(value[1])
    return float(value), float(value)


def collocation_settings(config) -> CollocationSettings | None:
    """Parse ``PipelineConfig.collocation`` into settings, or None to keep xcol=xd."""
    spec = getattr(config, "collocation", None) or {}
    unknown = set(spec) - _VALID_KEYS
    if unknown:
        raise ValueError(
            f"Unknown collocation config field(s): {sorted(unknown)}. "
            f"Valid fields: {sorted(_VALID_KEYS)}."
        )
    density = spec.get("density")
    if density is None:
        return None
    return CollocationSettings(
        density=_as_region_pair(density),
        interface_density=_as_region_pair(spec.get("interface_density", density)),
        interface_band_km=float(spec.get("interface_band_km", 10.0)),
    )


def _interface_points(regions: DomainRegions, is_grounded: bool) -> np.ndarray:
    """Interface vertices for a region: GL (grounded) or GL + front (floating)."""
    parts = [regions.grounding_line.all_points()]
    if not is_grounded and regions.calving_front is not None:
        parts.append(regions.calving_front.all_points())
    parts = [p for p in parts if p.size]
    return np.vstack(parts) if parts else np.zeros((0, 2))


def _halton_in_polygon(polygon, n_target: int) -> np.ndarray:
    """``n_target`` deterministic Halton points inside ``polygon`` (rejection).

    Points are drawn in the polygon's bounding box from a single unscrambled
    Halton sequence and kept where inside; the sequence keeps advancing across
    batches, so it stays low-discrepancy and reproducible.
    """
    if n_target <= 0 or polygon.is_empty:
        return np.zeros((0, 2))
    minx, miny, maxx, maxy = polygon.bounds
    span_x, span_y = maxx - minx, maxy - miny
    if span_x <= 0 or span_y <= 0:
        return np.zeros((0, 2))
    engine = qmc.Halton(d=2, scramble=False)
    kept_x, kept_y = [], []
    got = 0
    # Oversize each batch by the inverse fill fraction so we usually finish in
    # one or two draws; guard against pathological thin polygons with a cap.
    fill = max(polygon.area / (span_x * span_y), 1e-3)
    for _ in range(1000):
        batch = int(max(1024, (n_target - got) / fill * 1.3))
        u = engine.random(batch)
        px = minx + u[:, 0] * span_x
        py = miny + u[:, 1] * span_y
        inside = shapely.contains_xy(polygon, px, py)
        kept_x.append(px[inside])
        kept_y.append(py[inside])
        got += int(inside.sum())
        if got >= n_target:
            break
    x = np.concatenate(kept_x) if kept_x else np.zeros(0)
    y = np.concatenate(kept_y) if kept_y else np.zeros(0)
    return np.column_stack([x[:n_target], y[:n_target]])


def sample_collocation(regions: DomainRegions, polygon, is_grounded: bool,
                       settings: CollocationSettings) -> tuple[np.ndarray, np.ndarray]:
    """Two-tier Halton collocation points inside ``polygon``.

    Returns ``(xcol, ycol)`` as 1-D arrays. The band around the region's
    interfaces is sampled at ``interface_density``; the rest at ``density``.
    """
    idx = 0 if is_grounded else 1
    density = settings.density[idx]
    interface_density = settings.interface_density[idx]

    interface_pts = _interface_points(regions, is_grounded)
    band_m = settings.interface_band_km * 1e3
    if interface_pts.size and band_m > 0.0:
        band_zone = MultiPoint(interface_pts).buffer(band_m)
        band = polygon.intersection(band_zone)
        core = polygon.difference(band_zone)
    else:
        band = Polygon()  # no interfaces / zero band width -> everything is core
        core = polygon

    core_n = int(round(core.area / 1e6 * density))
    band_n = int(round(band.area / 1e6 * interface_density))
    core_xy = _halton_in_polygon(core, core_n)
    band_xy = _halton_in_polygon(band, band_n)
    xy = np.vstack([core_xy, band_xy]) if (core_xy.size or band_xy.size) else np.zeros((0, 2))
    return xy[:, 0], xy[:, 1]
