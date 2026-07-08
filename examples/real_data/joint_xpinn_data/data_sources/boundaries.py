"""Extent lookup: named basin / ice-shelf polygons.

Source: Antarctic-boundaries/data/{basins_refined_v2,iceshelves_2008_v2}.mat
(MEaSURES Antarctic Boundaries for IPY 2007-2009, reformatted to .mat).
Both files are MATLAB v7.3 (HDF5), storing per-feature `name`/`x`/`y` as
cell arrays (HDF5 object references).

This module answers "roughly where is <name>" for cropping other sources.
It is deliberately not part of the pluggable GL/calving-front registries in
grounding_line.py / calving_front.py — extent selection is stable
infrastructure, not something we expect to swap between runs.
"""

from functools import lru_cache

import h5py
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from joint_xpinn_data.contracts import split_on_nan


def _decode_str(h5file: h5py.File, ref) -> str:
    arr = np.array(h5file[ref][()]).flatten()
    return "".join(chr(c) for c in arr)


def _decode_xy(h5file: h5py.File, ref) -> np.ndarray:
    return np.array(h5file[ref][()]).flatten()


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1] + x[-1] * y[0] - x[0] * y[-1])


def _rings_to_polygon(rings: list[np.ndarray]):
    """Assemble closed rings into a (Multi)Polygon, honoring interior holes.

    These shapefile-derived rings encode ice rises / rock outcrops /
    islands as interior holes via winding order (ESRI shapefile
    convention: exterior rings wind clockwise, holes counter-clockwise —
    confirmed empirically: Amery's 8 rings split into exactly one large
    negative-signed-area exterior and 7 small positive-signed-area holes
    matching known embedded features). Treating every ring as solid fill
    (the previous behavior) silently inflated shelf/basin polygons to
    include grounded islands, which corrupted anything downstream that
    depends on "is this point really floating ice" — notably the
    calving-front mask lookup in calving_front.py.
    """
    exteriors = [r for r in rings if len(r) >= 4 and _signed_area(r) < 0]
    holes = [r for r in rings if len(r) >= 4 and _signed_area(r) >= 0]
    if not exteriors:
        raise ValueError("No exterior (clockwise) rings found to build a polygon from")

    ext_polys = [Polygon(r) for r in exteriors]
    ext_holes = [[] for _ in ext_polys]
    for hole in holes:
        hole_point = Point(hole[0])
        for i, ep in enumerate(ext_polys):
            if ep.contains(hole_point):
                ext_holes[i].append(hole)
                break
        # a hole not contained by any exterior (shouldn't happen for valid
        # data) is silently dropped rather than mis-assigned

    polys = []
    for ext, hs in zip(exteriors, ext_holes):
        p = Polygon(ext, hs)
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty and p.area > 0:
            polys.append(p)
    if not polys:
        raise ValueError("No valid rings to build a polygon from")
    return unary_union(polys)


@lru_cache(maxsize=None)
def _load_named_geometry(mat_path: str) -> dict:
    """Return {name: (x, y)} with x, y the raw NaN-separated arrays."""
    out = {}
    with h5py.File(mat_path, "r") as h:
        name_refs = h["name"][()].flatten()
        x_refs = h["x"][()].flatten()
        y_refs = h["y"][()].flatten()
        for name_ref, x_ref, y_ref in zip(name_refs, x_refs, y_refs):
            name = _decode_str(h, name_ref)
            out[name] = (_decode_xy(h, x_ref), _decode_xy(h, y_ref))
    return out


def get_named_polygon(mat_path: str, name: str):
    """Look up a named basin/ice-shelf polygon by exact name.

    Raises KeyError with the list of available names if `name` isn't found
    — the boundaries files use specific capitalization/spelling
    (e.g. "Amery", "Lambert") so a clear error beats a silent empty result.
    """
    geometry = _load_named_geometry(str(mat_path))
    if name not in geometry:
        raise KeyError(
            f"{name!r} not found in {mat_path}. Available names: "
            f"{sorted(geometry)}"
        )
    x, y = geometry[name]
    rings = split_on_nan(x, y)
    return _rings_to_polygon(rings)


def list_names(mat_path: str) -> list[str]:
    return sorted(_load_named_geometry(str(mat_path)))


@lru_cache(maxsize=None)
def get_label_points(mat_path: str) -> dict[str, tuple[float, float]]:
    """Return {name: (x_center, y_center)} — a representative interior
    point for each feature, as chosen by the original data curator (more
    reliable for label placement than a geometric centroid, which can
    fall outside a concave or multi-part polygon)."""
    with h5py.File(str(mat_path), "r") as h:
        name_refs = h["name"][()].flatten()
        names = [_decode_str(h, r) for r in name_refs]
        x_center = np.array(h["x_center"][()]).flatten()
        y_center = np.array(h["y_center"][()]).flatten()
    return {n: (xc, yc) for n, xc, yc in zip(names, x_center, y_center)}
