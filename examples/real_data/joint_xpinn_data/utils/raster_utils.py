"""Shared raster helpers for mask-based GL/calving-front providers.

Both BedMachine (netCDF) and the Antarctic-Boundaries mask (GeoTIFF)
encode categories on a regular grid with the same row/y-descending,
col/x-ascending convention, so the pixel-adjacency logic that finds a
transition between two categories (grounded/floating, or shelf/ocean) is
identical regardless of which file format or category values a given
source uses.
"""

import numpy as np
import rasterio
import shapely
from rasterio.windows import Window, from_bounds
from scipy import ndimage


def read_geotiff_crop(path: str, bx0: float, by0: float, bx1: float, by1: float, pad_m: float = 0.0):
    """Read a GeoTIFF cropped to [bx0-pad, bx1+pad] x [by0-pad, by1+pad].

    Returns (x_sub, y_sub, arr) with x_sub ascending, y_sub descending —
    the same convention used for the netCDF sources (velocity.py,
    thickness.py) — so mask_boundary_points works unmodified on either.
    """
    with rasterio.open(path) as ds:
        win = from_bounds(bx0 - pad_m, by0 - pad_m, bx1 + pad_m, by1 + pad_m, transform=ds.transform)
        col_off = max(int(np.floor(win.col_off)), 0)
        row_off = max(int(np.floor(win.row_off)), 0)
        col_end = min(int(np.ceil(win.col_off + win.width)), ds.width)
        row_end = min(int(np.ceil(win.row_off + win.height)), ds.height)
        window = Window(col_off, row_off, col_end - col_off, row_end - row_off)

        arr = ds.read(1, window=window)
        transform = ds.window_transform(window)

    ncols, nrows = arr.shape[1], arr.shape[0]
    x_sub = transform.c + (np.arange(ncols) + 0.5) * transform.a
    y_sub = transform.f + (np.arange(nrows) + 0.5) * transform.e
    return x_sub, y_sub, arr


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected True component of a boolean mask.

    Mask-based GL/front providers crop to the requested polygon's *bounding
    box* (padded), not its exact shape, since the shape isn't known to be a
    single simple raster region ahead of time. That box can also contain
    other, unrelated ice with the same mask category — a neighboring ice
    shelf or glacier tongue, drifting sea ice classified as floating_ice,
    an isolated no-data-locked valid-velocity patch — which then produces
    spurious boundary points far from the actual feature of interest. The
    named feature is always overwhelmingly the largest connected blob of
    its category within a bounding-box crop (a real ice shelf/coverage
    area is orders of magnitude bigger than any stray fragment sharing the
    box), so dropping every component except the largest removes those
    fragments without per-shelf tuning.
    """
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def rasterize_polygon(polygon, pixel_size_m: float, pad_km: float = 5.0):
    """Rasterize a shapely (Multi)Polygon to a boolean mask on a regular
    grid at `pixel_size_m` resolution, padded `pad_km` beyond its bounds.
    `x`/`y` are both ascending (unlike `read_geotiff_crop`'s
    y-descending raster convention — there's no source file's own row
    order to match here)."""
    bx0, by0, bx1, by1 = polygon.bounds
    pad_m = pad_km * 1000.0
    x = np.arange(bx0 - pad_m, bx1 + pad_m, pixel_size_m)
    y = np.arange(by0 - pad_m, by1 + pad_m, pixel_size_m)
    xx, yy = np.meshgrid(x, y)
    mask = shapely.contains(polygon, shapely.points(xx.ravel(), yy.ravel())).reshape(xx.shape)
    return x, y, mask


def polygon_from_level_set(x: np.ndarray, y: np.ndarray, field: np.ndarray, level: float):
    """Trace `field == level` via marching squares (`skimage.measure.
    find_contours`) and return the largest closed loop as a shapely
    Polygon, in the same (ascending x, ascending y) coordinates as `x`/`y`.

    Deliberately not `rasterize_polygon`'s inverse (a pixel mask's own
    outline, traced pixel-edge to pixel-edge) — that gives a blocky,
    stair-stepped boundary at `pixel_size_m` resolution regardless of how
    smooth the underlying field is. Marching squares linearly interpolates
    the crossing point *within* each grid cell, so a smoothly-varying
    field (e.g. a Gaussian-smoothed velocity field) produces a genuinely
    smooth-looking contour instead — see domain._grounded_corridor, which
    needs exactly this to avoid a raw pixel mask's own jaggedness leaking
    into the grounded region's boundary.
    """
    from skimage import measure

    contours = measure.find_contours(field, level)
    if not contours:
        return shapely.Polygon()
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    polygons = []
    for c in contours:
        if len(c) < 4:
            continue
        xs = x[0] + c[:, 1] * dx
        ys = y[0] + c[:, 0] * dy
        poly = shapely.Polygon(np.column_stack([xs, ys]))
        if poly.is_valid and poly.area > 0:
            polygons.append(poly)
    if not polygons:
        return shapely.Polygon()
    return max(polygons, key=lambda p: p.area)


def boundary_sharpness(polygon, pixel_size_m: float = 500.0, sigma_km: float = 1.0, pad_km: float = 5.0):
    """Locate sharp width transitions along `polygon`'s boundary — e.g.
    an unphysical step where a flow corridor's buffer radius jumps
    abruptly — using a standard convolutional corner/curvature detector:
    a Laplacian-of-Gaussian (LoG) filter over the *distance-to-boundary*
    field, not the raw mask.

    Running LoG directly on the raw boolean mask doesn't discriminate
    "sharp" from "smooth" boundary: a binary mask transitions the same
    way at every boundary pixel, so a plain edge detector just re-traces
    the whole outline uniformly, everywhere. The distance transform
    (`ndimage.distance_transform_edt`) turns the mask into a
    smoothly-varying "local half-width" field instead — LoG on *that*
    responds near zero where width changes smoothly downstream and
    strongly wherever it changes abruptly (a corner/kink in the width
    profile), which is exactly the "sharp edge" failure mode a taper
    that jumps too quickly (or is seeded from a wider interface than the
    reported grounding line, see floating_region.py) produces.

    Returns `(x, y, response, mask, distance)` — `response` is signed
    (LoG flips sign across a ridge/valley in the width field, e.g. going
    from narrowing to widening), so `abs(response)` is the sharpness
    measure to threshold/rank by; see `sharpest_points`, which also needs
    `distance` (see its docstring for why).
    """
    x, y, mask = rasterize_polygon(polygon, pixel_size_m, pad_km)
    distance = ndimage.distance_transform_edt(mask, sampling=pixel_size_m)
    sigma_px = (sigma_km * 1000.0) / pixel_size_m
    response = ndimage.gaussian_laplace(distance, sigma=sigma_px)
    return x, y, response, mask, distance


def sharpest_points(
    x, y, response, mask, distance, near_boundary_m: float, top_n: int = 20, min_separation_px: int = 5,
):
    """Top `top_n` locations within `near_boundary_m` of the true
    boundary (`mask & (distance < near_boundary_m)`) by `abs(response)`
    (see `boundary_sharpness`), each at least `min_separation_px` (grid
    index) from every other — otherwise the list is dominated by one
    real corner's whole neighborhood rather than distinct features.

    The `near_boundary_m` restriction is required, not optional:
    `gaussian_laplace` of a distance transform also responds strongly
    along a shape's *medial axis* (skeleton) — any elongated shape has
    one, corner or not, since that's where the "nearest boundary point"
    switches sides and the distance field has a real kink. Confirmed via
    a synthetic test (a corridor with a genuine step-width transition vs.
    one that widens smoothly, same overall shape otherwise): without this
    restriction both showed similar peak response (medial-axis response
    dominating in both); restricting to near-boundary pixels cut the
    smooth corridor's peak by ~45% while leaving the step corridor's peak
    — correctly located exactly at the step — essentially unchanged.
    Pick `near_boundary_m` a few pixels wide (e.g. 2-3x `pixel_size_m`).

    Returns an (n, 3) array of (x, y, abs(response)), sorted descending
    by sharpness.
    """
    candidates = mask & (distance < near_boundary_m)
    abs_response = np.where(candidates, np.abs(response), 0.0)
    order = np.argsort(abs_response.ravel())[::-1]
    rows, cols = np.unravel_index(order, abs_response.shape)

    picked_rows, picked_cols = [], []
    for r, c in zip(rows, cols):
        if abs_response[r, c] <= 0:
            break
        too_close = any(
            (r - pr) ** 2 + (c - pc) ** 2 < min_separation_px ** 2
            for pr, pc in zip(picked_rows, picked_cols)
        )
        if not too_close:
            picked_rows.append(r)
            picked_cols.append(c)
        if len(picked_rows) >= top_n:
            break
    return np.column_stack([x[picked_cols], y[picked_rows], abs_response[picked_rows, picked_cols]])


def boundary_indices(region_a: np.ndarray, region_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of every `region_a` pixel 4-connected-adjacent to a
    `region_b` pixel — the row/col detection `mask_boundary_points` layers
    point-position and outward-normal extraction on top of. Exposed
    separately for callers that only need *which* pixels are on the
    boundary (e.g. to look up a real data value there directly), not an
    outward normal — see build_dataset._dirichlet_from_mask.

    Same row/y-descending, col/x-ascending convention as
    `mask_boundary_points` (see its docstring).
    """
    padded_b = np.pad(region_b, 1, mode="constant", constant_values=False)
    b_up = padded_b[0:-2, 1:-1]
    b_down = padded_b[2:, 1:-1]
    b_left = padded_b[1:-1, 0:-2]
    b_right = padded_b[1:-1, 2:]
    boundary_mask = region_a & (b_up | b_down | b_left | b_right)
    return np.nonzero(boundary_mask)


def mask_boundary_points(x_sub: np.ndarray, y_sub: np.ndarray, region_a: np.ndarray, region_b: np.ndarray):
    """Pixels of `region_a` (boolean grid) adjacent to `region_b`, with unit
    normals pointing from region_a towards region_b.

    Assumes row index increases as y decreases and col index increases as
    x increases (true for every raster source in this package). Pixels
    where opposing neighbors cancel (e.g. a 1-pixel-wide bridge with
    region_b on both north and south) are dropped rather than assigned a
    degenerate zero-length normal.
    """
    padded_b = np.pad(region_b, 1, mode="constant", constant_values=False)
    b_up = padded_b[0:-2, 1:-1]     # row-1 neighbor -> +y (north)
    b_down = padded_b[2:, 1:-1]     # row+1 neighbor -> -y (south)
    b_left = padded_b[1:-1, 0:-2]   # col-1 neighbor -> -x (west)
    b_right = padded_b[1:-1, 2:]    # col+1 neighbor -> +x (east)

    rows, cols = boundary_indices(region_a, region_b)
    if len(rows) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    points = np.column_stack([x_sub[cols], y_sub[rows]])
    normal_x = b_right.astype(float) - b_left.astype(float)
    normal_y = b_up.astype(float) - b_down.astype(float)
    normals = np.column_stack([normal_x[rows, cols], normal_y[rows, cols]])
    norm = np.linalg.norm(normals, axis=1)
    keep = norm > 0
    return points[keep], normals[keep] / norm[keep, None]
