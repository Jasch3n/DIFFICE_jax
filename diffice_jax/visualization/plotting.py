"""Small plotting helpers for notebooks and examples."""

from __future__ import annotations

import warnings

import numpy as np


def _finite_unique_triplets(x, y, values):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if not (x.size == y.size == values.size):
        raise ValueError(
            "x, y, and values must contain the same number of entries; "
            f"got {x.size}, {y.size}, and {values.size}."
        )

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[mask]
    y = y[mask]
    values = values[mask]
    if x.size == 0:
        return x, y, values

    xy = np.column_stack((x, y))
    _, keep = np.unique(xy, axis=0, return_index=True)
    keep = np.sort(keep)
    return x[keep], y[keep], values[keep]


def tripcolor_scattered(
    x,
    y,
    values,
    ax=None,
    *,
    coordinate_scale=1.0,
    shading="flat",
    cmap=None,
    vmin=None,
    vmax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    add_colorbar=False,
    colorbar_label=None,
    colorbar_kwargs=None,
    mask_long_triangles=False,
    triangle_edge_scale=3.0,
    max_triangle_edge=None,
    min_points=3,
    **tripcolor_kwargs,
):
    """Plot scattered ``(x, y, values)`` triplets with ``Axes.tripcolor``.

    The inputs are flattened, non-finite entries are removed, and duplicate
    coordinate pairs are dropped before triangulation. ``coordinate_scale`` is
    multiplied into both coordinates, so pass ``1e-3`` to display meters as km.

    Returns the Matplotlib collection created by ``ax.tripcolor``.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    x_plot, y_plot, values_plot = _finite_unique_triplets(x, y, values)
    if x_plot.size < min_points:
        raise ValueError(
            f"tripcolor_scattered needs at least {min_points} finite unique "
            f"points; got {x_plot.size}."
        )

    x_plot = x_plot * coordinate_scale
    y_plot = y_plot * coordinate_scale
    plot_args = (x_plot, y_plot, values_plot)
    if mask_long_triangles:
        import matplotlib.tri as mtri

        tri = mtri.Triangulation(x_plot, y_plot)
        xy = np.column_stack((x_plot, y_plot))
        tri_xy = xy[tri.triangles]
        edge_lengths = np.stack(
            (
                np.linalg.norm(tri_xy[:, 0] - tri_xy[:, 1], axis=1),
                np.linalg.norm(tri_xy[:, 1] - tri_xy[:, 2], axis=1),
                np.linalg.norm(tri_xy[:, 2] - tri_xy[:, 0], axis=1),
            ),
            axis=1,
        )
        edge_limit = max_triangle_edge
        if edge_limit is None:
            edge_limit = triangle_edge_scale * np.nanmedian(edge_lengths)
        mask_tri = np.nanmax(edge_lengths, axis=1) > edge_limit
        if np.any(~mask_tri):
            tri.set_mask(mask_tri)
            plot_args = (tri, values_plot)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in cast",
            category=RuntimeWarning,
        )
        image = ax.tripcolor(
            *plot_args,
            shading=shading,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **tripcolor_kwargs,
        )

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if add_colorbar:
        kwargs = dict(colorbar_kwargs or {})
        cbar = ax.figure.colorbar(image, ax=ax, **kwargs)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    return image
