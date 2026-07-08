"""Shared plotting helper for diagnostics scatter maps."""

from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_colorbar(fig, ax, mappable, label=None, size="4%", pad=0.15):
    """Attach a colorbar sized to `ax`'s actual plotted box, not its
    reserved subplot slot.

    Every diagnostic scatter map here calls `ax.set_aspect("equal")` on
    real-world x/y data, and every shelf/basin shape in this package is
    long and narrow relative to a square subplot slot — so matplotlib
    shrinks the axes' visible box to keep the aspect ratio correct,
    leaving blank margin inside the slot. `fig.colorbar(mappable, ax=ax)`
    sizes the colorbar to the *reserved slot* (via `ax.get_position(original=True)`),
    not the shrunk box, so the colorbar ends up far taller than the plot
    next to it. A locatable-axes divider attaches the colorbar directly
    to the actual box instead, so the two always match.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax, label=label)
