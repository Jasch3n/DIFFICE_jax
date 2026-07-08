"""Shared color palette helper for multi-category diagnostic plots."""

import numpy as np


def distinct_colors(n: int, seed: int = 0):
    """n hues evenly spaced around the color wheel, then shuffled so
    alphabetically/sequentially-adjacent categories (which are often
    geographically or logically adjacent too) don't end up with
    near-identical colors. Always fully saturated (pure HSV), so a
    category color never collides with a muted "gray = unresolved/other"
    marker used alongside it.
    """
    import matplotlib.pyplot as plt

    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.default_rng(seed).shuffle(hues)
    return plt.cm.hsv(hues)
