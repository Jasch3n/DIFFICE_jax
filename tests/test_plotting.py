import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

_PLOTTING_PATH = Path(__file__).resolve().parents[1] / "diffice_jax" / "visualization" / "plotting.py"
_SPEC = importlib.util.spec_from_file_location("diffice_jax_plotting", _PLOTTING_PATH)
_PLOTTING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLOTTING)
tripcolor_scattered = _PLOTTING.tripcolor_scattered
_finite_unique_triplets = _PLOTTING._finite_unique_triplets


def test_tripcolor_scattered_filters_invalid_and_duplicate_points():
    fig, ax = plt.subplots()
    image = tripcolor_scattered(
        [0.0, 1.0, 0.0, 0.0, float("nan")],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 3.0, 99.0, 4.0],
        ax=ax,
        coordinate_scale=1e-3,
        cmap="viridis",
    )

    assert image.axes is ax
    x, y, values = _finite_unique_triplets(
        [0.0, 1.0, 0.0, 0.0, float("nan")],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 3.0, 99.0, 4.0],
    )
    assert x.tolist() == [0.0, 1.0, 0.0]
    assert y.tolist() == [0.0, 0.0, 1.0]
    assert values.tolist() == [1.0, 2.0, 3.0]
    plt.close(fig)


def test_tripcolor_scattered_rejects_too_few_unique_points():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="at least 3 finite unique points"):
        tripcolor_scattered([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], ax=ax)
    plt.close(fig)
