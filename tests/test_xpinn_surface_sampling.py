import numpy as np
import pytest
import jax.numpy as jnp
from jax import random

from diffice_jax.data.xpinns.preprocessing import normalize_each
from diffice_jax.data.xpinns.sampling import data_sample_create


def _obj(value):
    arr = np.empty((1, 1), dtype=object)
    arr[0, 0] = np.asarray(value, dtype=np.float32).reshape(-1, 1)
    return arr


def _raw_region(include_surface_coords=True, sd=None, xd_s=None, yd_s=None):
    data = {
        "xd": _obj([0.0, 1.0, 0.0, 1.0]),
        "yd": _obj([0.0, 0.0, 1.0, 1.0]),
        "ud": _obj([1.0, 2.0, 3.0, 4.0]),
        "vd": _obj([4.0, 3.0, 2.0, 1.0]),
        "xd_h": _obj([0.0, 1.0, 0.5]),
        "yd_h": _obj([0.0, 1.0, 0.5]),
        "hd": _obj([100.0, 150.0, 200.0]),
        "sd": _obj([10.0, 20.0, 30.0, 40.0] if sd is None else sd),
        "xcol": _obj([0.2, 0.8, 0.2, 0.8]),
        "ycol": _obj([0.2, 0.2, 0.8, 0.8]),
        "xct": _obj([1.0, 1.0]),
        "yct": _obj([0.0, 1.0]),
        "nnct": _obj([1.0, 0.0, 1.0, 0.0]),
        "x_md": _obj([0.5, 0.5]),
        "y_md": _obj([0.0, 1.0]),
    }
    data["nnct"][0, 0] = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    if include_surface_coords:
        data["xd_s"] = _obj([0.0, 1.0, 0.0, 1.0] if xd_s is None else xd_s)
        data["yd_s"] = _obj([0.0, 0.0, 1.0, 1.0] if yd_s is None else yd_s)
    return data


def test_xpinn_preprocessing_rejects_surface_without_surface_coordinates():
    with pytest.raises(ValueError, match="XPINN surface observation grid requires `sd`, `xd_s`, and `yd_s`"):
        normalize_each(_raw_region(include_surface_coords=False), 0, 1)


def test_xpinn_preprocessing_rejects_surface_coordinate_shape_mismatch():
    with pytest.raises(ValueError, match=r"region 0 shapes are sd=\(4, 1\), xd_s=\(3, 1\), yd_s=\(4, 1\)"):
        normalize_each(_raw_region(xd_s=[0.0, 1.0, 0.0]), 0, 1)


def test_xpinn_preprocessing_filters_surface_triplet_finite_mask():
    data = _raw_region(
        sd=[10.0, 20.0, 30.0, 40.0],
        xd_s=[0.0, np.nan, 2.0, 3.0],
        yd_s=[0.0, 1.0, 2.0, np.nan],
    )

    normalized = normalize_each(data, 0, 1)

    assert normalized[0][3].shape == (2, 2)
    assert normalized[1][2].shape == (2, 1)
    assert normalized[4][4][2].tolist() == [0, 2]


def _region(offset):
    X_v = jnp.arange(offset, offset + 40, dtype=jnp.float32).reshape(20, 2)
    X_h = jnp.arange(offset + 100, offset + 116, dtype=jnp.float32).reshape(8, 2)
    X_col = jnp.arange(offset + 200, offset + 240, dtype=jnp.float32).reshape(20, 2)
    X_s = jnp.arange(offset + 300, offset + 340, dtype=jnp.float32).reshape(20, 2)
    U_v = jnp.arange(offset + 400, offset + 440, dtype=jnp.float32).reshape(20, 2)
    H = jnp.arange(offset + 500, offset + 508, dtype=jnp.float32).reshape(8, 1)
    S = jnp.arange(offset + 600, offset + 620, dtype=jnp.float32).reshape(20, 1)
    X_ct = jnp.arange(offset + 700, offset + 708, dtype=jnp.float32).reshape(4, 2)
    nn_ct = jnp.ones((4, 2), dtype=jnp.float32)
    X_md = jnp.arange(offset + 800, offset + 812, dtype=jnp.float32).reshape(6, 2)
    return [X_v, X_h, X_col, X_s], [U_v, H, S], X_ct, nn_ct, None, X_md


def test_xpinn_sampler_draws_surface_from_independent_locations():
    data_all = [_region(0), _region(1000)]
    dataf = data_sample_create(
        data_all,
        [0, 1],
        [[6, 6], [3, 3], [5, 5], [4, 4], [2, 2], 2],
        basal_mask=[False, False],
    )

    batch = dataf(random.PRNGKey(17))
    sample = batch["smp"]

    assert [item.shape for item in sample.H_smp] == [(3, 1), (3, 1)]
    assert [item.shape for item in sample.S_smp] == [(5, 1), (5, 1)]
    assert [item.shape for item in sample.Xs_smp] == [(5, 2), (5, 2)]
    assert all(jnp.all(item[:, 0] >= 300) for item in sample.Xs_smp)
