import numpy as np
import jax.numpy as jnp
from jax import random

from diffice_jax.data.xpinns.preprocessing import (
    normalize_data,
    SubScaleResult,
    DataMean,
    DataRange,
    DynamicScale,
)
from diffice_jax.data.xpinns.sampling import data_regression_sample_create
from diffice_jax.model.xpinns.loss import loss_regression_create


def _obj_pair(left, right):
    arr = np.empty((1, 2), dtype=object)
    arr[0, 0] = np.asarray(left)
    arr[0, 1] = np.asarray(right)
    return arr


def _raw_two_region_data():
    x0 = np.array([[0.0, 1.0], [0.0, 1.0]])
    y0 = np.array([[0.0, 0.0], [1.0, 1.0]])
    u0 = np.array([[1.0, 2.0], [1.5, 2.5]])
    v0 = np.array([[0.5, 0.6], [0.7, 0.8]])
    h0 = np.array([[100.0, 110.0], [120.0, 130.0]])
    s0 = np.array([[10.0, 11.0], [12.0, 13.0]])
    mu0 = np.array([[5.0, 7.0], [5.0, 7.0]]) * 1e13
    c0 = np.array([[1.0, 2.0], [3.0, 4.0]])

    x1 = np.array([[2.0, 3.0], [2.0, 3.0]])
    y1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    u1 = np.array([[0.4, 0.5], [0.6, 0.7]])
    v1 = np.array([[0.2, 0.3], [0.3, 0.4]])
    h1 = np.array([[200.0, 210.0], [220.0, 230.0]])
    s1 = np.array([[20.0, 21.0], [22.0, 23.0]])
    mu1 = np.array([[8.0, 9.0], [8.0, 9.0]]) * 1e13
    c1 = np.array([[5.0, 6.0], [7.0, 8.0]])

    x_md = np.array([[1.0], [1.0]])
    y_md = np.array([[0.0], [1.0]])
    x_md_cell = np.empty((1, 1), dtype=object)
    x_md_cell[0, 0] = x_md
    y_md_cell = np.empty((1, 1), dtype=object)
    y_md_cell[0, 0] = y_md

    data = {
        'xd': _obj_pair(x0, x1),
        'yd': _obj_pair(y0, y1),
        'ud': _obj_pair(u0, u1),
        'vd': _obj_pair(v0, v1),
        'xd_h': _obj_pair(x0, x1),
        'yd_h': _obj_pair(y0, y1),
        'hd': _obj_pair(h0, h1),
        'sd': _obj_pair(s0, s1),
        'xd_s': _obj_pair(x0, x1),
        'yd_s': _obj_pair(y0, y1),
        'xcol': _obj_pair(x0, x1),
        'ycol': _obj_pair(y0, y1),
        'xct': _obj_pair(np.zeros((0, 0)), np.array([[3.0], [3.0]])),
        'yct': _obj_pair(np.zeros((0, 0)), np.array([[0.0], [1.0]])),
        'nnct': _obj_pair(np.zeros((0, 0)), np.array([[1.0, 0.0], [1.0, 0.0]])),
        'x_md': x_md_cell,
        'y_md': y_md_cell,
        'mud': _obj_pair(mu0, mu1),
        'alpha2d': _obj_pair(c0, c1),
        'Xe': np.zeros((2, 2)),
        'Ye': np.zeros((2, 2)),
        'Xe_h': np.zeros((2, 2)),
        'Ye_h': np.zeros((2, 2)),
        'idxcrop': np.array([[0, 1, 0, 1], [0, 1, 0, 1]]),
        'idxcrop_h': np.array([[0, 1, 0, 1], [0, 1, 0, 1]]),
    }
    return data


def test_preprocessing_builds_grounded_interface_mu_boundary_from_floating_side():
    data_all, idxgall, _, _ = normalize_data(
        _raw_two_region_data(),
        basal_mask=[False, True],
        use_regression=True,
        grounded_only_interface_mu_ct=True,
    )

    grounded = data_all[1]
    x_ct = grounded[2]
    x_md = grounded[5]
    mu_bd = grounded[6][0]
    mu0_grounded = grounded[4][6].dynamic_scale.mu0

    assert jnp.allclose(x_ct, x_md)
    assert mu_bd.shape == (2, 1)
    assert jnp.allclose(mu_bd.flatten(), jnp.array([7.0e13, 7.0e13]) / mu0_grounded)


def _region_data(offset, boundary=None):
    x_data = jnp.arange(offset, offset + 20, dtype=float).reshape(10, 2)
    u_data = jnp.arange(offset, offset + 20, dtype=float).reshape(10, 2)
    xh_data = jnp.arange(offset, offset + 20, dtype=float).reshape(10, 2)
    h_data = jnp.arange(offset, offset + 10, dtype=float).reshape(10, 1)
    s_data = h_data + 1.0
    mu_data = h_data + 2.0
    c_data = h_data + 3.0
    x_col = jnp.arange(offset, offset + 16, dtype=float).reshape(8, 2)
    x_ct = jnp.arange(offset, offset + 12, dtype=float).reshape(6, 2)
    nnct = jnp.zeros((6, 2))
    x_md = jnp.arange(offset, offset + 12, dtype=float).reshape(6, 2)
    return [[x_data, xh_data, x_col], [u_data, h_data, s_data, mu_data, c_data], x_ct, nnct, None, x_md, boundary]


def test_regression_sampler_returns_grounded_interface_mu_ct_payload():
    floating = _region_data(0, boundary=None)
    grounded = _region_data(100, boundary=[jnp.arange(6, dtype=float).reshape(6, 1) + 10.0])

    dataf = data_regression_sample_create(
        [floating, grounded],
        [0, 1],
        [[2, 3], [4, 5], [6, 7]],
        basal_mask=[False, True],
        grounded_only_interface_mu_ct=True,
    )

    data = dataf(random.PRNGKey(0))

    assert data['ct'][0][1].shape == (6, 2)
    assert data['ct'][1][1].shape == (6, 1)
    assert jnp.allclose(data['ct'][1][1], grounded[6][0])
    assert jnp.allclose(data['ct'][1][0], jnp.zeros((6, 1)))


def _pred_nn(params, x, idx):
    return jnp.array([0.0, 0.0, 0.0, 0.0, params[idx], 0.0])


def _grad_nn(params, x, idx):
    return jnp.zeros(17)


def _front_eqn(net, x, nn, scale):
    return jnp.zeros((2,)), None


def _loss_data(mu_target_grounded):
    zeros_xy = [jnp.zeros((2, 2)), jnp.zeros((2, 2))]
    zeros_uv = [jnp.zeros((2, 2)), jnp.zeros((2, 2))]
    zeros_hs = [jnp.zeros((2, 1)), jnp.zeros((2, 1))]
    ones_mu = [jnp.ones((2, 1)), jnp.ones((2, 1))]
    zeros_c = [jnp.zeros((2, 1)), jnp.zeros((2, 1))]
    return {
        'smp': [zeros_xy, zeros_uv, zeros_xy, zeros_hs, zeros_hs, ones_mu, zeros_c],
        'col': [zeros_xy],
        'ct': [zeros_xy, [jnp.zeros((2, 1)), jnp.full((2, 1), mu_target_grounded)]],
        'md': [[jnp.zeros((2, 4))]],
        'active_regions': [1],
        'eqn_region_weights': jnp.ones(2),
        'match_weight': 1.0,
    }


def test_grounded_interface_mu_ct_loss_uses_log_mu_and_active_regions():
    scales = [
        SubScaleResult(
            DataMean(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            DataRange(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            DynamicScale(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        SubScaleResult(
            DataMean(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            DataRange(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            DynamicScale(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ),
    ]
    loss_fn = loss_regression_create(
        (_pred_nn, _grad_nn), [0, 1],
        basal_mask=[False, True],
        eqn=None,
        front_eqn=_front_eqn,
        match=False,
        scales=scales,
        grounded_only_interface_mu_ct=True,
    )

    loss_match, _, reg_err_match = loss_fn(jnp.array([100.0, 2.0]), _loss_data(2.0))
    loss_mismatch, _, reg_err_mismatch = loss_fn(jnp.array([1000.0, 4.0]), _loss_data(2.0))

    assert jnp.allclose(reg_err_match[3][1, 0], 0.0)
    assert jnp.allclose(reg_err_mismatch[3][1, 0], jnp.log(2.0) ** 2)
    assert loss_match < loss_mismatch
