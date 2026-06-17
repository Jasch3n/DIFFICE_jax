import jax.numpy as jnp
from jax import random

from diffice_jax.data.xpinns import sampling as xpinn_sampling
from diffice_jax.data.xpinns.sampling import data_regression_sample_create, data_sample_create


def _region_data(offset):
    x_data = jnp.arange(offset, offset + 20, dtype=float).reshape(10, 2)
    u_data = jnp.arange(offset, offset + 20, dtype=float).reshape(10, 2)
    xh_data = jnp.arange(offset, offset + 20, dtype=float).reshape(10, 2)
    h_data = jnp.arange(offset, offset + 10, dtype=float).reshape(10, 1)
    s_data = h_data + 1.0
    mu_data = h_data + 2.0
    c_data = h_data + 3.0
    x_col = jnp.arange(offset, offset + 16, dtype=float).reshape(8, 2)
    x_ct = jnp.zeros((0, 2))
    nnct = jnp.zeros((0, 2))
    x_md = jnp.arange(offset, offset + 12, dtype=float).reshape(6, 2)
    return [[x_data, xh_data, x_col], [u_data, h_data, s_data, mu_data, c_data], x_ct, nnct, None, x_md]


def test_regression_sampler_uses_independent_collocation_counts():
    dataf = data_regression_sample_create(
        [_region_data(0), _region_data(100)],
        [0, 1],
        [[2, 3], [4, 5], [6, 7]],
        basal_mask=[False, True],
    )

    data = dataf(random.PRNGKey(0))

    assert data['smp'].X_smp[0].shape[0] == 2
    assert data['smp'].X_smp[1].shape[0] == 3
    assert data['smp'].Xh_smp[0].shape[0] == 4
    assert data['smp'].Xh_smp[1].shape[0] == 5
    assert data['col'][0][0].shape[0] == 6 + xpinn_sampling.N_INTERFACE_COLLOCATION
    assert data['col'][0][1].shape[0] == 7 + xpinn_sampling.N_INTERFACE_COLLOCATION


def test_regression_sampler_keeps_velocity_count_as_collocation_fallback():
    dataf = data_regression_sample_create(
        [_region_data(0), _region_data(100)],
        [0, 1],
        [[2, 3], [4, 5]],
        basal_mask=[False, True],
    )

    data = dataf(random.PRNGKey(1))

    assert data['col'][0][0].shape[0] == 2 + xpinn_sampling.N_INTERFACE_COLLOCATION
    assert data['col'][0][1].shape[0] == 3 + xpinn_sampling.N_INTERFACE_COLLOCATION


def test_regression_sampler_does_not_add_interface_collocation_for_same_mask_interface():
    dataf = data_regression_sample_create(
        [_region_data(0), _region_data(100)],
        [0, 1],
        [[2, 3], [4, 5], [6, 7]],
        basal_mask=[False, False],
    )

    data = dataf(random.PRNGKey(2))

    assert data['col'][0][0].shape[0] == 6
    assert data['col'][0][1].shape[0] == 7


def test_sampler_adds_interface_collocation_to_regular_xpinn_batches():
    dataf = data_sample_create(
        [_region_data(0), _region_data(100)],
        [0, 1],
        [[2, 3], [4, 5], [6, 7], [1, 1], [2]],
        basal_mask=[False, True],
    )

    data = dataf(random.PRNGKey(3))

    assert data['col'][0][0].shape[0] == 6 + xpinn_sampling.N_INTERFACE_COLLOCATION
    assert data['col'][0][1].shape[0] == 7 + xpinn_sampling.N_INTERFACE_COLLOCATION


def test_regression_sampler_pairs_two_region_matching_coordinates():
    dataf = data_regression_sample_create(
        [_region_data(0), _region_data(100)],
        [0, 1],
        [[2, 3], [4, 5], [6, 7]],
        basal_mask=[False, True],
    )

    data = dataf(random.PRNGKey(2))
    x_md = data['md'][0][0]

    assert x_md.shape == (6, 4)
    assert jnp.allclose(x_md[:, 0:2], _region_data(0)[5])
    assert jnp.allclose(x_md[:, 2:4], _region_data(100)[5])


def test_regression_sampler_splits_middle_region_matching_coordinates():
    region0 = _region_data(0)
    region1 = _region_data(100)
    region2 = _region_data(200)
    left_mid = jnp.arange(100, 112, dtype=float).reshape(6, 2)
    right_mid = jnp.arange(300, 308, dtype=float).reshape(4, 2)
    right_next = jnp.arange(400, 408, dtype=float).reshape(4, 2)
    region1[5] = jnp.vstack([left_mid, right_mid])
    region2[5] = right_next
    dataf = data_regression_sample_create(
        [region0, region1, region2],
        [0, 1, 2],
        [[2, 2, 2], [4, 4, 4], [6, 6, 6]],
        basal_mask=[False, True, True],
    )

    data = dataf(random.PRNGKey(3))

    assert data['md'][0][0].shape == (6, 4)
    assert jnp.allclose(data['md'][0][0][:, 0:2], region0[5])
    assert jnp.allclose(data['md'][0][0][:, 2:4], left_mid)
    assert data['md'][0][1].shape == (4, 4)
    assert jnp.allclose(data['md'][0][1][:, 0:2], right_mid)
    assert jnp.allclose(data['md'][0][1][:, 2:4], right_next)
