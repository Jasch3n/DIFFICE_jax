import jax.numpy as jnp
from jax import random

from diffice_jax.data.xpinns.preprocessing import DataMean, DataRange, DynamicScale, SubScaleResult
from diffice_jax.model.xpinns.initialization import init_nets
from diffice_jax.model.xpinns.networks import neural_net, solu_create


def _scale(h_mean=1000.0, s_mean=50.0, s_range=200.0):
    return SubScaleResult(
        DataMean(0.0, 0.0, 0.0, 0.0, h_mean, s_mean),
        DataRange(1.0, 1.0, 1.0, 1.0, 100.0, s_range),
        DynamicScale(1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0),
    )


def test_floating_surface_is_derived_from_thickness():
    scales = [_scale()]
    params = init_nets(random.PRNGKey(0), 2, 8, n_sub=1, basal_mask=[False])
    pred, _ = solu_create(scales, basal_mask=[False])
    x = jnp.ones((4, 2))

    out = pred(params, x, 0)
    h = out[:, 2:3] * scales[0].data_mean.h_mean
    s = out[:, 3:4] * scales[0].data_range.s_range + scales[0].data_mean.s_mean
    alpha = (scales[0].dynamic_scale.rho_w - scales[0].dynamic_scale.rho) / scales[0].dynamic_scale.rho_w

    assert out.shape[1] == 6
    assert jnp.allclose(s, alpha * h)
    assert jnp.allclose(out[:, 5:6], 0.0)


def test_floating_surface_handles_single_point_calls():
    scales = [_scale()]
    params = init_nets(random.PRNGKey(2), 2, 8, n_sub=1, basal_mask=[False])
    pred, _ = solu_create(scales, basal_mask=[False])
    x = jnp.ones((2,))

    out = pred(params, x, 0)
    h = out[2] * scales[0].data_mean.h_mean
    s = out[3] * scales[0].data_range.s_range + scales[0].data_mean.s_mean
    alpha = (scales[0].dynamic_scale.rho_w - scales[0].dynamic_scale.rho) / scales[0].dynamic_scale.rho_w

    assert out.shape == (6,)
    assert jnp.allclose(s, alpha * h)
    assert jnp.allclose(out[5], 0.0)


def test_grounded_surface_remains_network_output():
    scales = [_scale(), _scale()]
    params = init_nets(random.PRNGKey(1), 2, 8, n_sub=2, basal_mask=[False, True])
    pred, _ = solu_create(scales, basal_mask=[False, True])
    x = jnp.ones((4, 2))

    raw_grounded = neural_net(params["net_u"][1], x, 1)
    out_grounded = pred(params, x, 1)

    assert jnp.allclose(out_grounded[:, 3:4], raw_grounded[:, 3:4])
