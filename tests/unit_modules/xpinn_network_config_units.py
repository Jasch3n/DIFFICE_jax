import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import random

from diffice_jax.data.xpinns.preprocessing import DataMean, DataRange, DynamicScale, SubScaleResult
from diffice_jax.model.xpinns.initialization import init_nets
from diffice_jax.model.xpinns.networks import solu_create


def _scale():
    return SubScaleResult(
        DataMean(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        DataRange(2.0, 3.0, 4.0, 5.0, 1.0, 6.0),
        DynamicScale(1.5, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0),
    )


def test_xpinn_network_config_sets_each_region_network_shape():
    network_config = [
        dict(
            net_u=dict(depth=2, width=8),
            net_mu=dict(depth=3, width=9),
        ),
        dict(
            net_u=dict(depth=4, width=10),
            net_mu=dict(depth=1, width=11),
            net_c=dict(depth=2, width=12),
        ),
    ]
    params = init_nets(
        random.PRNGKey(0), 6, 30, n_sub=2,
        basal_mask=[False, True], network_config=network_config)

    assert len(params["net_u"][0]) == 3
    assert params["net_u"][0][0][0].shape == (2, 8)
    assert params["net_u"][0][-1][0].shape == (8, 4)

    assert len(params["net_mu"][0]) == 4
    assert params["net_mu"][0][0][0].shape == (2, 9)
    assert params["net_mu"][0][-1][0].shape == (9, 1)

    assert len(params["net_u"][1]) == 5
    assert params["net_u"][1][0][0].shape == (2, 10)
    assert params["net_u"][1][-1][0].shape == (10, 4)

    assert len(params["net_mu"][1]) == 2
    assert params["net_mu"][1][0][0].shape == (2, 11)
    assert params["net_mu"][1][-1][0].shape == (11, 1)

    assert params["net_c"][0] is None
    assert len(params["net_c"][1]) == 3
    assert params["net_c"][1][0][0].shape == (2, 12)
    assert params["net_c"][1][-1][0].shape == (12, 1)


def test_xpinn_network_config_keeps_scalar_defaults():
    params = init_nets(random.PRNGKey(1), 4, 25, n_sub=2, basal_mask=[False, True])

    assert len(params["net_u"][0]) == 5
    assert params["net_u"][0][0][0].shape == (2, 25)
    assert len(params["net_mu"][1]) == 5
    assert params["net_mu"][1][0][0].shape == (2, 25)
    assert len(params["net_c"][1]) == 5
    assert params["net_c"][1][0][0].shape == (2, 25)


def test_xpinn_network_config_respects_embedding_input_widths():
    embedding_config = [
        dict(embedding_u=False, embedding_mu=False, embedding_c=False),
        dict(embedding_u=True, embedding_mu=True, embedding_c=True, embed_n=5),
    ]
    network_config = [
        dict(),
        dict(
            net_u=dict(depth=2, width=7),
            net_mu=dict(depth=1, width=9),
            net_c=dict(depth=1, width=11),
        ),
    ]
    params = init_nets(
        random.PRNGKey(2), 4, 25, n_sub=2, basal_mask=[False, True],
        embedding_config=embedding_config, network_config=network_config)

    assert params["net_u"][1][0][0].shape == (10, 7)
    assert len(params["net_u"][1]) == 4
    assert params["net_u"][1][-1][0].shape == (5, 2)

    assert params["net_mu"][1][0][0].shape == (10, 9)
    assert len(params["net_mu"][1]) == 3
    assert params["net_mu"][1][-1][0].shape == (5, 2)

    assert params["net_c"][1][0][0].shape == (10, 11)
    assert len(params["net_c"][1]) == 3
    assert params["net_c"][1][-1][0].shape == (5, 2)


def test_xpinn_gradf_matches_legacy_per_output_value_and_grad_layout():
    params = init_nets(random.PRNGKey(3), 2, 8, n_sub=1, basal_mask=[True])
    pred, grad = solu_create([_scale()], basal_mask=[True])
    x = jnp.array([[0.2, -0.4], [0.7, 0.3]])
    actual = grad(params, x, 0)

    def legacy_grad_point(z):
        vals_grads = [value_and_grad(lambda zz, i=i: pred(params, zz, 0)[i])(z) for i in range(6)]
        return jnp.ravel(jnp.stack([grad_i for _, grad_i in vals_grads]), order='C')

    raw_grad = jax.vmap(legacy_grad_point)(x)
    scale = _scale()
    lx0, ly0, u0, v0 = scale.data_range[0:4]
    u0m = max(u0, v0)
    l0m = min(lx0, ly0)
    coeff = jnp.hstack([u0/u0m/(lx0/l0m), u0/u0m/(ly0/l0m),
                        v0/u0m/(lx0/l0m), v0/u0m/(ly0/l0m),
                        1/(lx0/l0m), 1/(ly0/l0m)])
    duvh = raw_grad[:, 0:6] * coeff
    strate = (duvh[:, 0] ** 2 + duvh[:, 3] ** 2
              + 0.25 * (duvh[:, 1] + duvh[:, 2]) ** 2
              + duvh[:, 0] * duvh[:, 3]) ** 0.5
    expected = jnp.hstack([duvh, strate[:, None], raw_grad])

    assert jnp.allclose(actual, expected)
