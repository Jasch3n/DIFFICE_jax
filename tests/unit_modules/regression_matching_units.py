import jax
import jax.numpy as jnp
import pytest

import diffice_jax as djax
from diffice_jax.data.xpinns.preprocessing import DataMean, DataRange, DynamicScale, SubScaleResult


def test_regression_matching_mu_is_symmetric_across_grounding_line():
    idxgall = [0, 1]
    basal_mask = [False, True]
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

    def pred_nn(params, x, idx):
        mu = jnp.exp(params[idx])
        return jnp.array([0.0, 0.0, 0.0, 0.0, mu, 0.0])

    def grad_nn(params, x, idx):
        return jnp.zeros(17)

    loss_fn = djax.loss_regression_xpinn(
        (pred_nn, grad_nn), idxgall, basal_mask=basal_mask, match=True, scales=scales
    )

    zeros_xy = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_uv = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_hs = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    ones_mu = [jnp.ones((1, 1)), jnp.ones((1, 1))]
    zeros_c = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    data = {
        'smp': [zeros_xy, zeros_uv, zeros_xy, zeros_hs, zeros_hs, ones_mu, zeros_c],
        'md': [zeros_xy],
    }

    params = jnp.array([0.0, jnp.log(2.0)])
    grad = jax.grad(lambda p: loss_fn(p, data)[0])(params)

    assert abs(float(grad[0])) > 1e-8
    assert abs(float(grad[1])) > 1e-8


def test_regression_matching_accepts_paired_interface_coordinates():
    idxgall = [0, 1]
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

    def pred_nn(params, x, idx):
        mu = jnp.exp(params[idx] + x[0])
        return jnp.array([0.0, 0.0, 0.0, 0.0, mu, 0.0])

    def grad_nn(params, x, idx):
        return jnp.zeros(17)

    loss_fn = djax.loss_regression_xpinn(
        (pred_nn, grad_nn), idxgall, basal_mask=[False, True], match=True, scales=scales
    )
    zeros_xy = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_uv = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_hs = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    ones_mu = [jnp.ones((1, 1)), jnp.ones((1, 1))]
    zeros_c = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    data = {
        'smp': [zeros_xy, zeros_uv, zeros_xy, zeros_hs, zeros_hs, ones_mu, zeros_c],
        'md': [[jnp.array([[0.25, 0.0, 0.75, 0.0]])]],
    }

    _, _, reg_err_list = loss_fn(jnp.array([0.0, 0.0]), data)

    assert reg_err_list[2][0, 4] == pytest.approx(0.25, abs=1e-7)


def test_regression_matching_matches_surface_value():
    idxgall = [0, 1]
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

    def pred_nn(params, x, idx):
        return jnp.array([0.0, 0.0, 0.0, params[idx], 1.0, 0.0])

    def grad_nn(params, x, idx):
        return jnp.zeros(17)

    loss_fn = djax.loss_regression_xpinn(
        (pred_nn, grad_nn), idxgall, basal_mask=[False, True], match=True, scales=scales
    )
    zeros_xy = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_uv = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_hs = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    ones_mu = [jnp.ones((1, 1)), jnp.ones((1, 1))]
    zeros_c = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    data = {
        'smp': [zeros_xy, zeros_uv, zeros_xy, zeros_hs, zeros_hs, ones_mu, zeros_c],
        'md': [zeros_xy],
    }

    _, _, reg_err_list = loss_fn(jnp.array([0.0, 2.0]), data)

    assert reg_err_list[2][0, 3] == pytest.approx(4.0, abs=1e-7)


def test_regression_matching_reports_current_derivative_layout():
    idxgall = [0, 1]
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

    def pred_nn(params, x, idx):
        return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def grad_nn(params, x, idx):
        grad = jnp.zeros(17)
        return grad.at[13:15].set(params[idx])

    loss_fn = djax.loss_regression_xpinn(
        (pred_nn, grad_nn), idxgall, basal_mask=[False, True], match=True, scales=scales
    )
    zeros_xy = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_uv = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_hs = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    ones_mu = [jnp.ones((1, 1)), jnp.ones((1, 1))]
    zeros_c = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    data = {
        'smp': [zeros_xy, zeros_uv, zeros_xy, zeros_hs, zeros_hs, ones_mu, zeros_c],
        'md': [zeros_xy],
    }

    _, _, reg_err_list = loss_fn(jnp.array([0.0, 2.0]), data)

    assert reg_err_list[2].shape == (1, 19)
    assert jnp.allclose(reg_err_list[2][0, 5:19], 0.0)


def test_regression_matching_uses_log_mu_gradient():
    idxgall = [0, 1]
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

    def pred_nn(params, x, idx):
        return jnp.array([0.0, 0.0, 0.0, 0.0, params[idx], 0.0])

    def grad_nn(params, x, idx):
        grad = jnp.zeros(17)
        return grad.at[15:17].set(1.0)

    loss_fn = djax.loss_regression_xpinn(
        (pred_nn, grad_nn), idxgall, basal_mask=[False, True], match=True, scales=scales
    )
    zeros_xy = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_uv = [jnp.zeros((1, 2)), jnp.zeros((1, 2))]
    zeros_hs = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    ones_mu = [jnp.ones((1, 1)), jnp.ones((1, 1))]
    zeros_c = [jnp.zeros((1, 1)), jnp.zeros((1, 1))]
    data = {
        'smp': [zeros_xy, zeros_uv, zeros_xy, zeros_hs, zeros_hs, ones_mu, zeros_c],
        'md': [zeros_xy],
    }

    _, _, reg_err_list = loss_fn(jnp.array([2.0, 4.0]), data)

    assert reg_err_list[2][0, 11] == pytest.approx(0.0625, abs=1e-7)
    assert reg_err_list[2][0, 12] == pytest.approx(0.0625, abs=1e-7)
