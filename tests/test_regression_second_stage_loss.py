import jax
import jax.numpy as jnp

from diffice_jax.model.xpinns.loss import loss_regression_2ndstage_create


def _pred(params, x, idx):
    return jnp.broadcast_to(params[idx], (x.shape[0], params.shape[1]))


def _grad(params, x, idx):
    return jnp.zeros((x.shape[0], 17))


def _data(h_value, s_value):
    x = [jnp.zeros((2, 2))]
    return {
        'smp': [
            x,
            [jnp.array([[1.0, 2.0], [1.0, 2.0]])],
            x,
            [jnp.full((2, 1), h_value)],
            [jnp.full((2, 1), s_value)],
            [jnp.ones((2, 1))],
            [jnp.ones((2, 1))],
        ],
    }


def test_second_stage_ignores_observed_h_and_s():
    params = jnp.array([[1.2, 1.8, 3.0, 4.0, 2.0, 1.0]])
    x_all = jnp.zeros((2, 2))
    residual = jnp.array([[0.2, -0.2], [0.2, -0.2]])
    loss_fn = loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True],
        first_stage_velocity_misfit={0: (x_all, residual)})

    loss_a = loss_fn(params, _data(10.0, 20.0))[0]
    loss_b = loss_fn(params, _data(1000.0, 2000.0))[0]

    assert jnp.allclose(loss_a, loss_b)


def test_second_stage_first_stage_params_are_frozen():
    params = jnp.array([[0.3, -0.1, 3.0, 4.0, 2.0, 1.0]])
    first_stage_params = jnp.array([[1.0, 2.0, 3.5, 4.5, 2.5, 1.5]])
    x_all = jnp.zeros((2, 2))
    residual = jnp.array([[0.2, -0.2], [0.2, -0.2]])
    data = _data(10.0, 20.0)

    def loss_with_first_stage(first_params):
        loss_fn = loss_regression_2ndstage_create(
            (_pred, _grad), [0], basal_mask=[True],
            first_stage_velocity_misfit={0: (x_all, residual)})
        return loss_fn(params, data)[0]

    grad_first = jax.grad(loss_with_first_stage)(first_stage_params)
    grad_new = jax.grad(lambda p: loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True],
        first_stage_velocity_misfit={0: (x_all, residual)})(p, data)[0])(params)

    assert jnp.allclose(grad_first, 0.0)
    assert jnp.any(jnp.abs(grad_new[:, 0:2]) > 0.0)
    assert jnp.allclose(grad_new[:, 2:], 0.0)


def test_second_stage_fits_first_stage_velocity_residual():
    params = jnp.array([[0.2, -0.2, 3.0, 4.0, 2.0, 1.0]])
    x_all = jnp.zeros((2, 2))
    residual = jnp.array([[0.2, -0.2], [0.2, -0.2]])
    loss_fn = loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True],
        first_stage_velocity_misfit={0: (x_all, residual)})

    loss = loss_fn(params, _data(10.0, 20.0))[0]

    zero_velocity_loss_params = params.at[0, 0:2].set(jnp.array([1.0, 2.0]))
    full_velocity_loss = loss_fn(zero_velocity_loss_params, _data(10.0, 20.0))[0]

    assert loss < full_velocity_loss


def test_second_stage_uses_precomputed_first_stage_velocity_residual():
    params = jnp.array([[0.2, -0.2, 3.0, 4.0, 2.0, 1.0]])
    x_all = jnp.zeros((2, 2))
    residual = jnp.array([[0.2, -0.2], [0.2, -0.2]])
    loss_fn = loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True],
        first_stage_velocity_misfit={0: (x_all, residual)})

    loss = loss_fn(params, _data(10.0, 20.0))[0]

    stale_dynamic_residual_params = params.at[0, 0:2].set(jnp.array([-4.0, -3.0]))
    dynamic_residual_loss = loss_fn(stale_dynamic_residual_params, _data(10.0, 20.0))[0]

    assert loss < dynamic_residual_loss


def test_second_stage_kfac_residuals_are_velocity_only():
    params = jnp.array([[0.2, -0.2, 3.0, 4.0, 2.0, 1.0]])
    x_all = jnp.zeros((2, 2))
    residual = jnp.array([[0.2, -0.2], [0.2, -0.2]])
    loss_fn = loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True],
        first_stage_velocity_misfit={0: (x_all, residual)})

    residuals = loss_fn.kfac_residuals(params, _data(10.0, 20.0))

    assert residuals.shape == (4, 1)
    assert jnp.allclose(residuals, 0.0, atol=1e-7)


def test_second_stage_normalizes_velocity_residuals_by_rms():
    params = jnp.array([[1.1, -1.05, 3.0, 4.0, 2.0, 1.0]])
    x_all = jnp.zeros((2, 2))
    residual = jnp.array([[0.2, -0.2], [0.2, -0.2]])
    residual_rms = jnp.array([2.0, 4.0])
    loss_fn = loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True],
        first_stage_velocity_misfit={0: (x_all, residual, residual_rms)})

    _, loss_info, _ = loss_fn(params, _data(10.0, 20.0))

    assert jnp.allclose(loss_info[1], 1.0)
    assert jnp.allclose(loss_info[2], 1.0)


def test_second_stage_can_use_sampled_normalized_residual_target_directly():
    params = jnp.array([[0.2, -0.2, 3.0, 4.0, 2.0, 1.0]])
    loss_fn = loss_regression_2ndstage_create(
        (_pred, _grad), [0], basal_mask=[True])

    wrong_loss = loss_fn(params, _data(10.0, 20.0))[0]
    target_params = params.at[0, 0:2].set(jnp.array([1.0, 2.0]))
    loss = loss_fn(target_params, _data(10.0, 20.0))[0]

    assert loss < wrong_loss
