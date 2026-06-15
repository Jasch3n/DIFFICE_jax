import jax.numpy as jnp

from diffice_jax.data.xpinns.preprocessing import DataMean, DataRange, DynamicScale, SubScaleResult
from diffice_jax.model.xpinns.loss import loss_regression_create
from tests.test_xpinn_regression import xpinn_regression as xr

MATCH_INFO_TERMS = 19


def _gpinn_info_slice(data_terms=6, eqn_terms=2, match_terms=MATCH_INFO_TERMS, ct_terms=2):
    start = 1 + data_terms + eqn_terms + match_terms + ct_terms
    return slice(start, start + 5)


def _scale():
    return SubScaleResult(
        DataMean(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        DataRange(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        DynamicScale(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )


def _pred(params, x, idx):
    if x.ndim == 1:
        return params[idx]
    return jnp.broadcast_to(params[idx], (x.shape[0], params.shape[1]))


def _grad(params, x, idx):
    if x.ndim == 1:
        return jnp.zeros(17)
    return jnp.zeros((x.shape[0], 17))


def _data(n_sub=2):
    x = [jnp.zeros((2, 2)) for _ in range(n_sub)]
    return {
        'smp': [
            x,
            [jnp.ones((2, 2)) for _ in range(n_sub)],
            x,
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
        ],
        'col': [x],
        'md': [x],
    }


def _eqn_linear(net, x, scale, basal=False):
    return jnp.array([x[0] + 2.0 * x[1], 3.0 * x[0] - x[1]]), jnp.array([0.0])


def test_regression_kfac_residuals_match_active_data_loss_terms():
    params = jnp.array([
        [2.0, 3.0, 4.0, 5.0, 7.0, 11.0],
        [3.0, 5.0, 7.0, 9.0, 13.0, 17.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True]
    )
    data = _data()

    loss, _, _ = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss)


def test_regression_kfac_eval_matches_residual_objective_and_diagnostics():
    params = jnp.array([
        [2.0, 3.0, 4.0, 5.0, 7.0, 11.0],
        [3.0, 5.0, 7.0, 9.0, 13.0, 17.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        match=True, scales=[_scale(), _scale()]
    )
    data = _data()

    _, loss_info, reg_err_list = loss_fn(params, data)
    loss_n, eval_info, eval_reg_err_list, residuals = loss_fn.kfac_eval(params, data)

    assert jnp.allclose(residuals, loss_fn.kfac_residuals(params, data))
    assert jnp.allclose(loss_n, loss_fn.kfac_objective(params, data))
    assert jnp.allclose(eval_info, loss_info, equal_nan=True)
    for eval_item, item in zip(eval_reg_err_list, reg_err_list):
        assert jnp.allclose(eval_item, item, equal_nan=True)


def test_regression_kfac_residuals_include_matching_when_match_is_on():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 5.0, 7.0, 0.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        match=True, scales=[_scale(), _scale()]
    )
    data = _data()

    loss, _, _ = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss)


def test_regression_kfac_residuals_apply_default_matching_weight():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 5.0, 7.0, 0.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        match=True, scales=[_scale(), _scale()], match_weight=0.01
    )
    data = _data()

    loss, _, _ = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss)


def test_regression_kfac_residuals_apply_batch_matching_weight():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 5.0, 7.0, 0.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        match=True, scales=[_scale(), _scale()], match_weight=0.01
    )
    data = _data()
    data_with_weight = dict(data)
    data_with_weight['match_weight'] = jnp.array(0.25)

    loss_default, _, _ = loss_fn(params, data)
    loss_weighted, _, _ = loss_fn(params, data_with_weight)
    residuals = loss_fn.kfac_residuals(params, data_with_weight)

    assert loss_weighted > loss_default
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss_weighted)


def test_regression_kfac_residuals_filter_equation_region():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 5.0, 7.0, 0.0],
    ])

    def eqn(net, x, scale, basal=False):
        return jnp.array([1.0, 2.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        eqn=eqn, scales=[_scale(), _scale()]
    )
    data = _data()
    data['col'] = [[jnp.zeros((2, 2)), jnp.zeros((2, 2))]]

    residuals = loss_fn.kfac_residuals(data=data, params=params, terms=('eqn',), regions=[0])
    objective = loss_fn.kfac_objective(params, data, terms=('eqn',), regions=[0])

    assert residuals.shape == (4, 1)
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), objective)


def test_regression_equation_region_weight_scales_selected_region():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ])

    def eqn(net, x, scale, basal=False):
        return jnp.array([1.0, 2.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        eqn=eqn, scales=[_scale(), _scale()]
    )
    data = _data()
    weighted_data = dict(data)
    weighted_data['eqn_region_weights'] = jnp.array([4.0, 1.0])

    loss_default, _, _ = loss_fn(params, data)
    loss_weighted, loss_info, _ = loss_fn(params, weighted_data)
    residuals = loss_fn.kfac_residuals(params, weighted_data, terms=('eqn',), regions=[0])
    objective = loss_fn.kfac_objective(params, weighted_data, terms=('eqn',), regions=[0])

    assert loss_weighted > loss_default
    assert jnp.allclose(loss_info[7:9], jnp.array([2.5, 10.0]))
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), objective)


def test_regression_kfac_residuals_filter_matching_interfaces_by_region():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 5.0, 7.0, 0.0],
        [3.0, 5.0, 7.0, 9.0, 11.0, 0.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1, 2], basal_mask=[False, True, True],
        match=True, scales=[_scale(), _scale(), _scale()]
    )
    data = _data(n_sub=3)

    edge_residuals = loss_fn.kfac_residuals(params, data, terms=('match',), regions=[0])
    middle_residuals = loss_fn.kfac_residuals(params, data, terms=('match',), regions=[1])
    objective = loss_fn.kfac_objective(params, data, terms=('match',), regions=[0])

    assert edge_residuals.shape == (38, 1)
    assert middle_residuals.shape == (76, 1)
    assert jnp.allclose(jnp.sum(jnp.square(edge_residuals)), objective)


def test_regression_gpinn_regularization_default_is_off():
    params = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[False],
        eqn=_eqn_linear, scales=[_scale()]
    )
    data = _data(n_sub=1)

    _, loss_info, _ = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data, terms=('gpinn',), regions=[0])

    assert residuals.shape == (0, 1)
    assert jnp.allclose(loss_info[_gpinn_info_slice()], jnp.zeros(5))


def test_regression_gpinn_regularization_uses_equation_residual_gradient():
    params = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    data = _data(n_sub=1)
    weight = 2.0
    loss_off = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[True],
        eqn=_eqn_linear, scales=[_scale()]
    )
    loss_on = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[True],
        eqn=_eqn_linear, scales=[_scale()], gpinn_weight=weight
    )

    base_loss, _, _ = loss_off(params, data)
    reg_loss, loss_info, _ = loss_on(params, data)

    raw_gpinn_components = jnp.array([1.0, 4.0, 9.0, 1.0])
    raw_gpinn_loss = jnp.mean(raw_gpinn_components)
    assert jnp.allclose(reg_loss - base_loss, weight * raw_gpinn_loss)
    assert jnp.allclose(loss_info[_gpinn_info_slice()], jnp.hstack((raw_gpinn_loss, raw_gpinn_components)))


def test_regression_gpinn_kfac_objective_matches_residuals():
    params = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[True],
        eqn=_eqn_linear, scales=[_scale()], gpinn_weight=0.5
    )
    data = _data(n_sub=1)

    residuals = loss_fn.kfac_residuals(params, data, terms=('gpinn',), regions=[0])
    objective = loss_fn.kfac_objective(params, data, terms=('gpinn',), regions=[0])

    assert residuals.shape == (8, 1)
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), objective)


def test_regression_gpinn_regularization_uses_all_regions():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ])
    loss_on = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        eqn=_eqn_linear, scales=[_scale(), _scale()], gpinn_weight=2.0
    )
    data = _data(n_sub=2)

    _, loss_info, _ = loss_on(params, data)
    floating_residuals = loss_on.kfac_residuals(params, data, terms=('gpinn',), regions=[0])
    grounded_residuals = loss_on.kfac_residuals(params, data, terms=('gpinn',), regions=[1])

    raw_gpinn_components = jnp.array([1.0, 4.0, 9.0, 1.0])
    raw_gpinn_loss = jnp.mean(raw_gpinn_components)
    assert floating_residuals.shape == (8, 1)
    assert grounded_residuals.shape == (8, 1)
    assert jnp.allclose(loss_info[_gpinn_info_slice()], jnp.hstack((raw_gpinn_loss, raw_gpinn_components)))


def test_regression_kfac_residuals_match_equation_and_front_terms():
    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])

    def eqn(net, x, scale, basal=False):
        return jnp.array([1.0, 2.0]), jnp.array([0.0])

    def front_eqn(net, x, nn, scale):
        return jnp.array([3.0, 4.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[False],
        eqn=eqn, front_eqn=front_eqn, calving_front=True, scales=[_scale()]
    )
    data = _data(n_sub=1)
    data['col'] = [[jnp.zeros((2, 2))]]
    data['ct'] = [[jnp.zeros((2, 2))], [jnp.zeros((2, 2))]]

    loss, _, _ = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss)


def test_regression_equation_diagnostics_use_two_slots():
    params = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])

    def eqn(net, x, scale, basal=False):
        return jnp.array([1.0, 2.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[False],
        eqn=eqn, scales=[_scale()]
    )
    data = _data(n_sub=1)
    data['col'] = [[jnp.zeros((2, 2))]]

    _, loss_info, reg_err_list = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert reg_err_list[1].shape == (1, 2)
    assert jnp.allclose(loss_info[7:9], jnp.array([1.0, 4.0]))
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss_info[0])


def test_regression_kfac_residuals_omit_front_terms_when_calving_front_is_off():
    params = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])

    def front_eqn(net, x, nn, scale):
        return jnp.array([3.0, 4.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[False],
        front_eqn=front_eqn, calving_front=False, scales=[_scale()]
    )
    data = _data(n_sub=1)
    data['ct'] = [[jnp.zeros((2, 2))], [jnp.zeros((2, 2))]]

    loss, _, _ = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert jnp.allclose(loss, 0.0)
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss)


def test_regression_region_term_weights_override_scalar_loss():
    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])

    def eqn(net, x, scale, basal=False):
        return jnp.array([1.0, 2.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[False],
        eqn=eqn, scales=[_scale()]
    )
    data = _data(n_sub=1)
    data['col'] = [[jnp.zeros((2, 2))]]
    weighted_data = dict(data)
    weighted_data['region_term_weights'] = {
        'data': jnp.array([1.0]),
        'eqn': jnp.array([0.0]),
        'ct': jnp.array([0.0]),
        'match': jnp.array([0.0]),
        'gpinn': jnp.array([0.0]),
        'mu_grad': jnp.array([0.0]),
    }

    weighted_loss, _, _ = loss_fn(params, weighted_data)
    weighted_residuals = loss_fn.kfac_residuals(params, weighted_data)

    assert jnp.allclose(weighted_loss, jnp.mean(jnp.array([1.0, 4.0, 9.0, 16.0, 0.0, 0.0])))
    assert jnp.allclose(jnp.sum(jnp.square(weighted_residuals)), weighted_loss)


def test_regression_region_term_weights_scale_equation_objective_without_legacy_factor():
    params = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])

    def eqn(net, x, scale, basal=False):
        return jnp.array([1.0, 2.0]), jnp.array([0.0])

    loss_fn = loss_regression_create(
        (_pred, _grad), [0], basal_mask=[False],
        eqn=eqn, scales=[_scale()]
    )
    data = _data(n_sub=1)
    data['col'] = [[jnp.zeros((2, 2))]]
    data['region_term_weights'] = {
        'data': jnp.array([0.0]),
        'eqn': jnp.array([0.25]),
        'ct': jnp.array([0.0]),
        'match': jnp.array([0.0]),
        'gpinn': jnp.array([0.0]),
        'mu_grad': jnp.array([0.0]),
    }

    residuals = loss_fn.kfac_residuals(params, data, terms=('eqn',), regions=[0])
    objective = loss_fn.kfac_objective(params, data, terms=('eqn',), regions=[0])

    assert jnp.allclose(objective, 0.25 * jnp.mean(jnp.array([1.0, 4.0])))
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), objective)


def test_regression_region_term_weights_average_matching_interface_weights():
    params = jnp.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 5.0, 7.0, 0.0],
    ])
    loss_fn = loss_regression_create(
        (_pred, _grad), [0, 1], basal_mask=[False, True],
        match=True, scales=[_scale(), _scale()]
    )
    data = _data()
    data['region_term_weights'] = {
        'data': jnp.array([0.0, 0.0]),
        'eqn': jnp.array([0.0, 0.0]),
        'ct': jnp.array([0.0, 0.0]),
        'match': jnp.array([0.2, 0.8]),
        'gpinn': jnp.array([0.0, 0.0]),
        'mu_grad': jnp.array([0.0, 0.0]),
    }

    residuals = loss_fn.kfac_residuals(params, data, terms=('match',), regions=[0])
    objective = loss_fn.kfac_objective(params, data, terms=('match',), regions=[0])
    expected_match = 0.5 * loss_fn.region_term_values(params, data)['match'][0]

    assert jnp.allclose(objective, expected_match)
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), objective)


def test_compute_region_term_weights_uses_gradient_norm_over_sum_rule():
    params = jnp.array([2.0])
    frozen_params = params
    freeze_mask = jnp.zeros_like(params, dtype=bool)
    previous = xr.initialize_region_term_weights([0], [True])

    class FakeLoss:
        @staticmethod
        def region_term_values(active_params, batch):
            p = active_params[0]
            return {
                'data': jnp.array([2.0 * p]),
                'eqn': jnp.array([1.0 * p]),
                'ct': jnp.array([0.0]),
                'match': jnp.array([0.0]),
                'gpinn': jnp.array([0.5 * p]),
                'mu_grad': jnp.array([0.0]),
            }

    weights = xr.compute_region_term_weights(
        params=params,
        batch={},
        loss_f=FakeLoss(),
        idxgall=[0],
        basal_mask=[True],
        frozen_params=frozen_params,
        freeze_mask=freeze_mask,
        previous_weights=previous,
    )

    assert jnp.allclose(weights['data'], jnp.array([2.0 / 3.5]))
    assert jnp.allclose(weights['eqn'], jnp.array([1.0 / 3.5]))
    assert jnp.allclose(weights['gpinn'], jnp.array([0.5 / 3.5]))
    assert jnp.allclose(weights['ct'], jnp.array([0.0]))
    assert jnp.allclose(weights['match'], jnp.array([0.0]))
    assert jnp.allclose(weights.get('mu_grad', jnp.array([0.0])), jnp.array([0.0]))
