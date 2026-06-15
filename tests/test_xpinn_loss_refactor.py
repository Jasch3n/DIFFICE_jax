import jax.numpy as jnp
import pytest

import diffice_jax as djax
from diffice_jax.core.loss_terms import JointInversionLossBuilder, RegressionLossBuilder
from diffice_jax.core.solver import limit_xpinn_batch
from diffice_jax.data.xpinns.preprocessing import DataMean, DataRange, DynamicScale, SubScaleResult
from diffice_jax.model.xpinns.loss import (
    DIFFICEJointInversionConfig,
    DIFFICEXPINNRegressionConfig,
    loss_joint_create,
    loss_regression_create,
)


class RealDataSample:
    __slots__ = ("X_smp", "U_smp", "Xh_smp", "H_smp", "S_smp")

    def __init__(self, X_smp, U_smp, Xh_smp, H_smp, S_smp):
        self.X_smp = X_smp
        self.U_smp = U_smp
        self.Xh_smp = Xh_smp
        self.H_smp = H_smp
        self.S_smp = S_smp

    def __getitem__(self, idx):
        return (self.X_smp, self.U_smp, self.Xh_smp, self.H_smp, self.S_smp)[idx]


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


def _real_data(n_sub=1):
    x = [jnp.zeros((2, 2)) for _ in range(n_sub)]
    return {
        "smp": RealDataSample(
            x,
            [jnp.ones((2, 2)) for _ in range(n_sub)],
            x,
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
        ),
        "col": [x],
        "bd": [x, x],
        "md": [x],
    }


def _regression_data(n_sub=1):
    x = [jnp.zeros((2, 2)) for _ in range(n_sub)]
    return {
        "smp": [
            x,
            [jnp.ones((2, 2)) for _ in range(n_sub)],
            x,
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
            [jnp.ones((2, 1)) for _ in range(n_sub)],
        ],
        "col": [x],
        "ct": [x, x],
        "md": [x],
    }


def test_loss_joint_create_evaluates_without_inverse_targets():
    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])
    config = DIFFICEJointInversionConfig(
        idxgall=(0,),
        basal_mask=(False,),
        scales=(_scale(),),
    )
    loss_fn = loss_joint_create((_pred, _grad), None, None, config)
    data = _real_data()

    loss, info, reg_err = loss_fn(params, data)
    residuals = loss_fn.kfac_residuals(params, data)

    assert info.shape[0] == 36
    assert reg_err[0].shape == (1, 4)
    assert residuals.shape == (8, 1)
    assert jnp.allclose(jnp.sum(jnp.square(residuals)), loss)


def test_loss_regression_create_config_and_legacy_signatures_keep_inverse_targets():
    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])
    config = DIFFICEXPINNRegressionConfig(
        idxgall=(0,),
        basal_mask=(False,),
        scales=(_scale(),),
    )
    config_loss = loss_regression_create((_pred, _grad), None, None, config)
    legacy_loss = loss_regression_create((_pred, _grad), [0], basal_mask=[False])
    data = _regression_data()

    _, _, config_err = config_loss(params, data)
    _, _, legacy_err = legacy_loss(params, data)

    assert config_err[0].shape == (1, 6)
    assert legacy_err[0].shape == (1, 6)


def test_loss_builders_return_static_callable_losses():
    loss_config = djax.LossConfig(name="joint_inversion", matching=False, calving_front=False)
    joint = JointInversionLossBuilder(loss_config).create(
        (_pred, _grad), [0], [False], None, None, [_scale()]
    )
    regression = RegressionLossBuilder(loss_config, djax.DataConfig(source={}, sampling_counts=[])).create(
        (_pred, _grad), [0], [False], None, None, [_scale()]
    )

    assert isinstance(hash(joint), int)
    assert isinstance(hash(regression), int)
    assert hasattr(joint, "lref")
    assert hasattr(regression, "kfac_objective")


def test_limit_xpinn_batch_only_limits_interface_key():
    x = jnp.zeros((4, 2))
    batch = {"bd": [[x], [x]], "md": [[jnp.zeros((4, 4))]]}

    limited = limit_xpinn_batch(batch, interface_points=2)

    assert limited["md"][0][0].shape[0] == 2
    assert limited["bd"][0][0].shape[0] == 4
    assert limited["bd"][1][0].shape[0] == 4


def test_gpinn_uses_full_collocation_batch_when_interface_tail_is_present():
    def eqn(_, x, __, basal=False):
        return jnp.array([x[0], x[1]]), jnp.zeros((1,))

    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])
    config = DIFFICEJointInversionConfig(
        idxgall=(0,),
        basal_mask=(False,),
        scales=(_scale(),),
        gpinn_weight=1.0,
    )
    loss_fn = loss_joint_create((_pred, _grad), eqn, None, config)
    data = _real_data()
    data["col"] = [[jnp.zeros((3, 2))]]
    data["gpinn_col"] = [[jnp.zeros((1, 2))]]

    residuals = loss_fn.kfac_residuals(params, data, terms=("gpinn",))

    assert residuals.shape == (12, 1)


def test_training_global_weights_override_xpinn_loss_weights():
    def eqn(_, x, __, basal=False):
        return jnp.array([1.0, 1.0]), jnp.zeros((1,))

    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])
    data = _real_data()
    data["col"] = [[jnp.zeros((2, 2))]]
    config = DIFFICEJointInversionConfig(
        idxgall=(0,),
        basal_mask=(False,),
        scales=(_scale(),),
        global_weights={"equation": 0.25},
    )
    loss_fn = loss_joint_create((_pred, _grad), eqn, None, config)

    residuals = loss_fn.kfac_residuals(params, data, terms=("eqn",))

    assert jnp.allclose(jnp.sum(jnp.square(residuals)), 0.25)


def test_zero_equation_global_weight_warns_and_skips_equation_computation():
    def eqn(_, x, __, basal=False):
        raise AssertionError("equation residual should be skipped")

    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])
    config = DIFFICEJointInversionConfig(
        idxgall=(0,),
        basal_mask=(False,),
        scales=(_scale(),),
        global_weights={"equation": 0.0},
    )
    with pytest.warns(RuntimeWarning, match="Global weight for equation is zero"):
        loss_fn = loss_joint_create((_pred, _grad), eqn, None, config)

    data = _real_data()
    residuals = loss_fn.kfac_residuals(params, data, terms=("eqn",))
    loss, info, _ = loss_fn(params, data)

    assert residuals.shape == (0, 1)
    assert jnp.allclose(info[5:7], jnp.zeros(2))
    assert jnp.isfinite(loss)


def test_zero_data_global_weight_warns_and_skips_data_residuals():
    params = jnp.array([[2.0, 3.0, 4.0, 5.0, 7.0, 11.0]])
    config = DIFFICEJointInversionConfig(
        idxgall=(0,),
        basal_mask=(False,),
        scales=(_scale(),),
        global_weights={"data": 0.0},
    )
    with pytest.warns(RuntimeWarning, match="Global weight for data is zero"):
        loss_fn = loss_joint_create((_pred, _grad), None, None, config)

    data = _real_data()
    residuals = loss_fn.kfac_residuals(params, data, terms=("data",))
    loss, info, _ = loss_fn(params, data)

    assert residuals.shape == (0, 1)
    assert jnp.allclose(info[1:5], jnp.zeros(4))
    assert jnp.allclose(loss, 0.0)
