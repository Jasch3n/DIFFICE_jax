from __future__ import annotations

import jax.numpy as jnp

from diffice_jax.data.xpinns.sampling import DataSample
from diffice_jax.model.xpinns.loss import (
    DIFFICEJointInversionConfig,
    DIFFICEXPINNRegressionConfig,
    loss_joint_create,
    loss_regression_create,
)


def _effective_gpinn_options(loss_config):
    if loss_config.use_gpinn:
        return loss_config.gpinn_weight, loss_config.global_weights
    global_weights = None if loss_config.global_weights is None else dict(loss_config.global_weights)
    if global_weights is not None:
        global_weights.pop("gpinn", None)
    return 0.0, global_weights


def add_dummy_inverse_targets(data):
    """Supply positive placeholder ``mu`` and ``C`` targets for real data.

    The shared XPINN regression loss computes viscosity and basal-friction data
    residuals even when their weights are zero. Real joint inversion has no
    ground-truth targets for those fields, so placeholders keep the old codepath
    numerically valid while the zero weights preserve the intended objective.
    """

    sample = data["smp"]
    if len(sample.Mu_smp) != 0 and len(sample.C_smp) != 0:
        return data
    mu_smp = [jnp.ones((x.shape[0], 1), dtype=x.dtype) for x in sample.X_smp]
    C_smp = [jnp.ones((x.shape[0], 1), dtype=x.dtype) for x in sample.X_smp]
    data = dict(data)
    data["smp"] = DataSample(
        sample.X_smp,
        sample.U_smp,
        sample.Xh_smp,
        sample.H_smp,
        sample.S_smp,
        mu_smp,
        C_smp,
        sample.Xs_smp,
    )
    return data


def dataf_with_dummy_inverse_targets(dataf):
    """Wrap a sampler so joint-inversion batches satisfy legacy loss shape needs."""

    def wrapped(key, eval_adaptive=None, eval_f=None):
        return add_dummy_inverse_targets(dataf(key, eval_adaptive=eval_adaptive, eval_f=eval_f))

    return wrapped


def loss_joint_inversion_xpinn(
    solNN,
    sub_region_indices,
    basal_mask,
    eqn=None,
    front_eqn=None,
    matching=True,
    calving_front=True,
    scales=None,
    match_weight=1.0,
    match_component_weights=None,
    gpinn_weight=0.0,
    mu_grad_weight=0.0,
    global_weights=None,
):
    """Build the real-data XPINN joint-inversion loss from shared loss terms."""

    config = DIFFICEJointInversionConfig(
        idxgall=tuple(sub_region_indices),
        basal_mask=tuple(basal_mask),
        match=matching,
        calving_front=calving_front,
        scales=None if scales is None else tuple(scales),
        match_weight=match_weight,
        match_component_weights=match_component_weights,
        gpinn_weight=gpinn_weight,
        mu_grad_weight=mu_grad_weight,
        global_weights=global_weights,
    )
    return loss_joint_create(solNN, eqn, front_eqn, config)


class JointInversionLossBuilder:
    __slots__ = ("loss_config",)

    def __init__(self, loss_config):
        self.loss_config = loss_config

    def create(self, solution, sub_region_indices, basal_mask, eqn, front_eqn, scales):
        gpinn_weight, global_weights = _effective_gpinn_options(self.loss_config)
        config = DIFFICEJointInversionConfig(
            idxgall=tuple(sub_region_indices),
            basal_mask=tuple(basal_mask),
            match=self.loss_config.matching,
            calving_front=self.loss_config.calving_front,
            scales=tuple(scales),
            match_weight=self.loss_config.match_weight,
            match_component_weights=self.loss_config.match_component_weights,
            gpinn_weight=gpinn_weight,
            mu_grad_weight=self.loss_config.mu_grad_weight,
            global_weights=global_weights,
            active_regions=None if self.loss_config.active_regions is None else tuple(self.loss_config.active_regions),
        )
        return loss_joint_create(solution, eqn, front_eqn, config)


class RegressionLossBuilder:
    __slots__ = ("loss_config", "data_config")

    def __init__(self, loss_config, data_config):
        self.loss_config = loss_config
        self.data_config = data_config

    def create(self, solution, sub_region_indices, basal_mask, eqn, front_eqn, scales):
        gpinn_weight, global_weights = _effective_gpinn_options(self.loss_config)
        config = DIFFICEXPINNRegressionConfig(
            idxgall=tuple(sub_region_indices),
            basal_mask=tuple(basal_mask),
            match=self.loss_config.matching,
            calving_front=self.loss_config.calving_front,
            scales=tuple(scales),
            match_weight=self.loss_config.match_weight,
            match_component_weights=self.loss_config.match_component_weights,
            gpinn_weight=gpinn_weight,
            mu_grad_weight=self.loss_config.mu_grad_weight,
            global_weights=global_weights,
            grounded_only_interface_mu_ct=self.data_config.grounded_only_interface_mu_ct,
            active_regions=None if self.loss_config.active_regions is None else tuple(self.loss_config.active_regions),
        )
        return loss_regression_create(solution, eqn, front_eqn, config)
