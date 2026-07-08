import jax
import jax.numpy as jnp
import warnings
from dataclasses import dataclass
from jax.tree_util import tree_map
from jax import lax
import jax.debug as jdb
from typing import List, Tuple, Callable, Any
from jax.typing import ArrayLike
from diffice_jax.data.xpinns.preprocessing import SubScaleResult

DEBUG = False

# define the mean squared error
def ms_error(diff):
    return jnp.sum(jnp.square(diff), axis=0) / jnp.maximum(diff.shape[0], 1.0)


# take the nth power root with original sign
def nthrt(x, n):
    return jnp.sign(x) * jnp.abs(x) ** (1/n)


def unpack_sub_scale(scale: SubScaleResult, basal:bool=False):
    # define the global parameter
    rho = scale.dynamic_scale.rho
    rho_w = scale.dynamic_scale.rho_w
    g = scale.dynamic_scale.g
    gd = g * (1 - rho / rho_w)  # reduced gravitational acceleration

    # load the scale information
    dmean, drange = scale.data_mean, scale.data_range
    lx0, ly0, u0, v0, _  = drange[0:5]
    _,   _,   um, vm, h0 = dmean[0:5]

    u0m = scale.dynamic_scale.u0
    l0m = scale.dynamic_scale.l0

    mu0 = scale.dynamic_scale.mu0
    du0 = u0m / l0m
    dh0 = h0 / l0m

    term0 = h0**2 / l0m
    return u0, v0, h0, mu0, du0, dh0, term0, um/u0, vm/v0


def u_mag(u):
    return jnp.sqrt(jnp.sum(jnp.square(u), 1))


@dataclass(frozen=True)
class DIFFICEJointInversionConfig:
    idxgall: Tuple[int, ...]
    basal_mask: Tuple[bool, ...]
    match: bool = False
    calving_front: bool | None = None
    scales: Tuple[SubScaleResult, ...] | None = None
    match_weight: float = 1.0
    match_component_weights: Any | None = None
    gpinn_weight: float = 0.0
    mu_grad_weight: float = 0.0
    active_regions: Tuple[int, ...] | None = None
    global_weights: dict[str, Any] | None = None


@dataclass(frozen=True)
class DIFFICEXPINNRegressionConfig:
    idxgall: Tuple[int, ...]
    basal_mask: Tuple[bool, ...]
    match: bool = False
    calving_front: bool | None = None
    scales: Tuple[SubScaleResult, ...] | None = None
    match_weight: float = 1.0
    match_component_weights: Any | None = None
    gpinn_weight: float = 0.0
    mu_grad_weight: float = 0.0
    grounded_only_interface_mu_ct: bool = False
    active_regions: Tuple[int, ...] | None = None
    global_weights: dict[str, Any] | None = None


class _XPINNLoss:
    __slots__ = (
        "_call",
        "lref",
        "match_weight",
        "eqn_region_weights",
        "gpinn_weight",
        "mu_grad_weight",
        "region_term_values",
        "kfac_eval",
        "kfac_residuals",
        "kfac_objective",
    )
    __hash__ = object.__hash__

    def __call__(self, params, data):
        return self._call(params, data)

#%% loss for inferring isotropic viscosity

def loss_iso_create(solNN, eqn_all, sub_scales:List[SubScaleResult], idxgall, lw, basal_mask=None, gamma_eq=None, use_regression=False):
    ''' a function factory to create the loss function for isotropic analysis
    :param solNN: neural network function for solutions and its derivative [tuple(callable, callable)]
    :param eqn_all: include governing equation and boundary equation of SSA [tuple(callable, callable)]
    :param gamma_eq: optional equation weight override for MSNN higher stages.
                     If provided, loss = (1-gamma_eq)*loss_data + gamma_eq*loss_eqn.
                     If None, uses the standard lw-based weighting.
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # create default basal mask if not provided (all floating)
    if basal_mask is None:
        ng = len(idxgall)
        basal_mask = [False] * ng

    # obtain the viscosity and strain rate scale in each sub-region
    all_info = jnp.array(tree_map(lambda x: unpack_sub_scale(sub_scales[x], basal=basal_mask[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.mean(scale_info, axis=0)   # To do: check whether jnp.min or jnp.mean better
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    um, vm = jnp.split(mean_nm, 2, axis=1)

    # create the loss constraint for each sub-regions
    def loss_sub(params, data, idx):
        is_basal = basal_mask[idx]

        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)

        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]
        s_smp = data['smp'][4][idx]

        if use_regression:
            mu_smp = data['smp'][5][idx]
            C_smp  = data['smp'][6][idx]

        # load the position of collocation points
        x_col = data['col'][0][idx]

        # load boundary data (only for floating subregions)
        if not is_basal:
            x_bd = data['bd'][0][idx]
            nn_bd = data['bd'][1][idx]

        # network predictions
        out = net(x_smp)
        u_pred = out[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        s_pred = net(xh_smp)[:, 3:4]
        if use_regression:
            mu_pred = out[:, 4:5]
            C_pred  = out[:, 5:6]

        # ================= CALCULATE LOSSES FOR SUBREGION =================
        # equation residual
        f_pred = gov_eqn(net, x_col, sub_scales[idx], basal=is_basal)[0]

        # data errors
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_s_err = ms_error(s_pred - s_smp)
        if use_regression:
            data_mu_err = ms_error(mu_pred - mu_smp)
            data_C_err = ms_error(C_pred - C_smp)

        if is_basal:
            data_err = jnp.hstack((data_u_err, data_h_err))
            s_err_weighted = data_s_err
            data_err_all = jnp.hstack([data_err, s_err_weighted])
            eqn_err = ms_error(f_pred) # (2,)
            bd_err = jnp.array([0.0, 0.0])  # (2,)
        else:
            data_err = jnp.hstack((data_u_err, data_h_err))  # (3,)
            data_err_all = jnp.hstack([data_err, 0.0])  # (4,)
            eqn_err = ms_error(f_pred)  # (2,)
            f_bd = front_eqn(net, x_bd, nn_bd, sub_scales[idx])[0]
            bd_err = ms_error(f_bd)  # (2,)

        if use_regression:
            data_err_all = jnp.hstack((data_err_all, data_mu_err, data_C_err))
        err_all = jnp.hstack([data_err_all, eqn_err, bd_err])
        return err_all

    # create the continuation loss constraint at the interface of adjacent subregions
    def loss_match(params, data, idx):
        # jdb.print('......Doing matching loss calculation for regionIdx={x}', x=idx)
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)

        # Need to pass basal flag to gov_eqn if we use it in matching (C2)
        # gov_eqn signature updated to accept basal=...
        # But here we invoke it.
        # Check basal status of idx and idx+1
        is_basal_1 = basal_mask[idx]
        is_basal_2 = basal_mask[idx+1]

        fgovterm = lambda x, id, b: gov_eqn(lambda z: net(z, id), x, sub_scales[id], basal=b)[1]

        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        """C0 stitching condition at the boundary"""
        u0m = lax.max(u0[idx], u0[idx+1])
        v0m = lax.max(v0[idx], v0[idx+1])
        h0m = lax.max(h0[idx], h0[idx+1])
        mu0m = lax.max(mu0[idx], mu0[idx+1])

        # obtain the variable in sub-region 1 at the interface
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx] / u0m
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx] / v0m
        h_md1 = (U_md1[:, 2:3]) * h0[idx] / h0m
        mu_md1 = U_md1[:, 4:5] * mu0[idx] / mu0m
        # jdb.print("mean u_md1: {x}", x=jnp.mean(u_md1))
        # jdb.print("mean v_md1: {x}", x=jnp.mean(v_md1))
        # jdb.print("mean h_md1: {x}", x=jnp.mean(h_md1))
        # jdb.print("mu_md1: {x}", x=mu_md1)

        # vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2 * jnp.log(mu_md1)])
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, mu_md1])

        # obtain the variable in sub-region 2 at the interface
        U_md2 = net(x_md[:, 2:4], idx + 1)
        u_md2 = (U_md2[:, 0:1] + um[idx + 1]) * u0[idx + 1] / u0m
        v_md2 = (U_md2[:, 1:2] + vm[idx + 1]) * v0[idx + 1] / v0m
        h_md2 = (U_md2[:, 2:3]) * h0[idx + 1] / h0m
        mu_md2 = U_md2[:, 4:5] * mu0[idx + 1] / mu0m
        # jdb.print("mean u_md2: {x}", x=jnp.mean(u_md2))
        # jdb.print("mean v_md2: {x}", x=jnp.mean(v_md2))
        # jdb.print("mean h_md2: {x}", x=jnp.mean(h_md2))
        # jdb.print("mu_md2: {x}", x=mu_md2)
        # jdb.print("mean mu_md2: {x}", x=jnp.mean(2 * jnp.log(mu_md2)))

        # vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2 * jnp.log(mu_md1)])
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, mu_md1])

        # vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2 * jnp.log(mu_md2)])
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, mu_md2])

        # [NEW]: Apply one-way constraint for all variables at the grounding line
        if is_basal_1 and not is_basal_2:
            # Region 1 is grounded, Region 2 is floating
            # We want Region 1 to conform to Region 2 without altering Region 2
            vars_md2 = lax.stop_gradient(vars_md2)

        elif not is_basal_1 and is_basal_2:
            # Region 1 is floating, Region 2 is grounded
            # We want Region 2 to conform to Region 1 without altering Region 1
            vars_md1 = lax.stop_gradient(vars_md1)

        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)
        # jdb.print("mean match_c0_err: {x}", x=jnp.where(jnp.isnan(vars_md1 - vars_md2), 0, 1))
        # jdb.print("mean match_c0_err: {x}", x=jnp.mean(match_c0_err))

        """C1 stitching condition at the boundary"""
        du0m = lax.max(du0[idx], du0[idx+1])
        dh0m = lax.max(dh0[idx], dh0[idx+1])

        # obtain the variable in sub-region 1 at the interface
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx] / du0m
        dh_md1 = dU_md1[:, 4:6] * dh0[idx] / dh0m
        if is_basal_1 != is_basal_2:
            dvars_md1 = jnp.hstack([duv_md1])
        else:
            dvars_md1 = jnp.hstack([duv_md1, dh_md1])

        # obtain the variable in sub-region 2 at the interface
        dU_md2 = gdnet(x_md[:, 2:4], idx + 1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx + 1] / du0m
        dh_md2 = dU_md2[:, 4:6] * dh0[idx + 1] / dh0m
        if is_basal_1 != is_basal_2:
            dvars_md2 = jnp.hstack([duv_md2])
        else:
            dvars_md2 = jnp.hstack([duv_md2, dh_md2])

        # [NEW]: Apply one-way constraint for all C1 variables at the grounding line
        # if is_basal_1 and not is_basal_2:
        #     dvars_md2 = lax.stop_gradient(dvars_md2)
        # elif not is_basal_1 and is_basal_2:
        #     dvars_md1 = lax.stop_gradient(dvars_md1)

        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        """C2 stitching condition at the boundary"""
        # calculate equation residue in sub-region 1 at the interface
        # gov_eqn returns (f_eqn, val_term). we want val_term.
        # match terms depending on interface type
        if is_basal_1 != is_basal_2:
            # C2 continuity is not satisfied at grounding lines
            match_c2_err = 0.0
        else:
            term_md1 = fgovterm(x_md[:, 0:2], idx, is_basal_1)[:, 0:-1] * term0[idx]
            term_md2 = fgovterm(x_md[:, 2:4], idx + 1, is_basal_2)[:, 0:-1] * term0[idx + 1]
            if basal_mask is not None:
                # calculate equation residue in sub-region 1 at the interface
                # gov_eqn returns (f_eqn, val_term). we want val_term.
                # Same-type interface with basal regions: match first 6 terms
                match_c2_err = ms_error(nthrt(term_md1[:, 0:6], 2) - nthrt(term_md2[:, 0:6], 2))
            else:
                # Ice shelf interface: match all terms
                match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitched conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        # jdb.print("mc0_err: {x} | mc1_err: {y} | mc2_err: {z}", x=mc0_err, y=mc1_err, z=mc2_err)
        # jdb.print("mc1_err: {x}", x=mc1_err)
        # jdb.print("mc2_err: {x}", x=mc2_err)
        # [NOTE]: Turn on/off the different continuity terms as needed
        # match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.2])
        match_err = jnp.hstack([mc0_err, mc1_err*0.7, mc2_err*0.2])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        reg_err_list = jnp.array(tree_map(lambda x: loss_sub(params, data, x), idxgall))

        reg_err = jnp.mean(reg_err_list, axis=0)
        # calculate the error at the matching boundary
        match_err_list = jnp.array(tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1]))
        match_err = jnp.mean(match_err_list, axis=0)
        # jdb.print('===============================================================================\n'
        #           '\t\tfloat errs: u={f1:.3e}  | v={f2:.3e}  | h={f3:.3e} | s=N/A\n'
        #           '\t\t            e1={f4:.3e} | e2={f5:.3e} | bd1={bd1:.3e} | bd2={bd2:.3e}\n'
        #           '\t\tgrnd  errs: u={g1:.3e}  | v={g2:.3e}  | h={g3:.3e} | s={g4:.3e}\n'
        #           '\t\t            e1={g5:.3e} | e2={g6:.3e} | bd1={bd3:.3e} | bd2={bd4:.3e}\n'
        #           '\t\tmatch errs: mc0={mc0:.3e} | mc1={mc1:.3e} | mc2={mc2:.3e}',
        #           f1=reg_err_list[0,0], f2=reg_err_list[0,1], f3=reg_err_list[0,2], f4=reg_err_list[0,4], f5=reg_err_list[0,5],
        #           bd1=reg_err_list[0,6], bd2=reg_err_list[0,7],
        #           g1=reg_err_list[1,0], g2=reg_err_list[1,1], g3=reg_err_list[1,2], g4=reg_err_list[1,3], g5=reg_err_list[1,4], g6=reg_err_list[1,5],
        #           bd3=reg_err_list[1,6], bd4=reg_err_list[1,7],
        #           mc0=match_err[0], mc1=match_err[1], mc2=match_err[2])

        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        # data_w (u, v, h, s) -> 1, 1, 0.6, 0.6
        data_w = jnp.array([1., 1., 0.6, 0.6])
        if use_regression:
            data_w = jnp.array([1., 1., 0.6, 0.6, 1., 1.])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        md_w = jnp.array([1., 1., 1.])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, md_w])

        # calculate the overall data loss and equation loss
        last_data_term = 6 if use_regression else 4
        loss_each = err_all * wgh_all
        loss_data = jnp.sum(loss_each[0:last_data_term])
        loss_eqn = jnp.sum(loss_each[last_data_term:last_data_term+2])
        loss_bd = jnp.sum(loss_each[last_data_term+2:last_data_term+4])
        loss_md = jnp.sum(loss_each[last_data_term+4:])
        # jdb.print("loss_data: {x}", x=loss_data)
        # jdb.print("loss_eqn: {x}", x=loss_eqn)
        # jdb.print("loss_bd: {x}", x=loss_bd)
        # jdb.print("loss_md: {x}", x=loss_md)
        # loading the pre-saved loss parameter
        loss_ref = loss_fun.lref
        # load the (possibly mutated) loss weights
        _lw = loss_fun.lw
        # calculate the total loss
        _gamma_eq = loss_fun.gamma_eq
        if _gamma_eq is not None:
            # MSNN mode: use gamma_eq to balance data vs equation loss
            loss = ((1.0 - _gamma_eq) * (loss_data + loss_bd + loss_md)
                    + _gamma_eq * loss_eqn)
        else:
            # Standard mode
            loss = (_lw[0] * loss_data + _lw[1] * loss_eqn + _lw[2] * loss_bd + _lw[3] * loss_md)
        # normalize the loss by the initial reference value
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md]), err_all])
        return loss_n, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lref = 1.0
    # store loss weights as mutable attribute (can be updated between stages)
    loss_fun.lw = jnp.array(lw)
    # store MSNN gamma_eq (None = standard mode)
    loss_fun.gamma_eq = gamma_eq

    return loss_fun


def _loss_xpinn_create(solNN:Tuple[Callable], idxgall:List[int],
                       basal_mask: List[bool]|None = None,
                       eqn:Callable = None, front_eqn:Callable = None,
                       match:bool = False,  calving_front:bool|None=None,
                       scales: List[SubScaleResult]|None = None,
                       match_weight: float = 1.0,
                       match_component_weights: ArrayLike|None = None,
                       gpinn_weight: float = 0.0,
                       mu_grad_weight: float = 0.0,
                       grounded_only_interface_mu_ct: bool = False,
                       active_regions: List[int]|None = None,
                       global_weights: dict[str, Any]|None = None,
                       include_inverse_data: bool = True):
    global_weights = {} if global_weights is None else dict(global_weights)

    def explicit_global_weight(names):
        return any(name in global_weights for name in names)

    def global_weight(names, default):
        for name in names:
            if name in global_weights:
                return jnp.array(global_weights[name])
        return jnp.array(default)

    def weight_is_zero(value):
        return float(jnp.asarray(value)) == 0.0

    def warn_zero_global(term, names, weight, enabled=True):
        if enabled and explicit_global_weight(names) and weight_is_zero(weight):
            warnings.warn(
                f"Global weight for {term} is zero; skipping {term} computation.",
                RuntimeWarning,
                stacklevel=2,
            )

    match_weight = global_weight(("matching", "match"), match_weight)
    gpinn_weight = global_weight(("gpinn",), gpinn_weight)
    mu_grad_weight = global_weight(("mu_gradient", "mu_grad"), mu_grad_weight)

    if not (eqn is None):
        assert not (scales is None), "[loss_regression_create] : Scales must be provided when using the equation in the loss function."
    if match and (not weight_is_zero(match_weight)):
        assert not (scales is None), "[loss_regression_create] : Scales must be provided when using the matching loss."
    if calving_front:
        assert not (front_eqn is None), "[loss_regression_create] : front_eqn must be provided when using the calving front loss."
        assert not (scales is None), "[loss_regression_create] : Scales must be provided when using the calving front loss."
    if not weight_is_zero(gpinn_weight):
        assert not (eqn is None), "[loss_regression_create] : eqn must be provided when using gPINN regularization."
        assert not (scales is None), "[loss_regression_create] : Scales must be provided when using gPINN regularization."
    if not weight_is_zero(mu_grad_weight):
        assert not (scales is None), "[loss_regression_create] : Scales must be provided when using mu gradient regularization."

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN

    # create default basal mask if not provided (all floating)
    if basal_mask is None:
        raise ValueError('No basal mask was supplied to the loss function')

    idxgall = tuple(int(idx) for idx in idxgall)
    basal_mask = tuple(bool(v) for v in basal_mask)
    n_sub = len(idxgall)
    loss_fun = _XPINNLoss()
    active_regions_static = None if active_regions is None else tuple(int(idx) for idx in active_regions)
    n_data_terms = 6.0 if include_inverse_data else 4.0
    n_eqn_terms = 2
    n_ct_terms = 1.0 if grounded_only_interface_mu_ct else 2.0
    n_gpinn_terms = 4.0
    n_mu_grad_terms = 2.0
    floating_idx = tuple(idx for idx in idxgall if not basal_mask[idx])
    n_floating = max(len(floating_idx), 1)
    data_w = jnp.array([1., 1., 1., 1., 0., 0.]) if include_inverse_data else jnp.array([1., 1., 1., 1.])
    eqn_w = jnp.array([1., 1.])
    # eqn_w *= 10.0
    ct_w = jnp.array([1.]) if grounded_only_interface_mu_ct else jnp.array([1., 1.])
    # ct_w *= 10.0
    # md_w order: u, v, h, s, log(mu), 
    #             u_x, u_y, v_x, v_y, h_x, h_y, (log(mu))_x, (log(mu))_y, 
    #             u_xx, 0.5*(u_xy+u_yx), u_yy, v_xx, 0.5*(v_xy+v_yx), v_yy.
    md_w_default = jnp.array([1., 1., 1., 1., 1.,
                              0.6, 0.6, 0.6, 0.6, 0.0, 0.0, 0.6, 0.6,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    md_w = md_w_default if match_component_weights is None else jnp.array(match_component_weights)
    data_global_weight = global_weight(("data",), 1.0)
    eqn_global_weight = global_weight(("equation", "eqn"), 0.01)
    ct_global_weight = global_weight(("calving_front", "ct"), 0.01)
    eqn_available = (not eqn is None) and (not scales is None)
    ct_available = (not front_eqn is None) and (not scales is None) and (calving_front is not False)
    use_data = not weight_is_zero(data_global_weight)
    use_eqn = eqn_available and (not weight_is_zero(eqn_global_weight))
    use_ct = ct_available and (not weight_is_zero(ct_global_weight))
    use_match = match and (not weight_is_zero(match_weight))
    use_gpinn = eqn_available and (not weight_is_zero(gpinn_weight))
    use_mu_grad = not weight_is_zero(mu_grad_weight)

    warn_zero_global("data", ("data",), data_global_weight)
    warn_zero_global("equation", ("equation", "eqn"), eqn_global_weight, enabled=eqn_available)
    warn_zero_global("calving-front", ("calving_front", "ct"), ct_global_weight, enabled=ct_available)
    warn_zero_global("matching", ("matching", "match"), match_weight, enabled=match)
    warn_zero_global("gPINN", ("gpinn",), gpinn_weight, enabled=eqn_available)
    warn_zero_global("mu-gradient", ("mu_gradient", "mu_grad"), mu_grad_weight)

    def current_match_weight(data):
        return data.get('match_weight', loss_fun.match_weight)

    def current_eqn_weights(data):
        return data.get('eqn_region_weights', loss_fun.eqn_region_weights)

    def current_eqn_weight(data, idx):
        idx_pos = idxgall.index(idx)
        return current_eqn_weights(data)[idx_pos]

    def current_region_term_weights(data):
        return data.get('region_term_weights', None)

    def use_region_term_weights(data):
        return current_region_term_weights(data) is not None

    def current_region_term_weight(data, term, idx):
        region_term_weights = current_region_term_weights(data)
        if region_term_weights is None:
            return None
        return region_term_weights[term][idxgall.index(idx)]

    def current_match_interface_weight(data, idx):
        region_term_weights = current_region_term_weights(data)
        if region_term_weights is None:
            return None
        left = region_term_weights['match'][idxgall.index(idx)]
        right = region_term_weights['match'][idxgall.index(idx + 1)]
        return 0.5 * (left + right)

    def active_terms(terms):
        return ('data', 'eqn', 'ct', 'match', 'gpinn', 'mu_grad') if terms is None else tuple(terms)

    def active_regions(regions):
        return tuple(idxgall) if regions is None else tuple(regions)

    def current_active_regions(data):
        if active_regions_static is not None:
            return jnp.array(active_regions_static)
        regions = data.get('active_regions', None)
        if regions is None:
            return jnp.array(idxgall)
        return jnp.asarray(regions)

    def active_region_tuple(data):
        if active_regions_static is not None:
            return active_regions_static
        regions = data.get('active_regions', None)
        if regions is None:
            return tuple(idxgall)
        if isinstance(regions, (list, tuple)):
            return tuple(int(idx) for idx in regions)
        return tuple(idxgall)

    def active_region_count(data):
        return max(len(active_region_tuple(data)), 1)

    def region_mask(data):
        active = current_active_regions(data)
        idx_array = jnp.array(idxgall)
        mask = jnp.any(idx_array[:, None] == active[None, :], axis=1)
        return mask.astype(jnp.float32).reshape((n_sub, 1))

    def masked_region_mean(values, data):
        mask = region_mask(data)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(values * mask, axis=0) / denom

    def include_region(idx, regions):
        return idx in regions

    def include_match(idx, regions):
        return (idx in regions) or ((idx + 1) in regions)

    def pred_point(params, idx):
        return lambda z: predNN(params, z, idx)

    def pred_batch(params, idx):
        return lambda X: jax.vmap(lambda z: predNN(params, z, idx), in_axes=(0,))(X)

    def grad_batch(params, idx):
        return lambda X: jax.vmap(lambda z: gradNN(params, z, idx), in_axes=(0,))(X)

    def ct_data(data):
        return data['ct'] if 'ct' in data else data['bd']

    # create the loss constraint for each sub-regions
    def loss_data_sub(params, data, idx):
        is_basal = basal_mask[idx]

        # create the function for gradient calculation involves input Z only
        net = pred_batch(params, idx)

        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]
        s_smp = data['smp'][4][idx]

        # network predictions
        out = net(x_smp)
        out_h = net(xh_smp)
        u_pred = out[:, 0:2]
        h_pred = out_h[:, 2:3]
        s_pred = out_h[:, 3:4]
        mu_pred = out[:, 4:5]
        C_pred = out[:, 5:6]

        # data errors
        u_target = u_smp
        data_u_err = ms_error(u_pred - u_target)
        data_h_err = ms_error(h_pred - h_smp)
        data_s_err = ms_error(s_pred - s_smp)
        if include_inverse_data:
            mu_smp = data['smp'][5][idx]
            C_smp  = data['smp'][6][idx]
            data_mu_err = ms_error(jnp.log(mu_pred) - jnp.log(mu_smp))
            data_C_err = ms_error(C_pred - C_smp) if is_basal else jnp.zeros_like(data_mu_err)
            data_err = jnp.hstack((data_u_err, data_h_err,
                                   data_s_err,
                                   data_mu_err,
                                   data_C_err))
        else:
            data_err = jnp.hstack((data_u_err, data_h_err,
                                   data_s_err))
        if DEBUG:
            # jdb.print('[DEBUG] u_pred shape = {s}', s=u_pred.shape)
            # jdb.print('[DEBUG] h_pred shape = {s}', s=h_pred.shape)
            # jdb.print('[DEBUG] s_pred shape = {s}', s=s_pred.shape)
            # jdb.print('[DEBUG] mu_pred shape = {s}', s=mu_pred.shape)
            # jdb.print('[DEBUG] C_pred shape = {s}', s=C_pred.shape)
            jdb.print('[DEBUG]: Region = {mm}, data_u_err={x1} | data_h_err={x2} | data_s_err={x3} | data_mu_err={x4} | data_C_err={x5}',
                       mm=idx, x1=data_u_err, x2=data_h_err, x3=data_s_err, x4=data_mu_err, x5=data_C_err)
            jdb.print('[DEBUG]: Region = {mm}, data_err shape={s}', mm=idx, s=data_err.shape)
            jdb.print('[DEBUG]: data_err shape = {s}', s=data_err.shape)
            jdb.print('[DEBUG]: Region {i} C err = {e}', i=idx, e=data_C_err)
        # jdb.print('[DEBUG]: data_err shape = {s}', s=data_err.shape)
        return data_err

    def eqn_res(params, x, idx):
        net = pred_point(params, idx)
        return eqn(net, x, scales[idx], basal=basal_mask[idx])[0]

    def loss_eqn_sub(params, data, idx):
        # load the velocity data and their position
        x_col = data['col'][0][idx]

        # network predictions
        f = jax.vmap(lambda x: eqn_res(params, x, idx), in_axes=(0,))(x_col)

        eqn_err = ms_error(f)

        return eqn_err

    def loss_ct_sub(params, data, idx):
        ct = ct_data(data)
        X_ct = ct[0][idx]
        ct_target = ct[1][idx]
        if X_ct.shape[0] == 0:
            return jnp.zeros(int(n_ct_terms))

        if grounded_only_interface_mu_ct and basal_mask[idx]:
            net = pred_batch(params, idx)
            mu_pred = net(X_ct)[:, 4:5]
            f_bd = jnp.log(mu_pred) - jnp.log(ct_target)
        elif grounded_only_interface_mu_ct:
            f_bd = jnp.zeros((1, 1))
        elif basal_mask[idx]:
            f_bd = jnp.zeros((1, int(n_ct_terms)))
        else:
            net = pred_point(params, idx)
            bd_fn = lambda x, nn: front_eqn(net, x, nn, scales[idx])[0]
            f_bd = jax.vmap(bd_fn, in_axes=(0, 0))(X_ct, ct_target)
        f_bd_err = ms_error(f_bd)
        # jdb.print('[DEBUG]: region{idx}, f_bd shape = {s}', idx=idx, s=f_bd.shape)
        # jdb.print('[DEBUG]: region{idx}, f_bd_err shape = {s}', idx=idx, s=f_bd_err.shape)
        # jdb.print('[DEBUG]: region{idx}, X_ct shape = {s}', idx=idx, s=X_ct.shape)

        return f_bd_err

    def loss_data_res_sub(params, data, idx):
        net = pred_batch(params, idx)
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]
        s_smp = data['smp'][4][idx]
        out = net(x_smp)
        out_h = net(xh_smp)
        global_weight = current_region_term_weight(data, 'data', idx)
        if global_weight is None:
            global_weight = data_global_weight
        n_active = active_region_count(data)
        smp_weight = jnp.sqrt(global_weight * data_w / (n_data_terms * n_active * x_smp.shape[0]))
        h_weight = jnp.sqrt(global_weight * data_w / (n_data_terms * n_active * xh_smp.shape[0]))
        u_target = u_smp
        res = [smp_weight[0:2] * (out[:, 0:2] - u_target)]
        res += [
            h_weight[2:3] * (out_h[:, 2:3] - h_smp),
            h_weight[3:4] * (out_h[:, 3:4] - s_smp),
        ]
        if include_inverse_data:
            mu_smp = data['smp'][5][idx]
            C_smp = data['smp'][6][idx]
            res += [
                smp_weight[4:5] * (jnp.log(out[:, 4:5]) - jnp.log(mu_smp)),
                smp_weight[5:6] * (out[:, 5:6] - C_smp),
            ]
        return jnp.concatenate([r.reshape(-1) for r in res])

    def loss_eqn_res_sub(params, data, idx):
        x_col = data['col'][0][idx]
        f = jax.vmap(lambda x: eqn_res(params, x, idx), in_axes=(0,))(x_col)
        eqn_weight = current_eqn_weight(data, idx)
        global_weight = current_region_term_weight(data, 'eqn', idx)
        if global_weight is None:
            global_weight = eqn_global_weight * eqn_weight
        return (jnp.sqrt(global_weight * eqn_w / (n_eqn_terms * active_region_count(data) * x_col.shape[0])) * f).reshape(-1)

    def gpinn_collocation(data, idx):
        if 'col' in data:
            return data['col'][0][idx]
        return data['gpinn_col'][0][idxgall.index(idx)]

    def gpinn_res(params, data, idx):
        x_col = gpinn_collocation(data, idx)
        if x_col.shape[0] == 0:
            return jnp.zeros((0, int(n_gpinn_terms)))
        jac = jax.vmap(jax.jacfwd(lambda x: eqn_res(params, x, idx)))(x_col)
        return jac.reshape(x_col.shape[0], -1)

    def loss_gpinn_sub(params, data, idx):
        return ms_error(gpinn_res(params, data, idx))

    def loss_gpinn_res_sub(params, data, idx):
        x_col = gpinn_collocation(data, idx)
        if x_col.shape[0] == 0:
            return jnp.zeros((0,))
        global_weight = current_region_term_weight(data, 'gpinn', idx)
        if global_weight is None:
            global_weight = loss_fun.gpinn_weight
        weight = jnp.sqrt(global_weight / (n_gpinn_terms * active_region_count(data) * x_col.shape[0]))
        return (weight * gpinn_res(params, data, idx)).reshape(-1)

    def mu_grad_res(params, data, idx):
        x_col = data['col'][0][idx]
        grad = grad_batch(params, idx)(x_col)[:, 7:]
        mux_n = grad[:, 8:9]
        muy_n = grad[:, 9:10]
        x_range, y_range = scales[idx].data_range[0:2]
        l0 = scales[idx].dynamic_scale.l0
        return jnp.hstack((mux_n * l0 / x_range, muy_n * l0 / y_range))

    def loss_mu_grad_sub(params, data, idx):
        if basal_mask[idx]:
            return jnp.zeros(2)
        return ms_error(mu_grad_res(params, data, idx))

    def loss_mu_grad_res_sub(params, data, idx):
        if basal_mask[idx]:
            return jnp.zeros((0,))
        x_col = data['col'][0][idx]
        global_weight = current_region_term_weight(data, 'mu_grad', idx)
        if global_weight is None:
            global_weight = loss_fun.mu_grad_weight
        weight = jnp.sqrt(global_weight / (n_mu_grad_terms * n_floating * x_col.shape[0]))
        return (weight * mu_grad_res(params, data, idx)).reshape(-1)

    def loss_ct_res_sub(params, data, idx):
        if basal_mask[idx] and not grounded_only_interface_mu_ct:
            return jnp.zeros((0,))
        ct = ct_data(data)
        X_ct = ct[0][idx]
        ct_target = ct[1][idx]
        if X_ct.shape[0] == 0:
            return jnp.zeros((0,))
        if grounded_only_interface_mu_ct and basal_mask[idx]:
            net = pred_batch(params, idx)
            f_bd = jnp.log(net(X_ct)[:, 4:5]) - jnp.log(ct_target)
        else:
            net = pred_point(params, idx)
            bd_fn = lambda x, nn: front_eqn(net, x, nn, scales[idx])[0]
            f_bd = jax.vmap(bd_fn, in_axes=(0, 0))(X_ct, ct_target)
        global_weight = current_region_term_weight(data, 'ct', idx)
        if global_weight is None:
            global_weight = ct_global_weight
        return (jnp.sqrt(global_weight * ct_w / (n_ct_terms * active_region_count(data) * X_ct.shape[0])) * f_bd).reshape(-1)

    def loss_md_res_sub(params, data, idx):
        """
            Computes the matching loss between the idx-th region and the (idx+1)-th region.
        """
        is_basal_1 = basal_mask[idx]
        is_basal_2 = basal_mask[idx+1]
        Xmd = data['md'][0][idx]
        if Xmd.shape[1] == 4:
            Xmd_1 = Xmd[:, 0:2]
            Xmd_2 = Xmd[:, 2:4]
        else:
            Xmd_1 = Xmd
            Xmd_2 = data['md'][0][idx+1]

        scale_1 = scales[idx]
        scale_2 = scales[idx+1]

        net_1 = pred_batch(params, idx)
        net_2 = pred_batch(params, idx+1)

        grad_1 = grad_batch(params, idx)
        grad_2 = grad_batch(params, idx+1)
        grad2_1 = lambda X: jax.vmap(
            lambda z: jax.jacfwd(lambda zz: gradNN(params, zz, idx)[0:4])(z)
        )(X)
        grad2_2 = lambda X: jax.vmap(
            lambda z: jax.jacfwd(lambda zz: gradNN(params, zz, idx + 1)[0:4])(z)
        )(X)

        def dimensionalize(U:ArrayLike, grad_U:ArrayLike, grad2_U:ArrayLike, region_scale:SubScaleResult) -> ArrayLike:
            data_mean, data_range, dynamic_scale = region_scale
            _, _, u_mean, v_mean, h_mean, s_mean = data_mean
            x_range, y_range, u_range, v_range, _, s_range = data_range
            l0, u0, mu_scale, _, _, _, _, rho, rho_w, g = dynamic_scale
            ru0 = u_range / u0
            rv0 = v_range / u0
            rx0 = x_range / l0
            ry0 = y_range / l0

            # read off the dimensionless normalized variables
            u_n = U[:, 0:1]
            v_n = U[:, 1:2]
            h_n = U[:, 2:3]
            s_n = U[:, 3:4]
            mu_n = U[:, 4:5]

            # jdb.print('[DEBUG]: Total grad_U shape = {s}', s=grad_U.shape)
            grad_U = grad_U[:, 7:]
            ux_n = grad_U[:, 0:1]
            uy_n = grad_U[:, 1:2]
            vx_n = grad_U[:, 2:3]
            vy_n = grad_U[:, 3:4]
            hx_n = grad_U[:, 4:5]
            hy_n = grad_U[:, 5:6]
            sx_n = grad_U[:, 6:7]
            sy_n = grad_U[:, 7:8]
            mux_n = grad_U[:, 8:9]
            muy_n = grad_U[:, 9:10]

            # Dimensionalize the variables back to SI units
            u = u_n*u_range + u_mean
            v = v_n*v_range + v_mean
            h = h_n * h_mean
            s = s_n*s_range + s_mean
            mu = mu_n*mu_scale

            ux = ux_n * (u_range / x_range)
            uy = uy_n * (u_range / y_range)
            vx = vx_n * (v_range / x_range)
            vy = vy_n * (v_range / y_range)
            hx = hx_n * (h_mean / x_range)
            hy = hy_n * (h_mean / y_range)
            sx = sx_n * (s_range / x_range)
            sy = sy_n * (s_range / y_range)
            mux = mux_n * (mu_scale / x_range)
            muy = muy_n * (mu_scale / y_range)

            uxx = grad2_U[:, 0, 0:1] * (u_range / (x_range ** 2))
            uxy = grad2_U[:, 0, 1:2] * (u_range / (x_range * y_range))
            uyx = grad2_U[:, 1, 0:1] * (u_range / (x_range * y_range))
            uyy = grad2_U[:, 1, 1:2] * (u_range / (y_range ** 2))
            vxx = grad2_U[:, 2, 0:1] * (v_range / (x_range ** 2))
            vxy = grad2_U[:, 2, 1:2] * (v_range / (x_range * y_range))
            vyx = grad2_U[:, 3, 0:1] * (v_range / (x_range * y_range))
            vyy = grad2_U[:, 3, 1:2] * (v_range / (y_range ** 2))

            U_dim = jnp.hstack((u, v, h, s, mu))
            grad_U_dim = jnp.hstack((ux, uy, vx, vy, hx, hy, sx, sy, mux, muy))
            grad2_U_dim = jnp.hstack((uxx, uxy, uyx, uyy, vxx, vxy, vyx, vyy))
            return U_dim, grad_U_dim, grad2_U_dim

        def renormalize(U:ArrayLike, grad_U:ArrayLike, grad2_U:ArrayLike, scale_1:SubScaleResult, scale_2:SubScaleResult):
            combined_mean = (jnp.array(scale_1.data_mean) + jnp.array(scale_2.data_mean)) / 2.0
            combined_range = (jnp.array(scale_1.data_range) + jnp.array(scale_2.data_range)) / 2.0
            combined_dynamic_scale = (jnp.array(scale_1.dynamic_scale) + jnp.array(scale_2.dynamic_scale)) / 2.0

            _, _, u_mean, v_mean, h_mean, s_mean = combined_mean
            x_range, y_range, u_range, v_range, _, s_range = combined_range
            _, _, mu_scale, _, _, _, _, rho, rho_w, g = combined_dynamic_scale

            u = U[:, 0:1]
            v = U[:, 1:2]
            h = U[:, 2:3]
            s = U[:, 3:4]
            mu = U[:, 4:5]

            ux = grad_U[:, 0:1]
            uy = grad_U[:, 1:2]
            vx = grad_U[:, 2:3]
            vy = grad_U[:, 3:4]
            hx = grad_U[:, 4:5]
            hy = grad_U[:, 5:6]
            sx = grad_U[:, 6:7]
            sy = grad_U[:, 7:8]
            mux = grad_U[:, 8:9]
            muy = grad_U[:, 9:10]

            u_rn = (u - u_mean) / u_range
            v_rn = (v - v_mean) / v_range
            h_rn = h / h_mean
            s_rn = (s - s_mean) / s_range
            mu_rn = mu / mu_scale

            # rn stands for "renormalized"
            ux_rn = ux / (u_range / x_range)
            uy_rn = uy / (u_range / y_range)
            vx_rn = vx / (v_range / x_range)
            vy_rn = vy / (v_range / y_range)
            hx_rn = hx / (h_mean / x_range)
            hy_rn = hy / (h_mean / y_range)
            sx_rn = sx / (s_range / x_range)
            sy_rn = sy / (s_range / y_range)
            mux_rn = mux / (mu_scale / x_range)
            muy_rn = muy / (mu_scale / y_range)

            uxx_rn = grad2_U[:, 0:1] / (u_range / (x_range ** 2))
            uxy_rn = grad2_U[:, 1:2] / (u_range / (x_range * y_range))
            uyx_rn = grad2_U[:, 2:3] / (u_range / (x_range * y_range))
            uyy_rn = grad2_U[:, 3:4] / (u_range / (y_range ** 2))
            vxx_rn = grad2_U[:, 4:5] / (v_range / (x_range ** 2))
            vxy_rn = grad2_U[:, 5:6] / (v_range / (x_range * y_range))
            vyx_rn = grad2_U[:, 6:7] / (v_range / (x_range * y_range))
            vyy_rn = grad2_U[:, 7:8] / (v_range / (y_range ** 2))

            U_rn = jnp.hstack((u_rn, v_rn, h_rn, s_rn, mu_rn))
            grad_U_rn = jnp.hstack((ux_rn, uy_rn, vx_rn, vy_rn, hx_rn, hy_rn, sx_rn, sy_rn, mux_rn, muy_rn))
            grad2_U_rn = jnp.hstack((uxx_rn, uxy_rn, uyx_rn, uyy_rn, vxx_rn, vxy_rn, vyx_rn, vyy_rn))
            return U_rn, grad_U_rn, grad2_U_rn

        U1_phys, gradU1_phys, grad2U1_phys = dimensionalize(net_1(Xmd_1), grad_1(Xmd_1), grad2_1(Xmd_1), scale_1)
        U2_phys, gradU2_phys, grad2U2_phys = dimensionalize(net_2(Xmd_2), grad_2(Xmd_2), grad2_2(Xmd_2), scale_2)

        U1, gradU1, grad2U1 = renormalize(U1_phys, gradU1_phys, grad2U1_phys, scale_1, scale_2)
        U2, gradU2, grad2U2 = renormalize(U2_phys, gradU2_phys, grad2U2_phys, scale_1, scale_2)

        uvhs_1 = U1[:, 0:4]
        uvhs_2 = U2[:, 0:4]
        mu_1 = U1[:, 4:5]
        mu_2 = U2[:, 4:5]

        grad_uvhs_1 = gradU1[:, 0:6]
        grad_uvhs_2 = gradU2[:, 0:6]
        grad_mu_1 = gradU1[:, 8:10] / mu_1
        grad_mu_2 = gradU2[:, 8:10] / mu_2
        grad_str_1 = jnp.hstack((
            grad2U1[:, 0:1],
            0.5 * (grad2U1[:, 1:2] + grad2U1[:, 2:3]),
            grad2U1[:, 3:4],
            grad2U1[:, 4:5],
            0.5 * (grad2U1[:, 5:6] + grad2U1[:, 6:7]),
            grad2U1[:, 7:8],
        ))
        grad_str_2 = jnp.hstack((
            grad2U2[:, 0:1],
            0.5 * (grad2U2[:, 1:2] + grad2U2[:, 2:3]),
            grad2U2[:, 3:4],
            grad2U2[:, 4:5],
            0.5 * (grad2U2[:, 5:6] + grad2U2[:, 6:7]),
            grad2U2[:, 7:8],
        ))
        gradU1 = jnp.hstack([grad_uvhs_1, grad_mu_1])
        gradU2 = jnp.hstack([grad_uvhs_2, grad_mu_2])

        # if (not is_basal_1) and is_basal_2:
        #     mu_1 = lax.stop_gradient(mu_1)
        #     uvhs_1 = lax.stop_gradient(uvhs_1)
        #     grad_uvhs_1 = lax.stop_gradient(grad_uvhs_1)
        #     grad_mu_1 = lax.stop_gradient(grad_mu_1)
        # elif is_basal_1 and (not is_basal_2):
        #     mu_2 = lax.stop_gradient(mu_2)
        #     uvhs_2 = lax.stop_gradient(uvhs_2)
        #     grad_uvhs_2 = lax.stop_gradient(grad_uvhs_2)
        #     grad_mu_2 = lax.stop_gradient(grad_mu_2)

        C0_res = jnp.hstack((uvhs_1 - uvhs_2,
                             jnp.log(mu_1) - jnp.log(mu_2)))
        C1_res = jnp.hstack((grad_uvhs_1 - grad_uvhs_2,
                             grad_mu_1 - grad_mu_2,
                             grad_str_1 - grad_str_2))

        # jdb.print('gradU1 shape = {s}', s=gradU1.shape)
        # jdb.print('gradU2 shape = {s}', s=gradU2.shape)
        # jdb.print('C0 mismatch shape = {s}', s=C0_mismatch.shape)
        # jdb.print('C1 mismatch shape = {s}', s=C1_mismatch.shape)

        # Aggregate error across different variables
        # C0_err = jnp.mean(C0_mismatch)
        # C1_err = jnp.mean(C1_mismatch)

        # mismatch = C0_err + 0.7 * C1_err
        return C0_res, C1_res

    def loss_md_sub(params, data, idx):
        C0_res, C1_res = loss_md_res_sub(params, data, idx)
        return jnp.hstack((ms_error(C0_res), ms_error(C1_res)))

    def loss_md_kfac_res_sub(params, data, idx):
        C0_res, C1_res = loss_md_res_sub(params, data, idx)
        n_match = len(idxgall) - 1
        match_global_weight = current_match_weight(data)
        interface_scale = current_match_interface_weight(data, idx)
        if interface_scale is not None:
            match_global_weight = interface_scale
        n_c0 = C0_res.shape[1]
        n_c1 = C1_res.shape[1]
        c0_weight = jnp.sqrt(match_global_weight * md_w[:n_c0] / (md_w.shape[0] * n_match * C0_res.shape[0]))
        c1_weight = jnp.sqrt(match_global_weight * md_w[n_c0:n_c0+n_c1] / (md_w.shape[0] * n_match * C1_res.shape[0]))
        return jnp.concatenate([
            (c0_weight * C0_res).reshape(-1),
            (c1_weight * C1_res).reshape(-1),
        ])

    def data_block(params, data, idx):
        is_basal = basal_mask[idx]
        net = pred_batch(params, idx)
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]
        s_smp = data['smp'][4][idx]
        out = net(x_smp)
        out_h = net(xh_smp)

        data_u_err = ms_error(out[:, 0:2] - u_smp)
        data_h_err = ms_error(out_h[:, 2:3] - h_smp)
        data_s_err = ms_error(out_h[:, 3:4] - s_smp)

        global_weight = current_region_term_weight(data, 'data', idx)
        if global_weight is None:
            global_weight = data_global_weight
        n_active = active_region_count(data)
        smp_weight = jnp.sqrt(global_weight * data_w / (n_data_terms * n_active * x_smp.shape[0]))
        h_weight = jnp.sqrt(global_weight * data_w / (n_data_terms * n_active * xh_smp.shape[0]))
        res = [
            smp_weight[0:2] * (out[:, 0:2] - u_smp),
            h_weight[2:3] * (out_h[:, 2:3] - h_smp),
            h_weight[3:4] * (out_h[:, 3:4] - s_smp),
        ]

        if include_inverse_data:
            mu_smp = data['smp'][5][idx]
            C_smp = data['smp'][6][idx]
            data_mu_err = ms_error(jnp.log(out[:, 4:5]) - jnp.log(mu_smp))
            data_C_err = ms_error(out[:, 5:6] - C_smp) if is_basal else jnp.zeros_like(data_mu_err)
            err = jnp.hstack((data_u_err, data_h_err, data_s_err, data_mu_err, data_C_err))
            res += [
                smp_weight[4:5] * (jnp.log(out[:, 4:5]) - jnp.log(mu_smp)),
                smp_weight[5:6] * (out[:, 5:6] - C_smp),
            ]
        else:
            err = jnp.hstack((data_u_err, data_h_err, data_s_err))
        return err, jnp.concatenate([r.reshape(-1) for r in res])

    def eqn_block(params, data, idx):
        x_col = data['col'][0][idx]
        f = jax.vmap(lambda x: eqn_res(params, x, idx), in_axes=(0,))(x_col)
        err = ms_error(f)
        eqn_weight = current_eqn_weight(data, idx)
        global_weight = current_region_term_weight(data, 'eqn', idx)
        if global_weight is None:
            global_weight = eqn_global_weight * eqn_weight
        res = (jnp.sqrt(global_weight * eqn_w / (n_eqn_terms * active_region_count(data) * x_col.shape[0])) * f).reshape(-1)
        return err, res

    def ct_block(params, data, idx):
        if basal_mask[idx] and not grounded_only_interface_mu_ct:
            return jnp.zeros(int(n_ct_terms)), jnp.zeros((0,))
        ct = ct_data(data)
        X_ct = ct[0][idx]
        ct_target = ct[1][idx]
        if X_ct.shape[0] == 0:
            return jnp.zeros(int(n_ct_terms)), jnp.zeros((0,))
        if grounded_only_interface_mu_ct and basal_mask[idx]:
            net = pred_batch(params, idx)
            f_bd = jnp.log(net(X_ct)[:, 4:5]) - jnp.log(ct_target)
        elif grounded_only_interface_mu_ct:
            f_bd = jnp.zeros((1, 1))
        else:
            net = pred_point(params, idx)
            bd_fn = lambda x, nn: front_eqn(net, x, nn, scales[idx])[0]
            f_bd = jax.vmap(bd_fn, in_axes=(0, 0))(X_ct, ct_target)
        err = ms_error(f_bd)
        global_weight = current_region_term_weight(data, 'ct', idx)
        if global_weight is None:
            global_weight = ct_global_weight
        res = (jnp.sqrt(global_weight * ct_w / (n_ct_terms * active_region_count(data) * X_ct.shape[0])) * f_bd).reshape(-1)
        return err, res

    def match_block(params, data, idx):
        C0_res, C1_res = loss_md_res_sub(params, data, idx)
        err = jnp.hstack((ms_error(C0_res), ms_error(C1_res)))
        n_match = len(idxgall) - 1
        match_global_weight = current_match_weight(data)
        interface_scale = current_match_interface_weight(data, idx)
        if interface_scale is not None:
            match_global_weight = interface_scale
        n_c0 = C0_res.shape[1]
        n_c1 = C1_res.shape[1]
        c0_weight = jnp.sqrt(match_global_weight * md_w[:n_c0] / (md_w.shape[0] * n_match * C0_res.shape[0]))
        c1_weight = jnp.sqrt(match_global_weight * md_w[n_c0:n_c0+n_c1] / (md_w.shape[0] * n_match * C1_res.shape[0]))
        res = jnp.concatenate([
            (c0_weight * C0_res).reshape(-1),
            (c1_weight * C1_res).reshape(-1),
        ])
        return err, res

    def gpinn_block(params, data, idx):
        raw = gpinn_res(params, data, idx)
        err = ms_error(raw)
        x_col = gpinn_collocation(data, idx)
        if x_col.shape[0] == 0:
            return err, jnp.zeros((0,))
        global_weight = current_region_term_weight(data, 'gpinn', idx)
        if global_weight is None:
            global_weight = loss_fun.gpinn_weight
        weight = jnp.sqrt(global_weight / (n_gpinn_terms * active_region_count(data) * x_col.shape[0]))
        return err, (weight * raw).reshape(-1)

    def mu_grad_block(params, data, idx):
        if basal_mask[idx]:
            return jnp.zeros(2), jnp.zeros((0,))
        raw = mu_grad_res(params, data, idx)
        err = ms_error(raw)
        x_col = data['col'][0][idx]
        global_weight = current_region_term_weight(data, 'mu_grad', idx)
        if global_weight is None:
            global_weight = loss_fun.mu_grad_weight
        weight = jnp.sqrt(global_weight / (n_mu_grad_terms * n_floating * x_col.shape[0]))
        return err, (weight * raw).reshape(-1)

    def region_term_values(params, data):
        active = active_region_tuple(data)
        if use_data:
            data_err_list = jnp.array([
                loss_data_sub(params, data, idx) if idx in active else jnp.zeros(int(n_data_terms))
                for idx in idxgall
            ])
            data_region = jnp.mean(data_w * data_err_list, axis=1)
        else:
            data_region = jnp.zeros(n_sub)

        if use_eqn:
            eqn_err_list = jnp.array([
                loss_eqn_sub(params, data, idx) if idx in active else jnp.zeros(n_eqn_terms)
                for idx in idxgall
            ])
            eqn_region = jnp.mean(eqn_w * eqn_err_list, axis=1)
        else:
            eqn_region = jnp.zeros(n_sub)

        if use_ct:
            ct_err_list = jnp.array([
                loss_ct_sub(params, data, idx) if idx in active else jnp.zeros(int(n_ct_terms))
                for idx in idxgall
            ])
            ct_region = jnp.mean(ct_w * ct_err_list, axis=1)
        else:
            ct_region = jnp.zeros(n_sub)

        if use_match:
            md_err_list = jnp.array(tree_map(lambda x: loss_md_sub(params, data, x), idxgall[:-1]))
            md_interface = jnp.mean(md_w * md_err_list, axis=1)
            md_region_sum = jnp.zeros(n_sub)
            md_region_count = jnp.zeros(n_sub)
            for pos, idx in enumerate(idxgall[:-1]):
                left = idxgall.index(idx)
                right = idxgall.index(idx + 1)
                md_region_sum = md_region_sum.at[left].add(md_interface[pos])
                md_region_sum = md_region_sum.at[right].add(md_interface[pos])
                md_region_count = md_region_count.at[left].add(1.0)
                md_region_count = md_region_count.at[right].add(1.0)
            md_region = jnp.where(md_region_count > 0.0, md_region_sum / md_region_count, 0.0)
        else:
            md_region = jnp.zeros(n_sub)

        if use_gpinn:
            gpinn_err_list = jnp.array([
                loss_gpinn_sub(params, data, idx) if idx in active else jnp.zeros(4)
                for idx in idxgall
            ])
            gpinn_region = jnp.mean(gpinn_err_list, axis=1)
        else:
            gpinn_region = jnp.zeros(n_sub)

        if use_mu_grad:
            mu_grad_err_list = jnp.array([
                loss_mu_grad_sub(params, data, idx) if idx in active else jnp.zeros(2)
                for idx in idxgall
            ])
            mu_grad_region = jnp.mean(mu_grad_err_list, axis=1)
        else:
            mu_grad_region = jnp.zeros(n_sub)

        return dict(
            data=data_region,
            eqn=eqn_region,
            ct=ct_region,
            match=md_region,
            gpinn=gpinn_region,
            mu_grad=mu_grad_region,
        )

    def kfac_residual_vector(params, data, terms=None, regions=None):
        terms = active_terms(terms)
        regions = active_regions(regions)
        res = []

        if use_data and 'data' in terms:
            res += [loss_data_res_sub(params, data, idx) for idx in idxgall if include_region(idx, regions)]
        if use_eqn and 'eqn' in terms:
            res += [loss_eqn_res_sub(params, data, idx) for idx in idxgall if include_region(idx, regions)]
        if use_ct and 'ct' in terms:
            res += [loss_ct_res_sub(params, data, idx) for idx in idxgall if include_region(idx, regions)]
        if use_match and 'match' in terms:
            res += [loss_md_kfac_res_sub(params, data, idx) for idx in idxgall[:-1] if include_match(idx, regions)]
        if use_gpinn and 'gpinn' in terms:
            res += [loss_gpinn_res_sub(params, data, idx) for idx in idxgall if include_region(idx, regions)]
        if use_mu_grad and 'mu_grad' in terms:
            res += [loss_mu_grad_res_sub(params, data, idx) for idx in idxgall if include_region(idx, regions)]

        return jnp.zeros((0, 1)) if len(res) == 0 else jnp.concatenate([r.reshape(-1) for r in res]).reshape(-1, 1)

    def kfac_eval(params, data, terms=None, regions=None):
        terms = active_terms(terms)
        regions = active_regions(regions)
        active = active_region_tuple(data)
        res = []

        if use_data:
            data_blocks = {idx: data_block(params, data, idx) for idx in idxgall}
            data_err_list = jnp.array([
                data_blocks[idx][0] if idx in active else jnp.zeros(int(n_data_terms))
                for idx in idxgall
            ])
            data_err_subavg = masked_region_mean(data_err_list, data)
            loss_data = jnp.mean(data_w * data_err_subavg)
            if 'data' in terms:
                res += [data_blocks[idx][1] for idx in idxgall if include_region(idx, regions)]
        else:
            data_err_list = jnp.zeros((n_sub, int(n_data_terms)))
            data_err_subavg = jnp.zeros(int(n_data_terms))
            loss_data = 0.0

        if use_eqn:
            eqn_blocks = {idx: eqn_block(params, data, idx) for idx in idxgall}
            eqn_weights = current_eqn_weights(data).reshape((n_sub, 1))
            eqn_err_list = jnp.array([
                eqn_blocks[idx][0] if idx in active else jnp.zeros(n_eqn_terms)
                for idx in idxgall
            ])
            eqn_err_subavg = masked_region_mean(eqn_weights * eqn_err_list, data)
            loss_eqn = jnp.mean(eqn_w * eqn_err_subavg)
            if 'eqn' in terms:
                res += [eqn_blocks[idx][1] for idx in idxgall if include_region(idx, regions)]
        else:
            eqn_err_list = jnp.zeros((n_sub, n_eqn_terms)) if eqn_available else jnp.nan * jnp.ones((n_sub, n_eqn_terms))
            eqn_err_subavg = jnp.zeros(n_eqn_terms) if eqn_available else jnp.nan * jnp.ones(n_eqn_terms)
            loss_eqn = 0.0

        if use_ct:
            ct_blocks = {idx: ct_block(params, data, idx) for idx in idxgall}
            ct_err_list = jnp.array([
                ct_blocks[idx][0] if idx in active else jnp.zeros(int(n_ct_terms))
                for idx in idxgall
            ])
            ct_err_subavg = masked_region_mean(ct_err_list, data)
            loss_ct = jnp.mean(ct_w * ct_err_subavg)
            if 'ct' in terms:
                res += [ct_blocks[idx][1] for idx in idxgall if include_region(idx, regions)]
        else:
            ct_err_list = jnp.zeros((len(idxgall), int(n_ct_terms))) if ct_available else jnp.nan * jnp.ones((len(idxgall), int(n_ct_terms)))
            ct_err_subavg = jnp.zeros(int(n_ct_terms)) if ct_available else jnp.nan * jnp.ones(int(n_ct_terms))
            loss_ct = 0.0

        if use_match:
            md_blocks = {idx: match_block(params, data, idx) for idx in idxgall[:-1]}
            md_err_list = jnp.array([md_blocks[idx][0] for idx in idxgall[:-1]])
            md_err_subavg = jnp.mean(md_err_list, axis=0)
            loss_md = jnp.mean(md_w * md_err_subavg)
            if 'match' in terms:
                res += [md_blocks[idx][1] for idx in idxgall[:-1] if include_match(idx, regions)]
        else:
            md_err_list = jnp.zeros((len(idxgall)-1, md_w.shape[0])) if match else jnp.nan * jnp.ones((len(idxgall)-1, md_w.shape[0]))
            md_err_subavg = jnp.mean(md_err_list, axis=0)
            loss_md = 0.0

        if use_gpinn:
            gpinn_blocks = {idx: gpinn_block(params, data, idx) for idx in idxgall}
            gpinn_err_list = jnp.array([
                gpinn_blocks[idx][0] if idx in active else jnp.zeros(4)
                for idx in idxgall
            ])
            gpinn_err_subavg = masked_region_mean(gpinn_err_list, data)
            loss_gpinn = jnp.mean(gpinn_err_subavg)
            if 'gpinn' in terms:
                res += [gpinn_blocks[idx][1] for idx in idxgall if include_region(idx, regions)]
        else:
            gpinn_err_list = jnp.zeros((n_sub, 4))
            gpinn_err_subavg = jnp.zeros(4)
            loss_gpinn = 0.0

        if use_mu_grad:
            mu_grad_blocks = {idx: mu_grad_block(params, data, idx) for idx in idxgall}
            mu_grad_err_list = jnp.array([
                mu_grad_blocks[idx][0] if idx in active else jnp.zeros(2)
                for idx in idxgall
            ])
            active_mask = region_mask(data).reshape((n_sub,))
            floating_mask = jnp.array([idx in floating_idx for idx in idxgall], dtype=jnp.float32)
            combined_mask = (active_mask * floating_mask).reshape((n_sub, 1))
            denom = jnp.sum(combined_mask)
            mu_grad_err_subavg = jnp.where(
                denom > 0.0,
                jnp.sum(mu_grad_err_list * combined_mask, axis=0) / denom,
                jnp.zeros(2),
            )
            loss_mu_grad = jnp.where(denom > 0.0, jnp.mean(mu_grad_err_subavg), 0.0)
            if 'mu_grad' in terms:
                res += [mu_grad_blocks[idx][1] for idx in idxgall if include_region(idx, regions)]
        else:
            mu_grad_err_list = jnp.zeros((n_sub, 2))
            mu_grad_err_subavg = jnp.zeros(2)
            loss_mu_grad = 0.0

        if use_region_term_weights(data):
            data_region = jnp.mean(data_w * data_err_list, axis=1)
            eqn_region = jnp.mean(eqn_w * eqn_err_list, axis=1) if use_eqn else jnp.zeros(n_sub)
            ct_region = jnp.mean(ct_w * ct_err_list, axis=1) if use_ct else jnp.zeros(n_sub)
            if use_match:
                md_interface = jnp.mean(md_w * md_err_list, axis=1)
                md_region_sum = jnp.zeros(n_sub)
                md_region_count = jnp.zeros(n_sub)
                for pos, idx in enumerate(idxgall[:-1]):
                    left = idxgall.index(idx)
                    right = idxgall.index(idx + 1)
                    md_region_sum = md_region_sum.at[left].add(md_interface[pos])
                    md_region_sum = md_region_sum.at[right].add(md_interface[pos])
                    md_region_count = md_region_count.at[left].add(1.0)
                    md_region_count = md_region_count.at[right].add(1.0)
                md_region = jnp.where(md_region_count > 0.0, md_region_sum / md_region_count, 0.0)
            else:
                md_region = jnp.zeros(n_sub)
            gpinn_region = jnp.mean(gpinn_err_list, axis=1) if use_gpinn else jnp.zeros(n_sub)
            mu_grad_region = jnp.mean(mu_grad_err_list, axis=1) if use_mu_grad else jnp.zeros(n_sub)
            data_term = current_region_term_weights(data)['data'] * data_region
            eqn_term = current_region_term_weights(data)['eqn'] * eqn_region
            ct_term = current_region_term_weights(data)['ct'] * ct_region
            match_term = current_region_term_weights(data)['match'] * md_region
            gpinn_term = current_region_term_weights(data)['gpinn'] * gpinn_region
            mu_grad_term = current_region_term_weights(data)['mu_grad'] * mu_grad_region
            scalar_loss = jnp.mean(data_term + eqn_term + ct_term + match_term + gpinn_term + mu_grad_term)
        else:
            scalar_loss = (data_global_weight*loss_data
                           + eqn_global_weight*loss_eqn
                           + ct_global_weight*loss_ct
                           + current_match_weight(data)*loss_md
                           + loss_fun.gpinn_weight*loss_gpinn
                           + loss_fun.mu_grad_weight*loss_mu_grad)

        residuals = jnp.zeros((0, 1)) if len(res) == 0 else jnp.concatenate([r.reshape(-1) for r in res]).reshape(-1, 1)
        loss_n = jnp.sum(jnp.square(residuals)) / loss_fun.lref
        loss_info = jnp.hstack((scalar_loss, data_err_subavg, eqn_err_subavg,
                                md_err_subavg, ct_err_subavg,
                                loss_gpinn, gpinn_err_subavg,
                                loss_mu_grad, mu_grad_err_subavg))
        return loss_n, loss_info, [data_err_list, eqn_err_list, md_err_list, ct_err_list], residuals

    def kfac_residuals(params, data, terms=None, regions=None):
        return kfac_residual_vector(params, data, terms=terms, regions=regions)

    def kfac_objective(params, data, terms=None, regions=None):
        residuals = kfac_residual_vector(params, data, terms=terms, regions=regions)
        return jnp.sum(jnp.square(residuals)) / loss_fun.lref

    # loss function used for the PINN training
    def loss_fun(params, data):
        active = active_region_tuple(data)
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        if use_data:
            data_err_list = jnp.array([
                loss_data_sub(params, data, idx) if idx in active else jnp.zeros(int(n_data_terms))
                for idx in idxgall
            ])
            data_err_subavg = masked_region_mean(data_err_list, data)
            loss_data = jnp.mean(data_w * data_err_subavg)
        else:
            data_err_list = jnp.zeros((n_sub, int(n_data_terms)))
            data_err_subavg = jnp.zeros(int(n_data_terms))
            loss_data = 0.0

        # Compute calving front error if called for
        if use_ct:
            ct_err_list = jnp.array([
                loss_ct_sub(params, data, idx) if idx in active else jnp.zeros(int(n_ct_terms))
                for idx in idxgall
            ])
            ct_err_subavg = masked_region_mean(ct_err_list, data)
            loss_ct = jnp.mean(ct_w * ct_err_subavg)
        else:
            ct_err_list = jnp.zeros((len(idxgall), int(n_ct_terms))) if ct_available else jnp.nan * jnp.ones((len(idxgall), int(n_ct_terms)))
            ct_err_subavg = jnp.zeros(int(n_ct_terms)) if ct_available else jnp.nan * jnp.ones(int(n_ct_terms))
            loss_ct = 0.0

        # Compute equation error if called for
        eqn_weights = current_eqn_weights(data).reshape((n_sub, 1))
        if use_eqn:
            eqn_err_list = jnp.array([
                loss_eqn_sub(params, data, idx) if idx in active else jnp.zeros(n_eqn_terms)
                for idx in idxgall
            ])
            eqn_err_subavg = masked_region_mean(eqn_weights * eqn_err_list, data)
            loss_eqn = jnp.mean(eqn_w * eqn_err_subavg)
        else:
            eqn_err_list = jnp.zeros((n_sub, n_eqn_terms)) if eqn_available else jnp.nan * jnp.ones((n_sub, n_eqn_terms))
            eqn_err_subavg = jnp.zeros(n_eqn_terms) if eqn_available else jnp.nan * jnp.ones(n_eqn_terms)
            loss_eqn = 0.0

        # Compute matching error if called for
        if use_match:
            md_err_list = jnp.array(tree_map(lambda x: loss_md_sub(params, data, x), idxgall[:-1]))
            md_err_subavg = jnp.mean(md_err_list, axis=0)
            # jdb.print('md_err_subavg shape = {s}', s=md_err_subavg.shape)
            loss_md = jnp.mean(md_w * md_err_subavg)
        else:
            md_err_list = jnp.zeros((len(idxgall)-1, md_w.shape[0])) if match else jnp.nan * jnp.ones((len(idxgall)-1, md_w.shape[0]))
            md_err_subavg = jnp.mean(md_err_list, axis=0)
            loss_md = 0.0

        if use_gpinn:
            gpinn_err_list = jnp.array([
                loss_gpinn_sub(params, data, idx) if idx in active else jnp.zeros(4)
                for idx in idxgall
            ])
            gpinn_err_subavg = masked_region_mean(gpinn_err_list, data)
            loss_gpinn = jnp.mean(gpinn_err_subavg)
        else:
            gpinn_err_list = jnp.zeros((n_sub, 4))
            gpinn_err_subavg = jnp.zeros(4)
            loss_gpinn = 0.0

        if use_mu_grad:
            mu_grad_err_list = jnp.array([
                loss_mu_grad_sub(params, data, idx) if idx in active else jnp.zeros(2)
                for idx in idxgall
            ])
            active_mask = region_mask(data).reshape((n_sub,))
            floating_mask = jnp.array([idx in floating_idx for idx in idxgall], dtype=jnp.float32)
            combined_mask = (active_mask * floating_mask).reshape((n_sub, 1))
            denom = jnp.sum(combined_mask)
            mu_grad_err_subavg = jnp.where(
                denom > 0.0,
                jnp.sum(mu_grad_err_list * combined_mask, axis=0) / denom,
                jnp.zeros(2),
            )
            loss_mu_grad = jnp.where(denom > 0.0, jnp.mean(mu_grad_err_subavg), 0.0)
        else:
            mu_grad_err_list = jnp.zeros((n_sub, 2))
            mu_grad_err_subavg = jnp.zeros(2)
            loss_mu_grad = 0.0

        if use_region_term_weights(data):
            region_terms = region_term_values(params, data)
            data_term = current_region_term_weights(data)['data'] * region_terms['data']
            eqn_term = current_region_term_weights(data)['eqn'] * region_terms['eqn']
            ct_term = current_region_term_weights(data)['ct'] * region_terms['ct']
            match_term = current_region_term_weights(data)['match'] * region_terms['match']
            gpinn_term = current_region_term_weights(data)['gpinn'] * region_terms['gpinn']
            mu_grad_term = current_region_term_weights(data)['mu_grad'] * region_terms['mu_grad']
            loss = jnp.mean(data_term + eqn_term + ct_term + match_term + gpinn_term + mu_grad_term)
        else:
            loss = (data_global_weight*loss_data
                    + eqn_global_weight*loss_eqn
                    + ct_global_weight*loss_ct
                    + current_match_weight(data)*loss_md
                    + loss_fun.gpinn_weight*loss_gpinn
                    + loss_fun.mu_grad_weight*loss_mu_grad)
        # normalize the loss by the initial reference value
        loss_ref = loss_fun.lref
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack((loss, data_err_subavg, eqn_err_subavg,
                                md_err_subavg, ct_err_subavg,
                                loss_gpinn, gpinn_err_subavg,
                                loss_mu_grad, mu_grad_err_subavg))
        return loss_n, loss_info, [data_err_list, eqn_err_list, md_err_list, ct_err_list]

    loss_fun.lref = 1.0
    loss_fun.match_weight = jnp.array(match_weight)
    loss_fun.eqn_region_weights = jnp.ones(n_sub)
    loss_fun.gpinn_weight = jnp.array(gpinn_weight)
    loss_fun.mu_grad_weight = jnp.array(mu_grad_weight)
    loss_fun.region_term_values = region_term_values
    loss_fun.kfac_eval = kfac_eval
    loss_fun.kfac_residuals = kfac_residuals
    loss_fun.kfac_objective = kfac_objective
    return loss_fun


def _legacy_or_config_args(solNN, idxgall, basal_mask, eqn, front_eqn, config_type):
    if isinstance(eqn, config_type):
        return solNN, idxgall, basal_mask, eqn
    return None


def loss_joint_create(solNN:Tuple[Callable], eqn:Callable, front_eqn:Callable,
                      config:DIFFICEJointInversionConfig):
    return _loss_xpinn_create(
        solNN,
        list(config.idxgall),
        basal_mask=list(config.basal_mask),
        eqn=eqn,
        front_eqn=front_eqn,
        match=config.match,
        calving_front=config.calving_front,
        scales=None if config.scales is None else list(config.scales),
        match_weight=config.match_weight,
        match_component_weights=config.match_component_weights,
        gpinn_weight=config.gpinn_weight,
        mu_grad_weight=config.mu_grad_weight,
        global_weights=config.global_weights,
        grounded_only_interface_mu_ct=False,
        active_regions=None if config.active_regions is None else list(config.active_regions),
        include_inverse_data=False,
    )


def loss_regression_create(solNN:Tuple[Callable], idxgall:List[int],
                           basal_mask: List[bool]|None = None,
                           eqn:Callable = None, front_eqn:Callable = None,
                           match:bool = False,  calving_front:bool|None=None,
                           scales: List[SubScaleResult]|None = None,
                           match_weight: float = 1.0,
                           match_component_weights: ArrayLike|None = None,
                           gpinn_weight: float = 0.0,
                           mu_grad_weight: float = 0.0,
                           grounded_only_interface_mu_ct: bool = False,
                           active_regions: List[int]|None = None,
                           global_weights: dict[str, Any]|None = None):
    config_args = _legacy_or_config_args(
        solNN, idxgall, basal_mask, eqn, front_eqn, DIFFICEXPINNRegressionConfig
    )
    if config_args is not None:
        solNN, eqn, front_eqn, config = config_args
        return _loss_xpinn_create(
            solNN,
            list(config.idxgall),
            basal_mask=list(config.basal_mask),
            eqn=eqn,
            front_eqn=front_eqn,
            match=config.match,
            calving_front=config.calving_front,
            scales=None if config.scales is None else list(config.scales),
            match_weight=config.match_weight,
            match_component_weights=config.match_component_weights,
            gpinn_weight=config.gpinn_weight,
            mu_grad_weight=config.mu_grad_weight,
            global_weights=config.global_weights,
            grounded_only_interface_mu_ct=config.grounded_only_interface_mu_ct,
            active_regions=None if config.active_regions is None else list(config.active_regions),
            include_inverse_data=True,
        )
    return _loss_xpinn_create(
        solNN,
        idxgall,
        basal_mask=basal_mask,
        eqn=eqn,
        front_eqn=front_eqn,
        match=match,
        calving_front=calving_front,
        scales=scales,
        match_weight=match_weight,
        match_component_weights=match_component_weights,
        gpinn_weight=gpinn_weight,
        mu_grad_weight=mu_grad_weight,
        global_weights=global_weights,
        grounded_only_interface_mu_ct=grounded_only_interface_mu_ct,
        active_regions=active_regions,
        include_inverse_data=True,
    )


#%% loss for XPINNs regression second stage
def loss_regression_2ndstage_create(solNN:Tuple[Callable], idxgall:List[int],
                                    basal_mask: List[bool]|None = None,
                                    first_stage_velocity_misfit = None):
    if basal_mask is None:
        raise ValueError('No basal mask was supplied to the loss function')
    predNN, gradNN = solNN
    n_sub = len(idxgall)

    def first_stage_misfit(z, idx):
        misfit_data = first_stage_velocity_misfit[idx]
        x_all, misfit_all = misfit_data[0], misfit_data[1]
        misfit_scale = misfit_data[2] if len(misfit_data) > 2 else jnp.ones((2,))
        is_match = jnp.all(x_all[None, :, :] == z[:, None, :], axis=2)
        match_idx = jnp.argmax(is_match, axis=1)
        return misfit_all[match_idx], misfit_scale

    def loss_data_sub(params, data, idx):
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]
        if first_stage_velocity_misfit is None:
            u_target = u_smp
        else:
            u_res, u_scale = first_stage_misfit(x_smp, idx)
            u_target = u_res / u_scale
        u_pred = predNN(params, x_smp, idx)[:, 0:2]

        if basal_mask[idx]:
            data_u_err = ms_error(u_pred - u_target)
        else:
            data_u_err = jnp.array([0, 0])
        return jnp.hstack((data_u_err, jnp.zeros(4)))

    def loss_data_res_sub(params, data, idx):
        x_smp = data['smp'][0][idx]
        if first_stage_velocity_misfit is None:
            u_target = data['smp'][1][idx]
        else:
            u_res, u_scale = first_stage_misfit(x_smp, idx)
            u_target = u_res / u_scale
        u_pred = predNN(params, x_smp, idx)[:, 0:2]
        smp_weight = jnp.sqrt(1.0 / (2.0 * n_sub * x_smp.shape[0]))
        return (smp_weight * (u_pred - u_target)).reshape(-1)

    def kfac_residuals(params, data):
        res = [loss_data_res_sub(params, data, idx) for idx in idxgall]
        return jnp.concatenate([r.reshape(-1) for r in res]).reshape(-1, 1)

    def loss_fun(params, data):
        data_regional_weights = jnp.ones((n_sub, 1))
        data_err_list = jnp.array(tree_map(lambda x: loss_data_sub(params, data, x), idxgall))
        data_err_subavg = jnp.mean(data_err_list * data_regional_weights, axis=0)
        loss_data = jnp.mean(data_err_subavg[0:2])

        eqn_err_list = jnp.nan * jnp.ones((n_sub, 2))
        eqn_err_subavg = jnp.nan * jnp.ones(2)
        md_err_list = jnp.nan * jnp.ones((len(idxgall)-1, 6))
        md_err_subavg = jnp.mean(md_err_list, axis=0)
        ct_err_list = jnp.nan * jnp.ones((n_sub, 2))
        ct_err_subavg = jnp.nan * jnp.ones(2)

        loss_ref = loss_fun.lref
        loss_n = loss_data / loss_ref
        loss_info = jnp.hstack((loss_data, data_err_subavg, eqn_err_subavg, md_err_subavg, ct_err_subavg))
        return loss_n, loss_info, [data_err_list, eqn_err_list, md_err_list, ct_err_list]

    loss_fun.lref = 1.0
    loss_fun.kfac_residuals = kfac_residuals
    return loss_fun


#%% loss for inferring anisotropic viscosity

def loss_aniso_create(solNN, eqn_all, scale, idxgall, lw):
    ''' a function factory to create the loss function for anisotropic analysis
    :param solNN: neural network function for solutions and its derivative [tuple(callable, callable)]
    :param eqn_all: include governing equation and boundary equation of SSA [tuple(callable, callable)]
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    predNN, gradNN = solNN
    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # obtain the viscosity and strain rate scale in each sub-region
    all_info = jnp.array(tree_map(lambda x: sub_scale(scale[x]), idxgall))
    scale_info = all_info[:, 0:7]
    scale_nm = scale_info / jnp.mean(scale_info, axis=0)   # To do: check whether jnp.min or jnp.mean better
    mean_nm = all_info[:, 7:]
    u0, v0, h0, mu0, du0, dh0, term0 = jnp.split(scale_nm, 7, axis=1)
    uvh0 = jnp.hstack([u0, v0, h0])
    um, vm = jnp.split(mean_nm, 2, axis=1)

    # create the loss constraint for each sub-regions
    def loss_sub(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predNN(params, z, idx)
        # load the velocity data and their position
        x_smp = data['smp'][0][idx]
        u_smp = data['smp'][1][idx]

        # load the thickness data and their position
        xh_smp = data['smp'][2][idx]
        h_smp = data['smp'][3][idx]

        # load the position and weight of collocation points
        x_col = data['col'][0][idx]
        x_bd = data['bd'][0][idx]
        nn_bd = data['bd'][1][idx]

        # calculate the gradient of phi at origin
        output = net(x_smp)
        u_pred = output[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]

        # calculate the residue of equation
        f_pred = gov_eqn(net, x_col, scale[idx])[0]
        f_bd = front_eqn(net, x_bd, nn_bd, scale[idx])[0]

        # calculate the mean squared error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err)) * uvh0[idx]
        # calculate the mean squared error of equation
        eqn_err = ms_error(f_pred) * term0[idx]
        bd_err = ms_error(f_bd) * h0[idx]
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred)) / 2) * mu0[idx]

        # group all the error for output
        err_all = jnp.hstack([data_err, eqn_err, bd_err, sp_err])
        return err_all, f_pred

    # create the continuation loss constraint at the interface of adjacent subregions
    def loss_match(params, data, idx):
        # create the function for gradient calculation involves input Z only
        net = lambda x, id: predNN(params, x, id)
        gdnet = lambda x, id: gradNN(params, x, id)
        fgovterm = lambda x, id: gov_eqn(lambda x: net(x, id), x, scale[id])[1]
        # load the position at the matching boundary between sub-regions
        x_md = data['md'][0][idx]

        """C0 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        U_md1 = net(x_md[:, 0:2], idx)
        u_md1 = (U_md1[:, 0:1] + um[idx]) * u0[idx]
        v_md1 = (U_md1[:, 1:2] + vm[idx]) * v0[idx]
        h_md1 = (U_md1[:, 2:3]) * h0[idx]
        mu_md1 = (U_md1[:, 3:5]) * mu0[idx]     # include both mu and eta
        vars_md1 = jnp.hstack([u_md1, v_md1, h_md1, 2*jnp.log(mu_md1)])
        # obtain the variable in sub-region 2 at the interface
        U_md2 = net(x_md[:, 2:4], idx+1)
        u_md2 = (U_md2[:, 0:1] + um[idx+1]) * u0[idx+1]
        v_md2 = (U_md2[:, 1:2] + vm[idx+1]) * v0[idx+1]
        h_md2 = (U_md2[:, 2:3]) * h0[idx+1]
        mu_md2 = (U_md2[:, 3:5]) * mu0[idx+1]   # include both mu and eta
        vars_md2 = jnp.hstack([u_md2, v_md2, h_md2, 2*jnp.log(mu_md2)])
        # group the c0 error
        match_c0_err = ms_error(vars_md1 - vars_md2)

        """C1 stitching condition at the boundary"""
        # obtain the variable in sub-region 1 at the interface
        dU_md1 = gdnet(x_md[:, 0:2], idx)
        duv_md1 = dU_md1[:, 0:4] * du0[idx]
        dh_md1 = dU_md1[:, 4:6] * dh0[idx]
        dvars_md1 = jnp.hstack([duv_md1, dh_md1])
        # obtain the variable in sub-region 2 at the interface
        dU_md2 = gdnet(x_md[:, 2:4], idx+1)
        duv_md2 = dU_md2[:, 0:4] * du0[idx+1]
        dh_md2 = dU_md2[:, 4:6] * dh0[idx+1]
        dvars_md2 = jnp.hstack([duv_md2, dh_md2])
        # group the c1 error
        match_c1_err = ms_error(nthrt(dvars_md1, 2) - nthrt(dvars_md2, 2))

        """C2 stitching condition at the boundary"""
        # calculate equation residue in sub-region 1 at the interface
        term_md1 = fgovterm(x_md[:, 0:2], idx)[:, 0:-1] * term0[idx]
        # calculate equation residue in sub-region 2 at the interface
        term_md2 = fgovterm(x_md[:, 2:4], idx+1)[:, 0:-1] * term0[idx+1]
        # calculate the c2 error
        match_c2_err = ms_error(nthrt(term_md1, 2) - nthrt(term_md2, 2))

        # group all the stitching conditions
        mc0_err = jnp.mean(match_c0_err)
        mc1_err = jnp.mean(match_c1_err)
        mc2_err = jnp.mean(match_c2_err)
        match_err = jnp.hstack([mc0_err, mc1_err*0.8, mc2_err*0.5])
        return match_err

    # loss function used for the PINN training
    def loss_fun(params, data):
        # calculate the data_err, eqn_err and bound_err for each sub-regions
        # Unpack the tuple return from loss_sub
        sub_results = [loss_sub(params, data, x) for x in idxgall]
        reg_err_list = [res[0] for res in sub_results]
        residuals_list = [res[1] for res in sub_results]

        reg_err = jnp.mean(jnp.array(reg_err_list), axis=0)
        # calculate the error at the matching boundary
        match_err_list = tree_map(lambda x: loss_match(params, data, x), idxgall[0:-1])
        match_err = jnp.mean(jnp.array(match_err_list), axis=0)
        # group all the error
        err_all = jnp.hstack([reg_err, match_err])

        # set the weight for each condition and equation
        data_w = jnp.array([1., 1., 0.6])
        eqn_w = jnp.array([1., 1.])
        bd_w = jnp.array([1., 1.])
        sp_w = jnp.array([1.])
        md_w = jnp.ones(match_err.shape[0])
        # group all the weight
        wgh_all = jnp.hstack([data_w, eqn_w, bd_w, sp_w, md_w])

        # modify the contribution of each loss term by their weights
        loss_each = err_all * wgh_all
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(loss_each[0:3])
        loss_eqn = jnp.sum(loss_each[3:5])
        loss_bd = jnp.sum(loss_each[5:7])
        loss_sp = jnp.sum(loss_each[7:8])
        loss_md = jnp.sum(loss_each[8:])

        # loading the pre-saved loss parameter
        loss_ref = loss_fun.lref
        # load the weight for the regularization loss
        wsp = loss_fun.wsp
        # calculate the total loss
        loss = (lw[0] * loss_data + lw[1] * loss_eqn + lw[2] * loss_bd + lw[3] * loss_md + wsp * loss_sp)
        # normalize the loss by the initial reference value
        loss_n = loss / loss_ref
        # group the loss of all conditions and equations
        loss_info = [jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_md, loss_sp]), err_all]), residuals_list]
        return loss_n, loss_info

    # setting the pre-saved loss parameter to loss_fun
    loss_fun.lref = 1.0
    loss_fun.wsp = lw[4]
    return loss_fun
