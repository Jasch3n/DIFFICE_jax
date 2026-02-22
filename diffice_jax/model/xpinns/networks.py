import sys
import jax.numpy as jnp
from jax import lax
from pathlib import Path
import jax.debug as jdb

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from equation.eqn_iso import vectgrad

# define the basic formation of neural network
def neural_net(params, x, scl, act_s=0):
    '''
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param scl: scale factor for the first layer
    :param act_s: activation mode:
                  0 = tanh everywhere
                  1 = sin everywhere
                  2 = sin on first hidden layer, tanh on rest (MSNN correction)
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    # choose the activation function for first and remaining layers
    first_actv = [jnp.tanh, jnp.sin, jnp.sin][act_s]
    rest_actv  = [jnp.tanh, jnp.sin, jnp.tanh][act_s]
    # normalize the input
    H = x  # input has been normalized
    # separate the first, hidden and last layers
    first, *hidden, last = params
    # calculate the first layers output with right scale
    H = first_actv(jnp.dot(H, first[0]) * scl + first[1])
    # calculate the middle layers output
    for layer in hidden:
        H = rest_actv(jnp.dot(H, layer[0]) + layer[1])
    # no activation function for last layer
    var = jnp.dot(H, last[0]) + last[1]
    return var


# wrapper to create solution function with given domain size
def solu_create(scale, scl=1, act_s=0, basal_mask=None):
    '''
    :param limit: domain size of the input
    :return: function of the solution (a callable)
    '''
    # create default basal mask if not provided (all floating)
    if basal_mask is None:
        ng = len(scale)
        basal_mask = [False] * ng

    def f(params, x, idx):
        # generate the NN
        uvh = neural_net(params['net_u'][idx], x, scl, act_s)
        mu = neural_net(params['net_mu'][idx], x, scl, act_s)
        
        if basal_mask[idx]:
            # jdb.print('......Doing forward calculation for a grounded region. regionIdx={x}', x=idx)
            c = neural_net(params['net_c'][idx], x, scl, act_s)
            sol = jnp.hstack([uvh, jnp.exp(mu), jnp.exp(c)])
        else:
            # jdb.print('......Doing forward calculation for a floating region. regionIdx={x}', x=idx)
            sol = jnp.hstack([uvh, jnp.exp(mu)])
        return sol

    def gradf(params, x, idx):
        drange = scale[idx][1]
        lx0, ly0, u0, v0 = drange[0:4]
        u0m = lax.max(u0, v0)
        l0m = lax.max(lx0, ly0)
        ru0 = u0 / u0m
        rv0 = v0 / u0m
        rx0 = lx0 / l0m
        ry0 = ly0 / l0m
        coeff = jnp.hstack([ru0/rx0, ru0/ry0, rv0/rx0, rv0/ry0, 1/rx0, 1/ry0])
        # load the network
        net = lambda x: f(params, x, idx)
        # calculate the gradient
        grad = vectgrad(net, x)[0]
        # ensure that the velocity gradient is normalize by the same scale
        # (this is an important step to compute the normalized strain rate)
        duvh = grad[:, 0:6] * coeff
        # calculate the strain rate
        u_x = duvh[:, 0:1]
        u_y = duvh[:, 1:2]
        v_x = duvh[:, 2:3]
        v_y = duvh[:, 3:4]
        strate = (u_x ** 2 + v_y ** 2 + 0.25 * (u_y + v_x) ** 2 + u_x * v_y) ** 0.5
        # group the solution
        gsol = jnp.hstack([duvh, strate])
        return gsol

    return f, gradf


def msnn_solu_create(scale, frozen_stages, active_epsilon, active_kappa,
                    scl=1, basal_mask=None):
    """Create a multi-stage prediction function (MSNN combined ansatz).

    The combined prediction is:
        u_combined(x) = u_0(x) + Σ_{k=1}^{K-1} ε_k ⊙ u_k(x, κ_k)  [frozen]
                      + ε_K ⊙ u_K(x, κ_K)                          [active, trained]

    where ⊙ denotes element-wise (per-variable) scaling.
    Epsilon arrays also include eps_mu (and eps_c for grounded) so that
    log-viscosity and log-basal-friction corrections are also scaled.

    Args:
        scale: Scale info per sub-region.
        frozen_stages: List of (params_dict, epsilon_list, kappa) tuples for
                       all completed stages (including Stage 0).
                       Stage 0 uses act_s=0 (tanh); higher stages use act_s=2.
                       epsilon_list[i] is either a scalar ε (Stage 0) or a
                       jnp array of per-variable ε values for sub-region i.
        active_epsilon: List of per-variable ε arrays for the current stage,
                        one per sub-region.  Each is a jnp array of shape
                        (n_uvh + 1,) for floating or (n_uvh + 2,) for grounded,
                        where the extra entries are eps_mu (and eps_c).
                        Can also be a scalar 0.0 to skip the active stage.
        active_kappa: κ value for the active stage (used as scl).
        scl: Scale factor for Stage 0 network (default 1).
        basal_mask: List of booleans per sub-region.

    Returns:
        (f, gradf): Combined prediction function and its gradient.
    """
    if basal_mask is None:
        ng = len(scale)
        basal_mask = [False] * ng

    # Pre-compute per-region skip flags at closure-creation time (Python level).
    # This avoids traced boolean conversions inside JIT.
    def _is_zero_eps(eps):
        if isinstance(eps, (int, float)):
            return eps == 0
        # eps is a numpy/jnp array — evaluate to concrete bool NOW, not inside JIT
        return bool(float(jnp.sum(jnp.abs(jnp.asarray(eps)))) == 0.0)

    _skip_active = [_is_zero_eps(active_epsilon[i]) for i in range(len(active_epsilon))]

    def f(params, x, idx):
        """Combined multi-stage forward pass.

        'params' contains ONLY the active (current) stage parameters.
        All frozen stages are captured in the closure.
        """
        # --- Frozen stages (no gradient) ---
        sol = jnp.zeros((x.shape[0], 0))  # will be replaced by stage 0
        mu_total = jnp.zeros((x.shape[0], 1))
        c_total = jnp.zeros((x.shape[0], 1))

        for stage_idx, (stage_params, eps_list, kappa) in enumerate(frozen_stages):
            if stage_idx == 0:
                # Stage 0: standard network (tanh, act_s=0)
                # NOTE: No stop_gradient here! We need the Jacobian w.r.t. x
                # for gov_eqn's PDE residual computation (spatial derivatives).
                # Frozen params are already safe from optimizer updates because
                # they're captured in the closure, not passed as `params`.
                uvh = neural_net(stage_params['net_u'][idx], x, scl, act_s=0)
                mu_net = neural_net(stage_params['net_mu'][idx], x, scl, act_s=0)

                mu_total = mu_total + mu_net

                if basal_mask[idx]:
                    c_net = neural_net(stage_params['net_c'][idx], x, scl, act_s=0)
                    c_total = c_total + c_net
                    sol = jnp.hstack([uvh])  # Append mu and c at the end
                else:
                    sol = jnp.hstack([uvh])
            # else:
            #     # Higher frozen stages: correction nets (sin-first, act_s=2)
            #     uvh_corr = neural_net(stage_params['net_u'][idx], x, kappa, act_s=2)
            #     mu_corr = neural_net(stage_params['net_mu'][idx], x, kappa, act_s=2)

            #     eps = eps_list[idx]
            #     # Per-variable scaling: eps[:n_uvh] for uvh, eps[n_uvh] for mu
            #     n_uvh = 4 if basal_mask[idx] else 3
            #     sol = sol.at[:, :n_uvh].add(eps[:n_uvh] * uvh_corr)
            #     eps_mu = eps[n_uvh]
            #     mu_total = mu_total + eps_mu * mu_corr

            #     if basal_mask[idx] and stage_params['net_c'][idx] is not None:
            #         c_corr = neural_net(stage_params['net_c'][idx], x, kappa, act_s=2)
            #         eps_c = eps[n_uvh + 1]
            #         c_total = c_total + eps_c * c_corr

        # --- Active stage (gradients flow through) ---
        # Skip flag was pre-computed at Python level to avoid traced bool conversion.
        if not _skip_active[idx]:
            uvh_active = neural_net(params['net_u'][idx], x, active_kappa, act_s=2)
            mu_active = neural_net(params['net_mu'][idx], x, active_kappa, act_s=2)

            eps_a = active_epsilon[idx]
            # Per-variable scaling: eps_a[:n_uvh] for uvh, eps_a[n_uvh] for mu
            n_uvh = 4 if basal_mask[idx] else 3
            sol = sol.at[:, :n_uvh].add(eps_a[:n_uvh] * uvh_active)
            eps_mu_a = eps_a[n_uvh]
            mu_total = mu_total + eps_mu_a * mu_active

            if basal_mask[idx] and params['net_c'][idx] is not None:
                c_active = neural_net(params['net_c'][idx], x, active_kappa, act_s=2)
                eps_c_a = eps_a[n_uvh + 1]
                c_total = c_total + eps_c_a * c_active

        # Finally, append the exponentiated accumulated variables to the solution
        if basal_mask[idx]:
            sol = jnp.hstack([sol, jnp.exp(mu_total), jnp.exp(c_total)])
        else:
            sol = jnp.hstack([sol, jnp.exp(mu_total)])

        return sol

    def gradf(params, x, idx):
        drange = scale[idx][1]
        lx0, ly0, u0, v0 = drange[0:4]
        u0m = lax.max(u0, v0)
        l0m = lax.max(lx0, ly0)
        ru0 = u0 / u0m
        rv0 = v0 / u0m
        rx0 = lx0 / l0m
        ry0 = ly0 / l0m
        coeff = jnp.hstack([ru0/rx0, ru0/ry0, rv0/rx0, rv0/ry0, 1/rx0, 1/ry0])
        # load the combined network
        net = lambda x: f(params, x, idx)
        # calculate the gradient
        grad = vectgrad(net, x)[0]
        # ensure that the velocity gradient is normalized by the same scale
        duvh = grad[:, 0:6] * coeff
        # calculate the strain rate
        u_x = duvh[:, 0:1]
        u_y = duvh[:, 1:2]
        v_x = duvh[:, 2:3]
        v_y = duvh[:, 3:4]
        strate = (u_x ** 2 + v_y ** 2 + 0.25 * (u_y + v_x) ** 2 + u_x * v_y) ** 0.5
        gsol = jnp.hstack([duvh, strate])
        return gsol

    return f, gradf
