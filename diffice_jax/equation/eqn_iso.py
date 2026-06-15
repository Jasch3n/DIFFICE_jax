"""
@author: Yongji Wang
"""

import jax
import jax.numpy as jnp
from jax import lax
import jax.debug as jdb
from diffice_jax.data.xpinns.preprocessing import SubScaleResult
from diffice_jax.core.contracts import SSA_ISO_FLOATING, SSA_ISO_GROUNDED
DEBUG = False

CONTRACTS = {
    "floating": SSA_ISO_FLOATING,
    "grounded": SSA_ISO_GROUNDED,
}

#%% Isotropic shallow-shelf approximation (SSA) equations in the normalized form
# [FIXME]: Remember to make scale generalizable to singular PINN scale
def gov_eqn(net, x, scale: SubScaleResult, basal=False):
    """
    :param net: the neural net instance for calculating the informed part
    """
    # [TODO]: pass these parameters in through the config file
    # setting the global parameters
    rho = scale.dynamic_scale.rho
    rho_w = scale.dynamic_scale.rho_w
    g = scale.dynamic_scale.g
    gamma_c = scale.dynamic_scale.gamma_c
    gamma_mu = scale.dynamic_scale.gamma_mu

    dmean, drange = scale.data_mean, scale.data_range
    lx0, ly0, u0, v0 = drange[0:4]
    ym = dmean[1]
    um = dmean[2]
    vm = dmean[3]
    h0 = dmean[4]

    u0m = scale.dynamic_scale.u0
    l0m = scale.dynamic_scale.l0
    ru0 = u0 / u0m
    rv0 = v0 / u0m
    rx0 = lx0 / l0m
    ry0 = ly0 / l0m

    def grad1stOrder(net, x, basal=basal):  # shape = (2, )
        # aux_ocean_mask=lax.stop_gradient(aux_ocean_mask)
        sol = net(x)
        grad = jnp.ravel(jax.jacfwd(net)(x), order='C')
        if DEBUG:
            jdb.print('[DEBUG eqn_iso.py]: grad shape = {s}', s=grad.shape)
            jdb.print('DEBUG eqn_iso.py]: sol shape = {s}', s=sol.shape)
        u = sol[0]
        v = sol[1]
        h = sol[2] # note that thickness is normalized as h = h_hat * h_m, where h_g has been approximated with h_m
        if basal:
            s = sol[3]
            mu = sol[4]
            c = sol[5]
        else:
            mu = sol[4] if sol.shape[0] > 4 else sol[3]
            c = 0.0

        u_x = grad[0] * ru0 / rx0
        u_y = grad[1] * ru0 / ry0
        v_x = grad[2] * rv0 / rx0
        v_y = grad[3] * rv0 / ry0
        h_x = grad[4] / rx0
        h_y = grad[5] / ry0
        if basal:
            s_x = grad[6] / rx0
            s_y = grad[7] / ry0
        strate = (u_x ** 2 + v_y ** 2 + 0.25 * (u_y + v_x) ** 2 + u_x * v_y) ** 0.5

        term1_1 = 2 * mu * h * (2 * u_x + v_y)
        term2_1 = 2 * mu * h * (2 * v_y + u_x)
        term12_2 = mu * h * (u_y + v_x)

        if basal:
            # term1_1 *= 100
            # term2_1 *= 100
            # term12_2 *= 100
            ud = u*u0 + um
            vd = v*v0 + vm
            # veld = jnp.sqrt(ud**2 + vd**2) + 1e-15
            term1_4 = c * (ud / u0m)
            term2_4 = c * (vd / u0m)
            term1_3 = h * s_x
            term2_3 = h * s_y
            
        else:
            term1_3 = h * h_x
            term2_3 = h * h_y

        if basal:
            return jnp.hstack([term1_1, term2_1, term12_2, term1_3, term2_3, strate, term1_4, term2_4, ud/u0m, vd/u0m])
        else:
            return jnp.hstack([term1_1, term2_1, term12_2, term1_3, term2_3, strate])

    # take the second order derivative in SSA
    func_g = lambda x: grad1stOrder(net, x)
    term = func_g(x)
    grad_term = jnp.ravel(jax.jacfwd(func_g)(x), order='C')

    e1term1 = grad_term[0] / rx0  # (term1_1, x)
    e1term2 = grad_term[5] / ry0  # (term12_2, y)
    e2term1 = grad_term[3] / ry0  # (term2_1, y)
    e2term2 = grad_term[4] / rx0  # (term12_2, x)
    Rxx     = term[0]
    Ryy     = term[1]
    Rxy     = term[2]
    e1term3 = term[3]
    e2term3 = term[4]
    strate = term[5]
    if basal:
        u      = term[8]
        v      = term[9]
        e1term4 = term[6]
        e2term4 = term[7]

    if basal:
        grav_factor  = 1.0     / gamma_mu
        basal_factor = gamma_c / gamma_mu
        visc_terms_1 = e1term1 + e1term2
        visc_terms_2 = e2term1 + e2term2
        # e1_SIA        = e1term3 + e1term4
        # e1_correction = jnp.array([0.0])
        e1 = visc_terms_1 - (grav_factor*e1term3 + basal_factor*e1term4)
        e2 = visc_terms_2 - (grav_factor*e2term3 + basal_factor*e2term4)
        
        # e1 = v*e1term1 + v*e1term2 - u*e2term2 - u*e2term2 - v*e1term3 + u*e2term3 
    else:
        e1 = e1term1 + e1term2 - e1term3
        e2 = e2term1 + e2term2 - e2term3

    if basal:
        f_eqn = jnp.hstack([e1, e2])
        # f_eqn = jnp.hstack([e1, e1])
    else:
        f_eqn = jnp.hstack([e1, e2])

    if DEBUG:
        jdb.print('Evaluating the SSA equation assuming {x} ice.', x='grounded' if basal else 'floating')

    if basal:
        val_term = jnp.hstack([e1term1, e1term2, e1term3, e2term1, e2term2, e2term3, strate, e1term4, e2term4])
        return f_eqn, val_term
    else:
        val_term = jnp.hstack([e1term1, e1term2, e1term3, e2term1, e2term2, e2term3, strate])
        return f_eqn, val_term


#%% Isotorpic dynamic boundary condition at calving front in the normalized form

def front_eqn(net, x, nb, scale: SubScaleResult):
    """
    :param net: the neural net instance for calculating the informed part
    :param nb: outward normal direction at the boundary
    """

    # setting the global parameters
    rho = scale.dynamic_scale.rho
    rho_w = scale.dynamic_scale.rho_w
    g = scale.dynamic_scale.g
    gd = g * (1 - rho / rho_w)  # gravitational acceleration

    dmean, drange = scale.data_mean, scale.data_range
    lx0, ly0, u0, v0 = drange[0:4]
    h0 = dmean[4]

    u0m = lax.max(u0, v0)
    l0m = lax.min(lx0, ly0)
    ru0 = u0 / u0m
    rv0 = v0 / u0m
    rx0 = lx0 / l0m
    ry0 = ly0 / l0m

    sol = net(x)
    grad = jnp.ravel(jax.jacfwd(net)(x), order='C')
    h = sol[2]
    # Vanilla floating PINNs output [u, v, h, mu]; newer floating schemas may
    # include surface elevation before mu. Avoid relying on JAX's out-of-bounds
    # gather behavior when evaluating the calving-front loss.
    mu = sol[4] if sol.shape[0] > 4 else sol[3]

    u_x = grad[0] * ru0 / rx0
    u_y = grad[1] * ru0 / ry0
    v_x = grad[2] * rv0 / rx0
    v_y = grad[3] * rv0 / ry0

    term1_1 = 2 * mu * (2 * u_x + v_y)
    term2_1 = 2 * mu * (2 * v_y + u_x)
    term12 = mu * (u_y + v_x)
    term_h = 0.5 * h

    e1 = term1_1 * nb[0] + term12 * nb[1] - term_h * nb[0]
    e2 = term12 * nb[0] + term2_1 * nb[1] - term_h * nb[1]

    f_eqn = jnp.hstack([e1, e2])
    val_term = jnp.hstack([term1_1, term2_1, term12, term_h])
    return f_eqn, val_term
