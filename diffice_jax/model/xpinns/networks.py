import sys
import jax
import jax.numpy as jnp
from jax import lax
from pathlib import Path
import jax.debug as jdb

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def floating_surface(uvhs, scale):
    """Derive floating-region surface elevation from thickness by flotation."""

    if hasattr(scale, "dynamic_scale"):
        rho = scale.dynamic_scale.rho
        rho_w = scale.dynamic_scale.rho_w
        h_mean = scale.data_mean.h_mean
        s_mean = scale.data_mean.s_mean
        s_range = scale.data_range.s_range
    else:
        # Legacy callers pass ``(data_mean, data_range)`` instead of
        # ``SubScaleResult``; keep that path until the old wrappers retire.
        data_mean, data_range = scale
        rho = 917.0
        rho_w = 1023.0
        h_mean = data_mean[4]
        s_mean = data_mean[5] if data_mean.shape[0] > 5 else 0.0
        s_range = data_range[5] if data_range.shape[0] > 5 else h_mean
    h = uvhs[:, 2:3] * h_mean if uvhs.ndim == 2 else uvhs[2] * h_mean
    s = (rho_w - rho) * h / rho_w
    s_n = (s - s_mean) / s_range
    if uvhs.ndim == 2:
        return jnp.hstack([uvhs[:, 0:3], s_n])
    return jnp.hstack([uvhs[0:3], jnp.array([s_n])])


def is_embedded_net(params):
    """Detect whether the final layer is a Fourier-feature embedding matrix."""

    if len(params) < 2:
        return False
    weight, bias = params[-1]
    return weight.ndim == 2 and bias.ndim == 1 and weight.shape[1] == 2 and weight.shape[0] == bias.shape[0]


# define the basic formation of neural network
def neural_net(params, x, scl, act_s=0):
    '''
    :param params: weights and biases
    :param x: input data [array with shape [m]]; m is number of inputs)
    :param scl: scale factor for the first layer
    :param act_s: activation mode:
                  0 = tanh everywhere
                  1 = sin everywhere
                  2 = sin on first hidden layer, tanh on rest (MSNN correction)
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    # jdb.print('params len = {s}', s=len(params))
    # for i in range(len(params)):
    #     jdb.print('params {i} shape = {s}', i=i, s=params[i].shape)
    # jdb.print('x shape = {s}', s=x.shape)
    # choose the activation function for first and remaining layers
    first_actv = [jnp.tanh, jnp.sin, jnp.sin][act_s]
    rest_actv  = [jnp.tanh, jnp.sin, jnp.tanh][act_s]
    # normalize the input
    H = x  # input has been normalized

    if is_embedded_net(params):
        embed_layer = params[-1]
        standard_params = params[:-1]
        B = lax.stop_gradient(embed_layer[0])
        x_proj = jnp.dot(x, B.T) * 2 * jnp.pi
        H = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    else:
        standard_params = params

    # separate the first, hidden and last layers
    first, *hidden, last = standard_params
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
    """Create XPINN solution and gradient functions for all sub-regions."""

    single_region_scale = (
        isinstance(scale, (list, tuple))
        and len(scale) == 2
        and not hasattr(scale[0], "dynamic_scale")
        and hasattr(scale[0], "shape")
    )

    def region_scale(idx):
        return scale if single_region_scale else scale[int(idx)]

    def uses_legacy_scale(idx):
        return not hasattr(region_scale(idx), "dynamic_scale")

    def is_basal_region(idx):
        idx = int(idx)
        return False if idx >= len(basal_mask) else basal_mask[idx]

    if basal_mask is None:
        ng = 1 if single_region_scale else len(scale)
        basal_mask = [False] * ng

    def f(params, x, idx):
        idx = int(idx)
        # generate the NN
        uvh = neural_net(params['net_u'][idx], x, scl, act_s)
        mu = neural_net(params['net_mu'][idx], x, scl, act_s)

        if is_basal_region(idx):
            c = neural_net(params['net_c'][idx], x, scl, act_s)
            sol = jnp.hstack([uvh, jnp.exp(mu), jnp.exp(c)])
        elif uses_legacy_scale(idx):
            sol = jnp.hstack([uvh[:, 0:3], jnp.exp(mu)]) if uvh.ndim == 2 else jnp.hstack([uvh[0:3], jnp.exp(mu)])
        else:
            uvh = floating_surface(uvh, region_scale(idx))
            sol = jnp.hstack([uvh, jnp.exp(mu), jnp.zeros_like(mu)])

        return sol

    def gradf(params, x, idx):
        idx = int(idx)
        drange = region_scale(idx)[1]
        lx0, ly0, u0, v0 = drange[0:4]
        u0m = lax.max(u0, v0)
        l0m = lax.min(lx0, ly0)
        ru0 = u0 / u0m
        rv0 = v0 / u0m
        rx0 = lx0 / l0m
        ry0 = ly0 / l0m

        # [IMPORTANT NOTE]: The gradf function is NOT used to compute the equation residuals
        #                   Therefore, the same scaling here must be done again when computing them.
        coeff = jnp.hstack([ru0/rx0, ru0/ry0, rv0/rx0, rv0/ry0, 1/rx0, 1/ry0])

        def grad_point(z):
            jac = jax.jacfwd(lambda zz: f(params, zz, idx)[:6])(z)
            return jnp.ravel(jac, order='C')

        grad = jax.vmap(grad_point)(x) if x.ndim == 2 else grad_point(x)

        # ensure that the velocity gradient is normalize by the same scale
        # (this is an important step to compute the normalized strain rate)
        duvh = grad[:, 0:6] * coeff if grad.ndim == 2 else grad[0:6] * coeff

        # calculate the strain rate
        u_x = duvh[:, 0] if duvh.ndim == 2 else duvh[0]
        u_y = duvh[:, 1] if duvh.ndim == 2 else duvh[1]
        v_x = duvh[:, 2] if duvh.ndim == 2 else duvh[2]
        v_y = duvh[:, 3] if duvh.ndim == 2 else duvh[3]
        strate = (u_x ** 2 + v_y ** 2 + 0.25 * (u_y + v_x) ** 2 + u_x * v_y) ** 0.5

        # group the solution
        if uses_legacy_scale(idx):
            gsol = jnp.hstack([duvh, strate[:, None]]) if grad.ndim == 2 else jnp.hstack([duvh, jnp.array([strate])])
        elif grad.ndim == 2:
            gsol = jnp.hstack([duvh, strate[:, None], grad])
        else:
            gsol = jnp.hstack([duvh, jnp.array([strate]), grad])
        return gsol

    return f, gradf
