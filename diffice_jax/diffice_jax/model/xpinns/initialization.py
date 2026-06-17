from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_map

# initialize weights and biases of a single network
def init_single_net(parent_key, layer_widths):
    params = []
    keys = random.split(parent_key, num=len(layer_widths) - 1)
    # create the weights and biases for the network
    for in_dim, out_dim, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = random.split(key)
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * xavier_stddev]
        )
    return params


# generate weights and biases for all networks required in the XPINNs problem
def init_nets(parent_key, n_hl, n_unit, n_sub=1, aniso=False, basal_mask=None):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    :param n_sub: number of sub-regions
    '''
    if basal_mask is None:
        basal_mask = [False] * n_sub

    # set the default number of output for viscosity
    n_mu = 1
    # for anisotropic model
    if aniso:
        # number of viscosity output is 2
        n_mu = 2

    # generate the random key for each network (in list format)
    # We need up to 3 keys per subregion (u, mu, c)
    sub_keys = random.split(parent_key, n_sub)

    params_u = []
    params_mu = []
    params_c = []

    for i, key in enumerate(sub_keys):
        k_u, k_mu, k_c = random.split(key, 3)
        
        is_basal = basal_mask[i]
        
        # set the neural network shape for u, v, h (and s if basal)
        layers1 = [2] + n_hl * [n_unit] + [4 if is_basal else 3]
        params_u.append(init_single_net(k_u, layers1))
        
        # set the neural network shape for mu
        layers2 = [2] + n_hl * [n_unit] + [n_mu]
        params_mu.append(init_single_net(k_mu, layers2))
        
        if is_basal:
            # set the neural network shape for c (basal friction)
            layers3 = [2] + n_hl * [n_unit] + [1]
            params_c.append(init_single_net(k_c, layers3))
        else:
            params_c.append(None)

    return dict(net_u=params_u, net_mu=params_mu, net_c=params_c)


def init_correction_nets(parent_key, n_hl, n_unit, n_sub, kappa_per_region,
                         aniso=False, basal_mask=None):
    """Initialize correction networks for an MSNN stage.

    Same structure as init_nets but with:
      - Configurable (smaller) n_hl and n_unit
      - Zero-initialized final layer (no perturbation at stage start)
      - κ scaling is applied at runtime via neural_net's scl parameter

    Args:
        parent_key: JAX PRNG key.
        n_hl: Number of hidden layers for correction nets.
        n_unit: Number of units per hidden layer.
        n_sub: Number of sub-regions.
        kappa_per_region: List of κ scale factors, one per sub-region.
        aniso: Whether anisotropic viscosity model is used.
        basal_mask: List of booleans (True = grounded) per sub-region.

    Returns:
        dict with 'net_u', 'net_mu', 'net_c' correction parameters.
    """
    if basal_mask is None:
        basal_mask = [False] * n_sub

    n_mu = 2 if aniso else 1

    sub_keys = random.split(parent_key, n_sub)

    params_u = []
    params_mu = []
    params_c = []

    for i, key in enumerate(sub_keys):
        k_u, k_mu, k_c = random.split(key, 3)
        is_basal = basal_mask[i]
        kappa = kappa_per_region[i]

        # Same output dimensions as the main networks
        layers1 = [2] + n_hl * [n_unit] + [4 if is_basal else 3]
        p_u = init_single_net(k_u, layers1)
        # NOTE: κ scaling is NOT applied here — it is handled at runtime
        # by neural_net(params, x, scl=kappa, act_s=2)
        # Zero-initialize the final layer to prevent loss jump at stage start
        p_u[-1][0] = jnp.zeros_like(p_u[-1][0])
        p_u[-1][1] = jnp.zeros_like(p_u[-1][1])
        params_u.append(p_u)

        layers2 = [2] + n_hl * [n_unit] + [n_mu]
        p_mu = init_single_net(k_mu, layers2)
        p_mu[-1][0] = jnp.zeros_like(p_mu[-1][0])
        p_mu[-1][1] = jnp.zeros_like(p_mu[-1][1])
        params_mu.append(p_mu)

        if is_basal:
            layers3 = [2] + n_hl * [n_unit] + [1]
            p_c = init_single_net(k_c, layers3)
            p_c[-1][0] = jnp.zeros_like(p_c[-1][0])
            p_c[-1][1] = jnp.zeros_like(p_c[-1][1])
            params_c.append(p_c)
        else:
            params_c.append(None)

    return dict(net_u=params_u, net_mu=params_mu, net_c=params_c)
