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
