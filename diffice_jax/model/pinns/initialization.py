from jax import random
import jax.numpy as jnp

# initialize weights and biases of a single network
def init_single_net(parent_key, layer_widths, embedding=False, embed_n=5, embed_std=3.0):
    if embedding:
        # generate the key for embedding
        key_embed, parent_key = random.split(parent_key)

    params = []
    keys = random.split(parent_key, num=len(layer_widths) - 1)
    # create the weights and biases for the network
    for in_dim, out_dim, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = random.split(key)
        xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
        params.append(
            [random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev,
             random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * 0]
        )
        
    # if using Fourier feature embedding, append it at the end
    if embedding:
        # generate the B matrix for embedding
        # shape is (embed_n, 2)
        B = random.normal(key_embed, shape=(embed_n, 2)) * embed_std
        # create a dummy bias for the embedding layer
        params.append([B, jnp.zeros(embed_n)])
    return params


# generate weights and biases for all networks required in the problem
def init_nets(parent_key, n_hl, n_unit, aniso=False, basal=False, embedding=False, embed_n=10, embed_std=1.0):
    '''
    :param n_hl: number of hidden layers [int]
    :param n_unit: number of units in each layer [int]
    '''
    # set the default number of output for viscosity
    n_mu = 1
    n_basal=0
    # for anisotropic model
    if aniso:
        # number of viscosity output is 2
        n_mu = 2
    elif basal:
        n_basal=1

    # set the neural network shape for u, v, h, and s
    if embedding:
        # if using embedding, the input dimension for the first trainable layer is 2 * embed_n
        first_layer_dim = 2 * embed_n
    else:
        first_layer_dim = 2

    layers1 = [first_layer_dim] + n_hl * [n_unit] + [4 if basal else 3]
    # set the neural network shape for mu
    layers2 = [first_layer_dim] + n_hl * [n_unit] + [n_mu]
    # if inferring for basal friction, add another layer for C
    if basal:
        layers3 = [first_layer_dim] + n_hl * [n_unit] + [n_basal]

    if aniso and basal:
        print("Warning: Inferring basal friction with anisotropic visocsity, the inverse problem is underdetermined.")

    # generate the random key for each network
    keys = random.split(parent_key, 2)
    # generate weights and biases for
    params_u = init_single_net(keys[0], layers1, embedding=embedding, embed_n=embed_n, embed_std=embed_std)
    params_mu = init_single_net(keys[0], layers2, embedding=embedding, embed_n=embed_n, embed_std=embed_std)
    if basal:
        params_c = init_single_net(keys[0], layers3, embedding=embedding, embed_n=embed_n, embed_std=embed_std)
        return [params_u, params_mu, params_c]
    else:
        return [params_u, params_mu]
