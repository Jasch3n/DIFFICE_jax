from jax import random
import jax.numpy as jnp


class XPINNParams(dict):
    """Parameter mapping with a legacy length for all-floating XPINNs."""

    def __len__(self):
        if "net_c" in self and all(params is None for params in self["net_c"]):
            return 2
        return super().__len__()

# initialize weights and biases of a single network
def init_single_net(parent_key, layer_widths, embedding=False, embed_n=5, embed_std=3.0):
    if embedding:
        key_embed, parent_key = random.split(parent_key)

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

    if embedding:
        B = random.normal(key_embed, shape=(embed_n, 2)) * embed_std
        params.append([B, jnp.zeros(embed_n)])
    return params


def normalize_embedding_config(embedding_config, n_sub):
    """Normalize optional per-region Fourier-feature embedding settings."""

    if embedding_config is None:
        return [dict(embedding_u=False, embedding_mu=False, embedding_c=False, embed_n=10, embed_std=1.0) for _ in range(n_sub)]

    if len(embedding_config) != n_sub:
        raise ValueError("embedding_config must have one entry per sub-region")

    configs = []
    for config in embedding_config:
        embedding = bool(config.get('embedding', False))
        embedding_u = bool(config.get('embedding_u', embedding))
        embedding_mu = bool(config.get('embedding_mu', embedding))
        embedding_c = bool(config.get('embedding_c', embedding))
        embed_n = int(config.get('embed_n', 10))
        embed_std = float(config.get('embed_std', 1.0))
        configs.append(dict(embedding_u=embedding_u, embedding_mu=embedding_mu,
                            embedding_c=embedding_c, embed_n=embed_n, embed_std=embed_std))
    return configs


def normalize_network_config(network_config, n_sub, n_hl, n_unit):
    """Normalize optional per-region network sizes.

    ``net_c0`` is accepted as a legacy alias, but XPINN initialization now
    returns a single basal-friction network family named ``net_c``.
    """

    default = dict(depth=int(n_hl), width=int(n_unit))
    if network_config is None:
        return [
            dict(net_u=default, net_mu=default, net_c=default)
            for _ in range(n_sub)
        ]

    if len(network_config) != n_sub:
        raise ValueError("network_config must have one entry per sub-region")

    def normalize_net_config(config):
        return dict(
            depth=int(config.get('depth', default['depth'])),
            width=int(config.get('width', default['width'])),
        )

    configs = []
    for config in network_config:
        configs.append(dict(
            net_u=normalize_net_config(config.get('net_u', default)),
            net_mu=normalize_net_config(config.get('net_mu', default)),
            net_c=normalize_net_config(config.get('net_c', config.get('net_c0', default))),
        ))
    return configs


# generate weights and biases for all networks required in the XPINNs problem
def init_nets(parent_key, n_hl, n_unit, n_sub=1, aniso=False, basal_mask=None,
              embedding_config=None, network_config=None):
    """Initialize all XPINN network families for each sub-region."""

    if basal_mask is None:
        basal_mask = [False] * n_sub

    embedding_config = normalize_embedding_config(embedding_config, n_sub)
    network_config = normalize_network_config(network_config, n_sub, n_hl, n_unit)

    # set the default number of output for viscosity
    n_mu = 1
    # for anisotropic model
    if aniso:
        # number of viscosity output is 2
        n_mu = 2

    sub_keys = random.split(parent_key, n_sub)

    params_u = []
    params_mu = []
    params_c = []

    for i, key in enumerate(sub_keys):
        k_u, k_mu, k_c = random.split(key, 3)
        is_basal = basal_mask[i]
        embed_cfg = embedding_config[i]
        net_cfg = network_config[i]
        embedding_u = embed_cfg['embedding_u']
        embedding_mu = embed_cfg['embedding_mu']
        embedding_c = embed_cfg['embedding_c']
        embed_n = embed_cfg['embed_n']
        embed_std = embed_cfg['embed_std']
        first_layer_dim_u = 2 * embed_n if embedding_u else 2
        first_layer_dim_mu = 2 * embed_n if embedding_mu else 2
        first_layer_dim_c = 2 * embed_n if embedding_c else 2

        # set the neural network shape for u, v, h (and s if basal)
        cfg_u = net_cfg['net_u']
        layers1 = [first_layer_dim_u] + cfg_u['depth'] * [cfg_u['width']] + [4]
        params_u.append(init_single_net(k_u, layers1, embedding=embedding_u, embed_n=embed_n, embed_std=embed_std))

        # set the neural network shape for mu
        cfg_mu = net_cfg['net_mu']
        layers2 = [first_layer_dim_mu] + cfg_mu['depth'] * [cfg_mu['width']] + [n_mu]
        params_mu.append(init_single_net(k_mu, layers2, embedding=embedding_mu, embed_n=embed_n, embed_std=embed_std))

        if is_basal:
            # set the neural network shape for c (basal friction)
            cfg_c = net_cfg['net_c']
            layers3 = [first_layer_dim_c] + cfg_c['depth'] * [cfg_c['width']] + [1]
            params_c.append(init_single_net(k_c, layers3, embedding=embedding_c, embed_n=embed_n, embed_std=embed_std))
        else:
            params_c.append(None)

    return XPINNParams(net_u=params_u, net_mu=params_mu, net_c=params_c)
