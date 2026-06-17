import numpy as np
import jax
from jax import lax, random
import jax.numpy as jnp


def resolve_architecture(architecture=None, use_modified_mlp=False):
    """Resolve legacy and explicit architecture flags to a canonical backbone name."""
    if architecture is None:
        return "modified_mlp" if use_modified_mlp else "mlp"
    if architecture not in ("mlp", "modified_mlp", "pirate"):
        raise ValueError(f"Unsupported architecture '{architecture}'")
    return architecture


def _init_dense_layer(parent_key, in_dim, out_dim, use_rwf=False, bias_scale=0.0):
    """Initialize one dense layer, optionally using random weight factorization."""
    weight_key, scale_key, bias_key = random.split(parent_key, 3)
    xavier_stddev = jnp.sqrt(2 / (in_dim + out_dim))
    b = random.truncated_normal(bias_key, -2, 2, shape=(out_dim,)) * bias_scale
    if use_rwf:
        # Match JaxPI's RWF parameterization: W = V * g with one scale per output channel.
        mean = 1.0
        stddev = 0.1
        g = jnp.exp(random.normal(scale_key, shape=(out_dim,)) * stddev + mean)
        w = random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev
        v = w / g[None, :]
        return [v, g, b]
    v = random.truncated_normal(weight_key, -2, 2, shape=(in_dim, out_dim)) * xavier_stddev
    return [v, b]


def _apply_dense_layer(layer, x, use_rwf=False, scale=1.0):
    """Apply a dense layer to batched inputs, with optional RWF scaling."""
    if use_rwf:
        v, g, b = layer
        if g.ndim == 1 and g.shape[0] == v.shape[1]:
            kernel = v * g[None, :]
        else:
            raise ValueError(
                f"Unsupported RWF scale shape {g.shape} for weight shape {v.shape}"
            )
        return jnp.dot(x, kernel) * scale + b
    return jnp.dot(x, layer[0]) * scale + layer[1]


def _zero_dense_layer(layer, use_rwf=False):
    """Return a copy of a dense layer with zeroed trainable weights and biases."""
    layer_zero = list(layer)
    layer_zero[0] = jnp.zeros_like(layer_zero[0])
    if use_rwf:
        layer_zero[2] = jnp.zeros_like(layer_zero[2])
    else:
        layer_zero[1] = jnp.zeros_like(layer_zero[1])
    return layer_zero


def _validate_modified_widths(layer_widths):
    """Ensure the modified-MLP backbone uses a uniform hidden width."""
    if len(layer_widths) < 3:
        raise ValueError("Modified MLP requires at least one hidden layer.")
    hidden_widths = layer_widths[1:-1]
    if any(w != hidden_widths[0] for w in hidden_widths):
        raise ValueError("Modified MLP requires uniform hidden-layer width.")


def _validate_pirate_widths(layer_widths):
    """Ensure the PirateNet backbone uses a uniform residual width."""
    if len(layer_widths) < 3:
        raise ValueError("PirateNet requires at least one residual block.")
    hidden_widths = layer_widths[1:-1]
    if any(w != hidden_widths[0] for w in hidden_widths):
        raise ValueError("PirateNet requires uniform hidden-layer width.")


def _make_embedding(parent_key, embed_n, embed_std):
    """Initialize a random Fourier feature embedding matrix."""
    b = random.normal(parent_key, shape=(2, embed_n)) * embed_std
    return [b, jnp.zeros(embed_n)]


def _apply_embedding(x, b):
    """Map 2D coordinates into Fourier features using a fixed embedding matrix."""
    x_proj = jnp.dot(x, b)
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


def _init_standard_network(
    parent_key,
    layer_widths,
    embedding=False,
    embed_n=5,
    embed_std=3.0,
    use_rwf=False,
    use_modified_mlp=False,
    zero_last_when_embedding=False,
):
    """Initialize parameters for the plain MLP or modified-MLP backbones."""
    if embedding:
        key_embed, parent_key = random.split(parent_key)

    params = []

    if use_modified_mlp:
        _validate_modified_widths(layer_widths)
        in_dim = layer_widths[0]
        hidden_dim = layer_widths[1]
        k_u, k_v, parent_key = random.split(parent_key, 3)
        params.append(_init_dense_layer(k_u, in_dim, hidden_dim, use_rwf=use_rwf, bias_scale=0.0))
        params.append(_init_dense_layer(k_v, in_dim, hidden_dim, use_rwf=use_rwf, bias_scale=0.0))

    keys = random.split(parent_key, num=len(layer_widths) - 1)
    for in_dim, out_dim, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        params.append(_init_dense_layer(key, in_dim, out_dim, use_rwf=use_rwf, bias_scale=0.0))

    if embedding:
        if zero_last_when_embedding:
            params[-1] = _zero_dense_layer(params[-1], use_rwf=use_rwf)
        params.append(_make_embedding(key_embed, embed_n, embed_std))

    return params


def _init_pirate_network(
    parent_key,
    layer_widths,
    embedding=False,
    embed_n=5,
    embed_std=3.0,
    use_rwf=False,
):
    """Initialize parameters for the PirateNet backbone."""
    _validate_pirate_widths(layer_widths)
    if embedding:
        key_embed, parent_key = random.split(parent_key)

    state_dim = layer_widths[0]
    hidden_dim = layer_widths[1]
    out_dim = layer_widths[-1]
    n_blocks = len(layer_widths) - 2

    keys = random.split(parent_key, 2 + 3 * n_blocks + 1)
    k_u, k_v = keys[0], keys[1]
    block_keys = keys[2:-1]
    k_final = keys[-1]

    blocks = []
    for i in range(n_blocks):
        k1, k2, k3 = block_keys[3 * i : 3 * (i + 1)]
        blocks.append(
            {
                "layers": [
                    _init_dense_layer(k1, state_dim, hidden_dim, use_rwf=use_rwf, bias_scale=0.0),
                    _init_dense_layer(k2, hidden_dim, hidden_dim, use_rwf=use_rwf, bias_scale=0.0),
                    _init_dense_layer(k3, hidden_dim, state_dim, use_rwf=use_rwf, bias_scale=0.0),
                ],
                "alpha": jnp.zeros((1,)),
            }
        )

    params = {
        "enc_u": _init_dense_layer(k_u, state_dim, hidden_dim, use_rwf=use_rwf, bias_scale=0.0),
        "enc_v": _init_dense_layer(k_v, state_dim, hidden_dim, use_rwf=use_rwf, bias_scale=0.0),
        "blocks": blocks,
        "final": _init_dense_layer(k_final, state_dim, out_dim, use_rwf=use_rwf, bias_scale=0.0),
    }
    if embedding:
        params["embedding"] = _make_embedding(key_embed, embed_n, embed_std)
    return params


def init_network_params(
    parent_key,
    layer_widths,
    embedding=False,
    embed_n=5,
    embed_std=3.0,
    use_rwf=False,
    use_modified_mlp=False,
    zero_last_when_embedding=False,
    architecture=None,
):
    """Initialize one network parameter tree for the requested architecture."""
    architecture = resolve_architecture(architecture=architecture, use_modified_mlp=use_modified_mlp)
    if architecture == "pirate":
        return _init_pirate_network(
            parent_key=parent_key,
            layer_widths=layer_widths,
            embedding=embedding,
            embed_n=embed_n,
            embed_std=embed_std,
            use_rwf=use_rwf,
        )
    return _init_standard_network(
        parent_key=parent_key,
        layer_widths=layer_widths,
        embedding=embedding,
        embed_n=embed_n,
        embed_std=embed_std,
        use_rwf=use_rwf,
        use_modified_mlp=(architecture == "modified_mlp"),
        zero_last_when_embedding=zero_last_when_embedding,
    )


def _split_standard_params(params, embedding=False, use_modified_mlp=False):
    """Split standard-network params into encoders, trainable layers, and embedding."""
    core = params
    b = None
    if embedding:
        b = core[-1][0]
        core = core[:-1]

    enc_u = None
    enc_v = None
    if use_modified_mlp:
        enc_u, enc_v = core[0], core[1]
        core = core[2:]
    return enc_u, enc_v, core, b


def _prepare_input(params, x, embedding=False, architecture="mlp"):
    """Prepare model input by applying Fourier embedding when enabled."""
    if architecture == "pirate":
        embed_layer = params.get("embedding")
        if embedding and embed_layer is not None:
            return _apply_embedding(x, embed_layer[0])
        return x

    _, _, _, b = _split_standard_params(
        params, embedding=embedding, use_modified_mlp=(architecture == "modified_mlp")
    )
    if embedding and b is not None:
        return _apply_embedding(x, b)
    return x


def _apply_standard_network(
    params,
    x,
    scl=1.0,
    first_act=jnp.tanh,
    rest_act=jnp.tanh,
    embedding=False,
    use_rwf=False,
    architecture="mlp",
):
    """Run the forward pass for the plain MLP or modified-MLP backbones."""
    use_modified_mlp = architecture == "modified_mlp"
    enc_u, enc_v, layers, _ = _split_standard_params(
        params, embedding=embedding, use_modified_mlp=use_modified_mlp
    )
    h = _prepare_input(params, x, embedding=embedding, architecture=architecture)

    if use_modified_mlp:
        u = first_act(_apply_dense_layer(enc_u, h, use_rwf=use_rwf))
        v = first_act(_apply_dense_layer(enc_v, h, use_rwf=use_rwf))

    first, *hidden, last = layers
    z = _apply_dense_layer(first, h, use_rwf=use_rwf, scale=scl)
    if use_modified_mlp:
        a = first_act(z)
        h = a * u + (1.0 - a) * v
    else:
        h = first_act(z)

    for layer in hidden:
        z = _apply_dense_layer(layer, h, use_rwf=use_rwf)
        if use_modified_mlp:
            a = rest_act(z)
            h = a * u + (1.0 - a) * v
        else:
            h = rest_act(z)

    out = _apply_dense_layer(last, h, use_rwf=use_rwf)
    return out


def pirate_latent_features(params, x, act=jnp.tanh, embedding=False, use_rwf=False):
    """Compute PirateNet latent features before the final linear readout."""
    state = _prepare_input(params, x, embedding=embedding, architecture="pirate")
    u = act(_apply_dense_layer(params["enc_u"], state, use_rwf=use_rwf))
    v = act(_apply_dense_layer(params["enc_v"], state, use_rwf=use_rwf))
    for block in params["blocks"]:
        identity = state
        z1 = act(_apply_dense_layer(block["layers"][0], state, use_rwf=use_rwf))
        z1 = z1 * u + (1.0 - z1) * v
        z2 = act(_apply_dense_layer(block["layers"][1], z1, use_rwf=use_rwf))
        z2 = z2 * u + (1.0 - z2) * v
        h_block = act(_apply_dense_layer(block["layers"][2], z2, use_rwf=use_rwf))
        alpha = jnp.reshape(block["alpha"], (1, 1))
        state = alpha * h_block + (1.0 - alpha) * identity
    return state


def _apply_pirate_network(
    params,
    x,
    first_act=jnp.tanh,
    embedding=False,
    use_rwf=False,
):
    """Run the forward pass for a PirateNet backbone."""
    latent = pirate_latent_features(
        params,
        x,
        act=first_act,
        embedding=embedding,
        use_rwf=use_rwf,
    )
    return _apply_dense_layer(params["final"], latent, use_rwf=use_rwf)


def apply_network(
    params,
    x,
    scl=1.0,
    first_act=jnp.tanh,
    rest_act=jnp.tanh,
    embedding=False,
    use_rwf=False,
    use_modified_mlp=False,
    architecture=None,
):
    """Apply the selected backbone to inputs and return network outputs."""
    architecture = resolve_architecture(architecture=architecture, use_modified_mlp=use_modified_mlp)
    if architecture == "pirate":
        return _apply_pirate_network(
            params=params,
            x=x,
            first_act=first_act,
            embedding=embedding,
            use_rwf=use_rwf,
        )
    return _apply_standard_network(
        params=params,
        x=x,
        scl=scl,
        first_act=first_act,
        rest_act=rest_act,
        embedding=embedding,
        use_rwf=use_rwf,
        architecture=architecture,
    )


def get_embedding_matrix(params, architecture=None, embedding=False, use_modified_mlp=False):
    """Extract the Fourier embedding matrix from a parameter tree when present."""
    architecture = resolve_architecture(architecture=architecture, use_modified_mlp=use_modified_mlp)
    if not embedding:
        return None
    if architecture == "pirate":
        embed_layer = params.get("embedding")
        return None if embed_layer is None else embed_layer[0]
    _, _, _, b = _split_standard_params(
        params, embedding=embedding, use_modified_mlp=(architecture == "modified_mlp")
    )
    return b


def scale_embedding_params(params, alpha, architecture=None, embedding=False, use_modified_mlp=False):
    """Return params with the embedding matrix scaled by a coarse-to-fine factor."""
    if alpha >= 1.0 or not embedding:
        return params
    architecture = resolve_architecture(architecture=architecture, use_modified_mlp=use_modified_mlp)
    if architecture == "pirate":
        if "embedding" not in params:
            return params
        out = dict(params)
        embed_layer = list(out["embedding"])
        embed_layer[0] = embed_layer[0] * alpha
        out["embedding"] = embed_layer
        return out
    if not isinstance(params, list) or len(params) == 0:
        return params
    net_scaled = list(params)
    embed_layer = net_scaled[-1]
    if not isinstance(embed_layer, (list, tuple)) or len(embed_layer) != 2:
        return params
    net_scaled[-1] = [embed_layer[0] * alpha, embed_layer[1]]
    return net_scaled


def get_last_trainable_layer(params, architecture=None, use_modified_mlp=False):
    """Return the final trainable affine layer for a parameter tree."""
    architecture = resolve_architecture(architecture=architecture, use_modified_mlp=use_modified_mlp)
    if architecture == "pirate":
        return params["final"]
    if not isinstance(params, list) or len(params) == 0:
        return params
    return params[-2] if isinstance(params[-1], (list, tuple)) and len(params[-1]) == 2 else params[-1]


def set_last_trainable_layer(params, layer, architecture=None, use_modified_mlp=False):
    """Return params with the final trainable affine layer replaced."""
    architecture = resolve_architecture(architecture=architecture, use_modified_mlp=use_modified_mlp)
    if architecture == "pirate":
        out = dict(params)
        out["final"] = layer
        return out
    out = list(params)
    idx = -2 if isinstance(out[-1], (list, tuple)) and len(out[-1]) == 2 else -1
    out[idx] = layer
    return out


def pirate_last_layer_least_squares(
    params,
    observations,
    embedding=False,
    use_rwf=False,
    act=jnp.tanh,
    ridge=1e-6,
):
    """Fit PirateNet's final linear layer to observed targets with ridge regression."""
    if not observations:
        return params

    final_layer = params["final"]
    weight = final_layer[0]
    bias = final_layer[2] if use_rwf else final_layer[1]
    hidden_dim, out_dim = weight.shape

    solved_weight = weight
    solved_bias = bias

    for x_obs, y_obs, out_idx in observations:
        if x_obs is None or y_obs is None or x_obs.shape[0] == 0:
            continue
        latent = pirate_latent_features(
            params,
            x_obs,
            act=act,
            embedding=embedding,
            use_rwf=use_rwf,
        )
        design = jnp.concatenate([latent, jnp.ones((latent.shape[0], 1))], axis=1)
        gram = design.T @ design + ridge * jnp.eye(design.shape[1])
        rhs = design.T @ y_obs
        if jax.default_backend() == "METAL":
            gram_np = np.asarray(gram)
            rhs_np = np.asarray(rhs)
            try:
                coeff_np = np.linalg.solve(gram_np, rhs_np)
            except np.linalg.LinAlgError:
                coeff_np = np.linalg.lstsq(gram_np, rhs_np, rcond=None)[0]
            coeff = jnp.asarray(coeff_np)
        else:
            coeff = jnp.linalg.solve(gram, rhs)

        solved_weight = solved_weight.at[:, out_idx].set(coeff[:-1, 0])
        solved_bias = solved_bias.at[out_idx].set(coeff[-1, 0])

    final_out = list(final_layer)
    final_out[0] = solved_weight
    if use_rwf:
        final_out[2] = solved_bias
    else:
        final_out[1] = solved_bias
    out = dict(params)
    out["final"] = final_out
    return out
