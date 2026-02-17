import jax.numpy as jnp

# define the basic formation of neural network
def neural_net(params, x, scl, act_s=0, embedding=False):
    '''
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param sgn:  1 for even function and -1 for odd function
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    # choose the activation function
    actv = [jnp.tanh, jnp.sin][act_s]
    # normalize the input
    H = x  # input has been normalized
    
    # if using Fourier feature embedding
    if embedding:
        # separate the embedding layer (it is at the end)
        embed_layer = params[-1]
        # the rest are standard layers
        standard_params = params[:-1]
        first, *hidden, last = standard_params
        
        # get the B matrix
        B = embed_layer[0]  # shape (embed_n, 2)
        # calculate the projection
        # x shape (N, 2), B shape (embed_n, 2)
        # result shape (N, embed_n)
        # Use stop_gradient to prevent updating B
        from jax import lax
        B = lax.stop_gradient(B)
        x_proj = jnp.dot(x, B.T) * 2 * jnp.pi
        # calculate the embedding
        # result shape (N, 2 * embed_n)
        H = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    else:
        # separate the first, hidden and last layers
        first, *hidden, last = params
        
    # calculate the first layers output with right scale
    H = actv(jnp.dot(H, first[0]) * scl + first[1])
    # calculate the middle layers output
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    # no activation function for last layer
    var = jnp.dot(H, last[0]) + last[1]
    return var


# wrapper to create solution function with given domain size
def solu_create(scl=1, act_s=0, basal=False, embedding=False):
    '''
    :param scale: normalization info
    :return: function of the solution (a callable)
    '''
    def f(params, x, basal=basal):
        # print("DEBUG: solu_create thinks basal is", basal)
        # generate the NN
        uvh = neural_net(params[0], x, scl, act_s, embedding=embedding)
        mu = neural_net(params[1], x, scl, act_s, embedding=embedding)
        if basal:
            c = neural_net(params[2], x, scl, act_s, embedding=embedding)
            sol = jnp.hstack([uvh, jnp.exp(mu), jnp.exp(c)])
        else:
            sol = jnp.hstack([uvh, jnp.exp(mu)])
        return sol
    return f
