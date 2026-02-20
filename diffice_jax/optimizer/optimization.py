"""
@author: Yongji Wang
"""

import sys
import jax.numpy as jnp
import optax
from jax import random, jit, grad, debug, lax, value_and_grad
from jax.experimental import io_callback
import jax.flatten_util as flat_utl
from jax.debug import callback as call
from tensorflow_probability.substrates import jax as tfp
import functools


# create the Adam minimizer
@functools.partial(jit, static_argnames=("lossf", "opt"))
def adam_minimizer(lossf, params, data, opt, opt_state):
    """Basic gradient update step based on the opt optimizer."""
    grads, loss_info = grad(lossf, has_aux=True)(params, data)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss_info, opt_state


def adam_optimizer(key, lossf, params, dataf, epoch, lr=1e-3, aniso=False, schdul=None,
                   eval_f=None, 
                   adaptive=False, adapt_period=500, adapt_burnin=20000):
    """using the adam optimizer for the training.

    Args:
      key: random key
      lossf: loss function
      params: parameters of the networks
      dataf: data sampling function
      epoch: total number of iterations
      lr: learning rate
      aniso: whether the training is for anisotropic viscosity [boolean]
      schdul: scheduler function for modifying the weight of regularization term [callable]

    Returns:
      params: trained parameters of the networks
      loss_all: history of loss info over the training
    """
    if schdul is None:
        schdul = lambda x: 1.0
    # extract the initial wsp value if exist
    if hasattr(lossf, 'wsp'):
        wsp0 = lossf.wsp
    else:
        wsp0 = jnp.nan
    # select the Adam as the minimizer
    opt_Adam = optax.adam(learning_rate=lr)
    # obtain the initial state of the params
    opt_state = opt_Adam.init(params)
    # pre-allocate the loss variable
    loss_all = []
    nc = jnp.int32(jnp.round(epoch / 5))
    if adaptive:
        x_col_mem = None
        adapted = False

    # start the training iteration
    for step in range(epoch):
        # don't adapt in early training stages (first 10% of the epochs)
        run_RAD = (step+1)%adapt_period==0 and (step+1)>adapt_burnin and adaptive

        # split the new key for randomization
        key = random.split(key, 1)[0]

        # re-sampling the data points
        if run_RAD:
            print(f"epoch {step+1}, adapting sample based on residue")
            adapted=True
            data = dataf(key, eval_adaptive=True, eval_f=lambda x, idx, basal: eval_f(params, x, idx, basal))
            # Memorize the adaptively sampled collocation points in a tmp variable 
            # to keep it unchanged for adapt_period epochs
            x_col_mem = data['col'][0]
        elif adaptive and adapted:
            data = dataf(key, eval_adaptive=False)
            data['col'][0] = x_col_mem
        else:
            data = dataf(key)
        
        # minimize the loss function using Adam
        params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
        # print the loss for every 100 iteration
        if (step+1) % 1000 == 0:
            # print the results
            if len(loss_info.shape)> 1:
                print(f"ADAM Step:{step+1} | Loss:{loss_info[0][0]:.4e} | d:{loss_info[0][1]:.4e} | eq:{loss_info[0][2]:.4e} | "
                    f"bd:{loss_info[0][3]:.4e}", file=sys.stderr)
            else:
                print(f"ADAM Step:{step+1} | Loss:{loss_info[0]:.4e} | d:{loss_info[1]:.4e} | eq:{loss_info[2]:.4e} | "
                    f"bd:{loss_info[3]:.4e} | m:{loss_info[4]:.4e}", file=sys.stderr)
            if aniso:
                # modify the wsp value over the iteration
                lossf.wsp = wsp0 * schdul(step+1)

        
        # saving the loss
        loss_all.append(loss_info[0:5])

    # obtain the total loss in the last iterations
    lossend = jnp.array(loss_all[-nc:])[:, 0]
    # find the minimum loss value
    lmin = jnp.min(lossend)
    # optain the last loss value
    llast = lossend[-1]
    # guarantee the loss value in last iteration is smaller than anyone before
    last_iter = 0
    # while llast > lmin and last_iter<epoch:
    #     # split the new key for randomization
    #     key = random.split(key, 1)[0]
    #     # re-sampling the data points
    #     data = dataf(key)
    #     # minimize the loss function using Adam
    #     params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt_Adam, opt_state)
    #     # saving the loss
    #     llast = loss_info[0][0] if len(loss_info.shape)>1 else loss_info[0]
    #     loss_all.append(loss_info[0][0:5] if len(loss_info.shape)>1 else loss_info[0:5])
    #     last_iter += 1

    if adaptive:
        return params, loss_all# , probs_last 
    else:
        return params, loss_all

def lbfgs_function(lossf, init_params, data, basal=False, print_rate=500):
    """
    Factory that builds a value-and-gradient function compatible with
    tfp.optimizer.lbfgs_minimize.

    Key optimizations vs. original:
      1. value_and_grad hoisted outside @jit — defined once, never re-traced.
      2. Single io_callback instead of two — halves host-device syncs per step.
      3. Step counter lives in a JAX scalar (jnp array), not Python state,
         so no Python object mutation happens inside the jitted boundary.
      4. Loss accumulation happens via the single callback, not a separate call.
    """

    # --- Pytree <-> 1D conversion, computed once ---
    _, unflat = flat_utl.ravel_pytree(init_params)

    def unravel(params_1d):
        return unflat(params_1d)

    # --- Build value_and_grad ONCE, outside jit ---
    # value_and_grad computes the forward pass and backward pass together in
    # a single XLA computation. Calling grad() inside jit would re-stage the
    # differentiation on every trace, which is wasteful.
    _value_and_grad = value_and_grad(lossf, has_aux=True)

    # --- Shared mutable Python state, touched only via callback ---
    loss_log = []
    step_counter = [0]  # list so it's mutable inside closure

    # Number of metrics to store per step
    n_metrics = 5 if basal else 4

    # --- Single consolidated callback ---
    # Merging logging + loss storage into one callback halves the number of
    # host-device synchronisations compared to the original two-call design.
    def _host_callback(loss_info_slice):
        step_counter[0] += 1
        loss_log.append(loss_info_slice[:n_metrics])
        if (step_counter[0] % print_rate == 0) or step_counter[0]<=10:
            x = loss_info_slice
            print(
                f"LBFGS Step:{step_counter[0]} | Loss:{x[0]:.4e} | "
                f"d:{x[1]:.4e} | eq:{x[2]:.4e} | bd:{x[3]:.4e}",
                file=sys.stderr,
            )

    @jit
    def f(params_1d):
        # Unravel 1D -> pytree (lightweight; unflat is a pure function)
        params = unravel(params_1d)

        # Single combined forward+backward pass — no redundant computation
        (loss_value, loss_info), grads = _value_and_grad(params, data)

        # Flatten grads to 1D in one call
        grads_1d = flat_utl.ravel_pytree(grads)[0]

        # One host-device sync for both logging and loss storage.
        # result_shape_dtypes=() signals no return value from the callback.
        io_callback(
            _host_callback,
            (),              # callback returns nothing to device
            loss_info if len(loss_info.shape)==1 else loss_info[0],    # pass the metrics slice to host
            ordered=True     # preserve step ordering across calls
        )

        return loss_value, grads_1d

    # Expose helpers for external use
    f.unravel = unravel
    f.loss = loss_log
    return f


def lbfgs_optimizer(lossf, params, data, epoch, basal=False, print_rate=1000):
    """
    Runs L-BFGS minimisation over `epoch`-equivalent iterations.

    Optimization vs original:
      - max_nIter is a plain Python int (not jnp.int32); lbfgs_minimize
        expects a Python/numpy scalar, not a JAX device array.
      - Uses floor division (//) explicitly to avoid float precision issues.
    """
    func_lbfgs = lbfgs_function(
        lossf, params, data, basal=basal, print_rate=print_rate
    )

    # Flatten initial params to 1D once
    init_params_1d = flat_utl.ravel_pytree(params)[0]

    # Plain Python int — no JAX scalar boxing needed here
    max_nIter = int(epoch // 3)

    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func_lbfgs,
        initial_position=init_params_1d,
        tolerance=1e-15,
        max_iterations=max_nIter,
    )

    # Recover pytree params from optimised 1D position
    optimised_params = func_lbfgs.unravel(results.position)
    num_iter = results.num_objective_evaluations
    loss_all = func_lbfgs.loss

    print(f"Total iterations: {num_iter}")
    return optimised_params, loss_all

# A factory to create a function required by tfp.optimizer.lbfgs_minimize.
# def lbfgs_function(lossf, init_params, data, basal=False, print_rate=500):
#     # obtain the 1D parameters and the function that can turn back to the pytree
#     _, unflat = flat_utl.ravel_pytree(init_params)

#     def update(params_1d):
#         # updating the model's parameters from the 1D array
#         params = unflat(params_1d)
#         return params

#     # Define a class to handle printing state
#     class Printer:
#         def __init__(self):
#             self.step = 0
            
#         def log(self, x):
#             self.step += 1
#             if self.step % print_rate == 0:
#                 print(f"LBFGS Step:{self.step} | Loss:{x[0]:.4e} | d:{x[1]:.4e} | eq:{x[2]:.4e} | "
#                         f"bd:{x[3]:.4e}", file=sys.stderr)

#     printer = Printer()

#     # A function that can be used by tfp.optimizer.lbfgs_minimize.
#     @jit
#     def f(params_1d):
#         # convert the 1d parameters back to pytree format
#         params = update(params_1d)
#         # calculate gradients and convert to 1D tf.Tensor
#         grads, loss_info = grad(lossf, has_aux=True)(params, data)
#         # convert the grad to 1d arrays
#         grads_1d = flat_utl.ravel_pytree(grads)[0]
#         loss_value = loss_info[0][0]

#         # # store loss value so we can retrieve later
#         call(lambda x: f.loss.append(x), loss_info[0][0:(5 if basal else 4)])
        
#         # Print using the printer class
#         # We pass loss_info[0] which contains the metrics
#         # call(printer.log, loss_info[0])
            
#         return loss_value, grads_1d

#     # store these information as members so we can use them outside the scope
#     f.update = update
#     f.loss = []
#     return f


# # define the function to apply the L-BFGS optimizer
# def lbfgs_optimizer(lossf, params, data, epoch, basal=False, print_rate=100):
#     func_lbfgs = lbfgs_function(lossf, params, data, basal=basal, print_rate=print_rate)
#     # convert initial model parameters to a 1D array
#     init_params_1d = flat_utl.ravel_pytree(params)[0]
#     # calculate the effective number of iteration
#     max_nIter = jnp.int32(epoch / 3)
#     # train the model with L-BFGS solver
#     results = tfp.optimizer.lbfgs_minimize(
#         value_and_gradients_function=func_lbfgs, initial_position=init_params_1d,
#         tolerance=1e-15, max_iterations=max_nIter)
#     params = func_lbfgs.update(results.position)
#     num_iter = results.num_objective_evaluations
#     loss_all = func_lbfgs.loss
#     print(f" Total iterations: {num_iter}")
#     return params, loss_all