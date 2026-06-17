
import time
import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
import optax
import diffice_jax.optimizer.optimization as opt_module

# Import from the codebase
from diffice_jax.model.pinns.initialization import init_nets as init_pinn
from diffice_jax.model.pinns.networks import solu_create as solu_pinn
from diffice_jax.equation.eqn_iso import gov_eqn as ssa_iso
from diffice_jax.equation.eqn_iso import front_eqn as dbc_iso
from diffice_jax.model.pinns.loss import loss_iso_create as loss_iso_pinn
from diffice_jax.optimizer.optimization import adam_minimizer

def run_repro():
    print("Setting up reproduction...")
    # Seed
    key = random.PRNGKey(0)
    keys = random.split(key, 10)

    # Config
    n_hl = 6
    n_unit = 40
    # Inputs typical for PINN
    # These are dummy values but the shapes matter
    n_smp = 8000
    nh_smp = 7500
    n_col = 8000
    n_cbd = 800
    
    # Fake data
    # Dimensions based on observation from code
    # x: (N, 2)
    # u: (N, 2)
    # xh: (N, 2)
    # h: (N, 1)
    
    x_smp = random.normal(keys[0], (n_smp, 2))
    u_smp = random.normal(keys[1], (n_smp, 2))
    
    xh_smp = random.normal(keys[2], (nh_smp, 2))
    h_smp = random.normal(keys[3], (nh_smp, 1))
    
    x_col = random.normal(keys[4], (n_col, 2))
    
    x_bd = random.normal(keys[5], (n_cbd, 2))
    nn_bd = random.normal(keys[6], (n_cbd, 2))
    
    # Scale: ((dmean, drange), ...)
    # dmean: lx0, ly0, u0, v0, h0
    dmean = jnp.array([1., 1., 1., 1., 1.])
    drange = jnp.array([1., 1., 1., 1., 1.])
    scale = (
        (dmean, drange), 
        None # unused in sample
    )
    
    # loss_fun takes data structure:
    # data['smp'] -> [x_smp, u_smp, xh_smp, h_smp]
    # data['col'] -> [x_col]
    # data['bd'] -> [x_bd, nn_bd]
    
    data = {
        'smp': [x_smp, u_smp, xh_smp, h_smp],
        'col': [x_col],
        'bd': [x_bd, nn_bd]
    }
    
    # Initialize network
    print("Initializing network...")
    # init_nets returns list of params
    params = init_pinn(keys[0], n_hl, n_unit, basal=False)
    
    # solu_create returns function
    pred_u = solu_pinn(basal=False)
    
    # Equations
    eqn_all = (ssa_iso, dbc_iso)
    lw = [0.05, 0.1] # equation weight, boundary weight
    
    print("Creating loss function...")
    # loss_iso_create returns callable loss_fun(params, data)
    loss_fn = loss_iso_pinn(pred_u, eqn_all, scale, lw, basal=False)
    
    # Set lref
    # We need to run it once to set lref usually, but let's just set it manually
    loss_fn.lref = 1.0
    
    # Optimizer
    opt_Adam = optax.adam(learning_rate=1e-3)
    opt_state = opt_Adam.init(params)
    
    # Helper to enforce compilation
    # We call adam_minimizer. Since it's jitted, the first call triggers compilation.
    print("Triggering compilation...")
    t0 = time.time()
    
    # Run one step
    # params, loss_info, opt_state = adam_minimizer(lossf, params, data, opt, opt_state)
    res = adam_minimizer(loss_fn, params, data, opt_Adam, opt_state)
    # block until ready to ensure compilation finished
    jax.block_until_ready(res[0]) 
    
    t1 = time.time()
    print(f"Compilation + Execution time: {t1 - t0:.4f} s")
    
    # Run second step (execution only)
    t2 = time.time()
    res = adam_minimizer(loss_fn, params, data, opt_Adam, opt_state)
    jax.block_until_ready(res[0])
    t3 = time.time()
    print(f"Execution time (2nd run): {t3 - t2:.4f} s")
    print(f"Estimated Compilation time: {(t1 - t0) - (t3 - t2):.4f} s")

if __name__ == "__main__":
    run_repro()
