import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import time
from scipy.io import savemat, loadmat
from pathlib import Path
import pickle

from diffice_jax import normdata_pinn, dsample_pinn
from diffice_jax import ssa_iso, dbc_iso
from diffice_jax import init_pinn, solu_pinn
from diffice_jax import loss_iso_pinn
from diffice_jax import predict_pinn
from diffice_jax import adam_opt, lbfgs_opt

# find the root directory
rootdir = Path(__file__).parent

#%% setting hyper-parameters

# select the random seed
seed = np.random.choice(3000, 1)[0]
key = random.PRNGKey(seed)
np.random.seed(seed)

# create the subkeys
keys = random.split(key, 4)

# select the size of neural network
n_hl = 6
n_unit = 40
# set the weight for 1. equation loss and 2. boundary condition loss
lw = [0.05, 0.1]

# number of sampling points
n_smp = 8000    # for velocity data
nh_smp = 7500   # for thickness data
n_col = 8000    # for collocation points
n_cbd = 800     # for boundary condition (calving front)
# group all the number of points
n_pt = jnp.array([n_smp, nh_smp, n_col, n_cbd], dtype='int32')
# double the points for L-BFGS training
n_pt2 = n_pt * 2


#%% data loading

# select the ice shelf for the training
shelfname = 'Amery'

# create the dataset filename
filename = 'data_pinns_' + shelfname + '.mat'
filepath = str(rootdir.joinpath('real_data').joinpath(filename))

# create the output file name
outputName = shelfname + f'_pinns_iso_seed={seed:.0f}'
# check whether sub-folder exists
outdir = rootdir.joinpath('results_' + shelfname)
isExist = os.path.exists(outdir)
# create the sub-folder if not exist
if not isExist:
    os.mkdir(outdir)

# load the datafile
rawdata = loadmat(filepath)
# obtain the data for training
data_all = normdata_pinn(rawdata)
scale = data_all[4][0:2]


#%% initialization

# initialize the weights and biases of the network
trained_params = init_pinns(keys[0], n_hl, n_unit)

# create the solution function
pred_u = solu_pinn()

# create the data function for Adam
dataf = dsample_pinn(data_all, n_pt)
keys_adam = random.split(keys[1], 5)
data = dataf(keys_adam[0])
# create the data function for L-BFGS
dataf_l = dsample_pinn(data_all, n_pt2)
key_lbfgs = keys[2]

# group the gov. eqn and bd cond.
eqn_all = (ssa_iso, dbc_iso)
# calculate the loss function
NN_loss = loss_iso_pinn(pred_u, eqn_all, scale, lw)
# calculate the initial loss and set it as the loss reference value
NN_loss.lref = NN_loss(trained_params, data)[0]


#%% networks training

# set the training iteration
epoch1 = 100000
epoch2 = 100000
# (above is the number of iterations required for high accuracy,
#  users are free to modify it based on your need)

# set the learning rate for Adam
lr = 1e-3
# training the neural network
start_time = time.time()

"""training with Adam"""
trained_params, loss1 = adam_opt(keys_adam[0], NN_loss, trained_params, dataf, epoch1, lr=lr)

# sample the data for L-BFGS training
data_l = dataf_l(key_lbfgs)
"""training with L-BFGS"""
trained_params, loss2 = lbfgs_opt(NN_loss, trained_params, data_l, epoch2)



kfac_config = dict(
    # When learning_rate, momentum, and damping are each set to None
    # this enables the respective adaptive methods for them.
    # 1e-4, 0.9, and 1e-4 are sensible values, but the adaptive
    # methods tend to gives better results. Only the damping
    # adaptation method benefits from tuning
    learning_rate=None,
    momentum=None,
    damping=jnp.nan,
    norm_constraint=1e-8,  # ignored when LR or momentum adaptation is used
    initial_damping=1e-0,  # used by the adaptive damping method
    min_damping=1e-14,  # used by the adaptive damping method
    curvature_block_type="naive_full",  # "naive_full" (very important setting)
    damping_adaptation_decay=0.998,
    curvature_ema=0.998,
    inverse_update_period=1,
    num_burnin_steps=0,  # TODO(jamesmartens): experiment with this
    always_use_exact_qmodel_for_damping_adjustment=True,
    include_norms_in_stats=True
    )


def kfac_optimizer(rng, lossf, config, params, dataf, epoch):
    optim = KfacOptimizer(
        loss_fn=lossf, **config).get_optimizer()

    rng, key = jax.random.split(rng)
    data = dataf(key)
    opt_state = optim.init(params, key, data)
    loss_all = []
    losse_all = []
    damping = config['initial_damping']
    damping_decay = config['damping_adaptation_decay']
    damping_min = config['min_damping']

    # start the training iteration
    for step in range(epoch):
        rng, *keys = jax.random.split(rng, 3)
        params, opt_state, stats = optim.step(
            params, opt_state, keys[0], batch=data, damping=damping, global_step_int=step)

        loss_info = stats['aux']
        loss_all.append(loss_info)
        losse_all.append(loss_info[2])

        if (step+1) % 50 == 0:
            dmp = stats['damping']
            print(f"Step: {step+1} | Loss: {loss_info[0]:.4e} | Loss_d: {loss_info[1]:.4e} | "
                  f"Loss_e: {loss_info[2]:.4e} | Dp: {dmp:.4e}", file=sys.stderr)


        if damping > damping_min:
            damping *= damping_decay

    return params, loss_all


# compute the total time of training
elapsed = time.time() - start_time
print('Training time: %.4f' % elapsed, file=sys.stderr)


#%% network saving

FileName = outputName + '.pkl'
FilePath = str(outdir.joinpath(FileName))
with open(FilePath, 'wb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it. However, users should have the same version of JAX
    # to load the data correctly.
    pickle.dump(trained_params, f, pickle.HIGHEST_PROTOCOL)


#%% prediction

# create the function for trained solution and equation residues
f_u = lambda x: pred_u(trained_params, x)
f_gu = lambda x: jax.vmap(jax.jacfwd(lambda z: f_u(z[None, :])[0]))(x).reshape(x.shape[0], -1)[:, 0:6]
# group all the function
func_all = (f_u, f_gu, gov_eqn)
# calculate the solution and equation residue at given grids for visualization
results = predict_pinn(func_all, data_all)

# generate the last loss
loss_all = jnp.array(loss1 + loss2)
# save the loss info into results
results['loss'] = loss_all


#%% output saving

# save the output into .mat file
FileName = outputName + '.mat'
FilePath = str(outdir.joinpath(FileName))
savemat(FilePath, results)
