import sys
import os
import jax.numpy as jnp
import numpy as np
from jax import random, lax
import time
from scipy.io import savemat, loadmat
from pathlib import Path
import pickle

from diffice_jax import normdata_pinn, dsample_pinn
from diffice_jax import vectgrad, ssa_aniso, dbc_aniso
from diffice_jax import init_pinn, solu_pinn
from diffice_jax import loss_aniso_pinn
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
# set the weights for 1. equation loss, 2. boundary condition loss and 3. regularization loss
lw = [0.05, 0.1, 0.25]

# number of sampling points
n_smp = 7000    # for velocity data
nh_smp = 6500   # for thickness data
n_col = 7000    # for collocation points
n_cbd = 600     # for boundary condition (calving front)
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
outputName = shelfname + f'_pinns_aniso_seed={seed:.0f}'
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
trained_params = init_pinns(keys[0], n_hl, n_unit, aniso=True)

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
eqn_all = (gov_eqn, front_eqn)
# calculate the loss function
NN_loss = loss_aniso_pinn(pred_u, eqn_all, scale, lw)
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
# define the scheduler function for weight of the regularization loss
wsp_schdul = lambda x: lax.max(10 ** (-x/epoch1 * 2), 0.0125)
# training the neural network
start_time = time.time()

"""training with Adam"""
trained_params, loss1 = adam_opt(
    keys_adam[0], NN_loss, trained_params, dataf, epoch1, lr=lr, aniso=True, schdul=wsp_schdul)

# sample the data for L-BFGS training
data_l = dataf_l(key_lbfgs)
"""training with L-BFGS"""
trained_params, loss2 = lbfgs_opt(NN_loss, trained_params, data_l, epoch2)

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
f_gu = lambda x: vectgrad(f_u, x)[0][:, 0:6]
# group all the function
func_all = (f_u, f_gu, gov_eqn)
# calculate the solution and equation residue at given grids for visualization
results = predict_pinn(func_all, data_all, aniso=True)

# generate the last loss
loss_all = jnp.array(loss1 + loss2)
# save the loss info into results
results['loss'] = loss_all


#%% output saving

# save the output into .mat file
FileName = outputName + '.mat'
FilePath = str(outdir.joinpath(FileName))
savemat(FilePath, results)
