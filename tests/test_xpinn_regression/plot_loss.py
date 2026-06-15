import matplotlib.pyplot as plt
import pickle, os, sys
import jax
import jax.numpy as jnp 
from jax import random

from xpinn_regression import load_data, initialize_xpinn, initialize_loss
from xpinn_regression import DataOutput, XPINNOutput, LossOutput

TEST_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CKPT_PATH = os.path.join(TEST_FOLDER, 'checkpoints')

with open(os.path.join(TEST_FOLDER, 'checkpoints', 'loss_history.pkl'), 'rb') as f:
    loss_history = pickle.load(f)

plt.figure()    
plt.semilogy(loss_history, 'k-')
plt.grid(which='both', ls='--')
plt.show()