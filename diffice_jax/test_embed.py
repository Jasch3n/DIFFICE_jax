import sys
import jax
import jax.numpy as jnp
from jax import random

sys.path.append("/Users/jiapchen/Research/DIFFICE_jax")
from diffice_jax import init_pinn, solu_pinn

key = random.PRNGKey(0)
params = init_pinn(key, 2, 10, basal=True, embedding=True, embed_n=10)
solu = solu_pinn(scl=1.0, basal=True, embedding=True)
x = jnp.array([[0.1, 0.2]])

try:
    y = solu(params, x)
    print("Forward pass successful. Shape:", y.shape)
except Exception as e:
    print("Error during forward pass:", e)

