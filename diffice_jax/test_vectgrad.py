import sys
import jax
import jax.numpy as jnp
from jax import random

sys.path.append("/Users/jiapchen/Research/DIFFICE_jax")
from diffice_jax import init_pinn, solu_pinn
from diffice_jax.equation.eqn_iso import vectgrad

key = random.PRNGKey(0)
params = init_pinn(key, 2, 10, basal=True, embedding=True, embed_n=10)
solu = solu_pinn(scl=1.0, basal=True, embedding=True)
x = jnp.array([[0.1, 0.2]])

f = lambda z: solu(params, z)

try:
    grad, sol = vectgrad(f, x)
    print("vectgrad successful. Shape:", grad.shape)
except Exception as e:
    import traceback
    traceback.print_exc()

