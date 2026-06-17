import sys
import jax
import jax.numpy as jnp
from jax import random

sys.path.append("/Users/jiapchen/Research/DIFFICE_jax")
from diffice_jax import init_pinn, solu_pinn
from diffice_jax.equation.eqn_iso import gov_eqn as ssa_iso, front_eqn as dbc_iso
from diffice_jax.model.pinns.loss import loss_iso_create

key = random.PRNGKey(0)
trained_params = init_pinn(key, 2, 10, basal=True, embedding=True, embed_n=10)
pred_u = solu_pinn(scl=1.0, basal=True, embedding=True)

N = 10
X_star = [jnp.ones((N,2))]
U_star_u = jnp.ones((N,2))
U_star_h = jnp.ones((N,1))
U_star_s = jnp.ones((N,1))

X_ct = jnp.ones((N,2))
nnct = jnp.ones((N,2))
boundary_star = [jnp.ones((N,1)) for _ in range(4)]
scale = [[1, 1, 1, 1, 1], [1, 1, 1, 1]]
lw = [1.0, 1.0, 1.0]

def dataf(k):
    return {'smp': [X_star[0], U_star_u, X_star[0], U_star_h, U_star_s], 
            'col': [X_star[0]],
            'bd': [X_ct, boundary_star[0], boundary_star[2], boundary_star[3]]}

eqn_all = (ssa_iso, dbc_iso)
NN_loss = loss_iso_create(pred_u, eqn_all, scale, lw, basal=True)

try:
    loss, val = NN_loss(trained_params, dataf(key))
    print("Initial Data Loss:", val[1])
    print("Initial Eqn Loss:", val[2])
    print("Initial Bd Loss:", val[3])
except Exception as e:
    import traceback
    traceback.print_exc()

