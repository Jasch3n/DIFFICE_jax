import jax
import jax.numpy as jnp
import time

@jax.jit
def get_data(i):
    return jnp.array(i * 1.0), jnp.array(i * 2.0), jnp.array(i * 3.0), jnp.array(i * 4.0), jnp.array(i * 5.0)

print("Testing python loop with sync device_get...")
t0 = time.time()
try:
    scalars = []
    for i in range(20000):
        # We simulate the JIT call and immediately wait for the result
        res = get_data(i)
        scalars.append(jax.device_get(res))
    print(f"Time for sync loop: {time.time() - t0:.4f}s")
except Exception as e:
    print("Failed sync loop:", e)
