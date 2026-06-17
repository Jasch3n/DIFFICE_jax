import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import time
from jax import grad, value_and_grad

# Suppress warnings
jax.config.update("jax_enable_x64", False)

@jax.jit
def lbfgs_test(init_params, data):
    # simulate a simple neural network loss
    def lossf(p, d):
        # A simple MLP on data
        # Let's say p is a flat array, we reshape it to 2 matrices for a 1-hidden layer MLP
        w1 = p[:10000].reshape(100, 100)
        w2 = p[10000:20000].reshape(100, 100)
        
        # d is [N, 100]
        h = jnp.tanh(jnp.dot(d, w1))
        out = jnp.dot(h, w2)
        return jnp.mean((out - 1.0)**2)
        
    _value_and_grad = value_and_grad(lossf)
    
    def f(p):
        v, g = _value_and_grad(p, data)
        return v, g
        
    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=f,
        initial_position=init_params,
        tolerance=1e-8,
        max_iterations=10,
        num_correction_pairs=100
    )
    return results

def run_nn_workload(batch_size, name):
    print(f"\n--- Testing NN Workload {name} (Batch: {batch_size:,}) ---")
    cpu_device = jax.devices("cpu")[0]
    
    print("Allocating parameters and data...")
    init_params = jax.device_put(jnp.ones((20000,), dtype=jnp.float32), cpu_device)
    data = jax.device_put(jnp.ones((batch_size, 100), dtype=jnp.float32), cpu_device)
    
    print("Starting JIT compilation and execution...")
    t0 = time.time()
    with jax.default_device(cpu_device):
        try:
            res = lbfgs_test(init_params, data)
            # block until ready
            res.objective_value.block_until_ready()
            print(f"Compilation + First Run Time: {time.time() - t0:.4f}s")
        except Exception as e:
            print(f"Failed with exception: {e}")

if __name__ == "__main__":
    run_nn_workload(1_000, "Small")
    run_nn_workload(100_000, "Large")
    run_nn_workload(1_000_000, "Very Large")
