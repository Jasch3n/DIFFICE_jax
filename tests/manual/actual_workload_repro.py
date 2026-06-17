import jax
import sys
import time
from diffice_jax import adam_opt, lbfgs_opt

# Attempt to reproduce using infer_xpinn_updated's functions
# instead of a dummy neural network
from diffice_jax.optimizer.optimization import lbfgs_optimizer

# Let's import the run_inference directly and modify parameters
import infer_xpinn_updated

print("STARTING LARGE WORKLOAD INFERENCE...")
t0 = time.time()
try:
    # Use larger parameters than default to simulate "large workload"
    # To save time on ADAM, we set epochs_adam=1
    res = infer_xpinn_updated.run_inference(
        epochs=10000,
        use_lbfgs=True,
        use_lbpinn=False,
    )
    print(f"FINISHED IN {time.time() - t0:.2f}s")
except Exception as e:
    print(f"FAILED: {e}")
