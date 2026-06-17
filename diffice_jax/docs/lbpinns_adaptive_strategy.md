# lbPINNs Self-Adaptive Loss Balancing — JAX Implementation

> **Note:** Code snippets below are illustrative references for the strategy, not a
> complete runnable implementation. Variable names (e.g. `params`, `batch`) stand in
> for whatever network and data structures your problem uses.

---

## Algorithm 2 Summary

The paper's Algorithm 2 has three steps, each mapped to JAX below.

---

### Step 1 — Initialize the noise collection `ε`

The algorithm requires an initial value for each noise scalar. The paper tests three
settings — `ε ∈ {0.02, 0.2, 2}` — and finds all converge to similar final weights,
so the method is robust to this choice. Which scalars are needed depends on which
loss terms are active in your problem:

| Loss term | Noise scalar | Active in |
|-----------|-------------|-----------|
| `L_PDE`  | `ε_f` | all problems |
| `L_BC`   | `ε_b` | all problems |
| `L_IC`   | `ε_i` | unsteady problems only |
| `L_data` | `ε_d` | when observed data is available |

Store `log_eps = log(ε)` rather than `ε` directly, so positivity is guaranteed
without any clamping and Adam operates on an unconstrained scalar:

```python
# Example: PDE + BC + IC problem, initialized at ε=2 for all terms
# (following the paper's [2, 2, 2] setting for Kovasznay and Beltrami flows)
eps_init = 2.0
log_eps = {
    'f': jnp.log(jnp.array(eps_init)),  # ε_f  — PDE residual
    'b': jnp.log(jnp.array(eps_init)),  # ε_b  — boundary condition
    'i': jnp.log(jnp.array(eps_init)),  # ε_i  — initial condition
}
```

The paper's final `ε` values settle near `10⁻²` regardless of initialization,
corresponding to effective weights `ω = 1/(2ε²) ≈ 5000`.

---

### Step 2 — Define the weighted loss (Eq. 8)

The Gaussian likelihood derivation produces a loss where each term is scaled by
`1/(2ε_k²)`, with `log(ε_k)` as a regularizer that prevents `ε_k → 0`:

```
L(ε, θ) = (1/2ε_f²) L_PDE + (1/2ε_b²) L_BC + (1/2ε_i²) L_IC + log(ε_f · ε_b · ε_i)
```

```python
# Illustrative reference — L_pde, L_bc, L_ic are scalar loss values
# computed from your own residual and mismatch functions
def adaptive_loss(log_eps, params, batch):
    L_pde, L_bc, L_ic = compute_losses(params, batch)  # your problem-specific losses

    eps_f = jnp.exp(log_eps['f'])
    eps_b = jnp.exp(log_eps['b'])
    eps_i = jnp.exp(log_eps['i'])

    return (L_pde / (2 * eps_f**2) + jnp.log(eps_f) +
            L_bc  / (2 * eps_b**2) + jnp.log(eps_b) +
            L_ic  / (2 * eps_i**2) + jnp.log(eps_i))
```

---

### Step 3 — Update both `ε` and `θ` jointly via Adam each epoch

Both the network weights `θ` (`params`) and the noise scalars `ε` (`log_eps`) are
passed as explicit arguments and updated by the same Adam optimizer in every step.
The JAX-specific requirement is that both appear in `argnums` so gradients flow
through both — if `log_eps` were a closure variable, JAX would treat it as a constant:

```python
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init((params, log_eps))  # joint state for both

grad_fn = jax.value_and_grad(adaptive_loss, argnums=(0, 1))  # (0=log_eps, 1=params)

@jax.jit
def update(log_eps, params, opt_state, batch):
    loss, (eps_grads, param_grads) = grad_fn(log_eps, params, batch)

    updates, new_opt_state = optimizer.update(
        (eps_grads, param_grads), opt_state
    )
    new_log_eps = optax.apply_updates(log_eps, updates[0])
    new_params  = optax.apply_updates(params,  updates[1])

    return new_log_eps, new_params, new_opt_state, loss

# Training loop (S steps, as in Algorithm 2)
for step in range(S):
    log_eps, params, opt_state, loss = update(log_eps, params, opt_state, batch)
```

To inspect the effective weights at any point:

```python
# ω_k = 1 / (2 ε_k²) — grows as ε_k shrinks toward 10⁻²
weights = {k: 1.0 / (2 * jnp.exp(v)**2) for k, v in log_eps.items()}
```
