# Optimize XPINN Joint-Inversion Workflow

## Context

The target workflow is the flatbed two-region XPINN joint inversion exercised by
`tests/test_xpinn_joint_inversion_kfac.py`. The current KFAC path is functional
but expensive because each optimizer step rebuilds several high-cost residual
terms, including equation derivatives and interface matching derivatives.

This note proposes four code-level acceleration paths. The first is the highest
priority because it removes duplicated work without intentionally changing the
KFAC objective.

## 1. Add `lossf.kfac_eval(...)`

### Problem

The current KFAC loss wrapper computes the XPINN loss twice per optimizer step:

```python
_, loss_info, _ = lossf(current_params, batch)
residuals = lossf.kfac_residuals(current_params, batch) / jnp.sqrt(lossf.lref)
```

The first call exists only to get `loss_info` diagnostics. The second call
builds the residual vector needed by `kfac_jax.register_squared_error_loss`.
Both paths recompute expensive term evaluations.

This is especially costly for matching loss. The scalar diagnostic path calls
`loss_md_sub(...)`, and the KFAC residual path calls
`loss_md_kfac_res_sub(...)`; both call `loss_md_res_sub(...)`, which evaluates
networks, first derivatives, and second derivatives on both sides of each XPINN
interface.

### Desired API

Add a single KFAC-oriented evaluation method to the loss object:

```python
def kfac_eval(params, data, terms=None, regions=None):
    loss_n, loss_info, reg_err_list, residuals = ...
    return loss_n, loss_info, reg_err_list, residuals
```

Attach it in `loss_regression_create(...)`:

```python
loss_fun.kfac_eval = kfac_eval
```

Keep existing public helpers as compatibility wrappers:

```python
def kfac_residuals(params, data, terms=None, regions=None):
    return kfac_eval(params, data, terms=terms, regions=regions)[3]

def kfac_objective(params, data, terms=None, regions=None):
    return kfac_eval(params, data, terms=terms, regions=regions)[0]
```

Then update KFAC call sites:

```python
def kfac_lossf(current_params, batch):
    loss_n, loss_info, _, residuals = lossf.kfac_eval(current_params, batch)
    residuals = residuals / jnp.sqrt(lossf.lref)
    kfac_loss_functions.register_squared_error_loss(
        residuals,
        targets=jnp.zeros_like(residuals),
    )
    return loss_n, loss_info
```

Apply this in both:

- `tests/test_xpinn_joint_inversion_kfac.py`
- `diffice_jax/core/solver.py`

### Refactor Shape

Introduce shared residual block helpers inside `loss_regression_create(...)`.
Each block computes the expensive raw term once and exposes both diagnostics and
KFAC residual contributions.

Example shape:

```python
def match_block(params, data, idx):
    C0_res, C1_res = loss_md_res_sub(params, data, idx)
    err = jnp.hstack((ms_error(C0_res), ms_error(C1_res)))
    weighted = jnp.concatenate([
        (c0_weight * C0_res).reshape(-1),
        (c1_weight * C1_res).reshape(-1),
    ])
    return err, weighted
```

Use the same pattern for:

- `data_block(...)`
- `eqn_block(...)`
- `ct_block(...)`
- `match_block(...)`
- `gpinn_block(...)`
- `mu_grad_block(...)`

`kfac_eval(...)` should then:

1. Compute active blocks once.
2. Build `loss_info` from block `err` values.
3. Build `reg_err_list` from the same error arrays.
4. Concatenate selected block `weighted` residuals.
5. Return `loss_n = sum(residuals ** 2) / loss_fun.lref`.

Do not make ordinary `loss_fun(...)` call `kfac_eval(...)` in the first pass.
Adam/L-BFGS should preserve the current scalar objective path until equivalence
and scientific behavior are validated.

### Validation

Use cheap deterministic checks before any long run:

- Fixed params and fixed batch: old `kfac_residuals(...)` equals new
  `kfac_eval(...)[3]`.
- Fixed params and fixed batch: old `kfac_objective(...)` equals new
  `kfac_eval(...)[0]`.
- Fixed params and fixed batch: new `loss_info` matches old `lossf(...)`
  diagnostics for all enabled terms.
- Short KFAC smoke run: no NaNs and objective decreases.

Only after those pass, run the expensive joint-inversion metric gate.

## 2. Replace Per-Output `value_and_grad` With One Jacobian

### Problem

`diffice_jax/model/xpinns/networks.py::gradf(...)` computes gradients through
six separate `value_and_grad(...)` calls, one for each network output component:

```python
vals_grads = [value_and_grad(lambda zz, i=i: f(params, zz, idx)[i])(z) for i in range(6)]
```

This repeats forward work and becomes more expensive when matching loss takes
second derivatives by differentiating `gradNN(...)`.

### Proposed Change

Compute the full output Jacobian once:

```python
def grad_point(z):
    jac = jax.jacfwd(lambda zz: f(params, zz, idx))(z)
    return jnp.ravel(jac[:6], order="C")
```

Then keep the existing downstream scaling and output layout unchanged.

### Validation

- Fixed params and fixed inputs: old `gradNN(params, x, idx)` equals new
  `gradNN(params, x, idx)` within numerical tolerance.
- Fixed params and fixed batch: matching loss diagnostics and residuals match.
- Fixed params and fixed batch: equation loss is unchanged where it depends on
  the same solution fields.

## 3. Reduce KFAC Inverse Update Frequency

### Problem

The current test config updates KFAC inverses every optimizer step:

```python
inverse_update_period=1
```

With `curvature_block_type="naive_full"`, this can be a large wall-time cost.

### Proposed Change

Benchmark larger update periods:

```python
inverse_update_period=5
inverse_update_period=10
```

This is a tuning change, not a pure refactor. It can change convergence speed in
iterations, so compare wall time and final inversion metrics rather than only
iteration count.

### Validation

For each candidate value, record:

- seconds per KFAC iteration
- objective at fixed wall time
- objective at fixed iteration count
- `mu_rel_mae`
- `c_rel_mae`

Keep `inverse_update_period=1` as the reference result until a slower-update
setting is shown to preserve or improve wall-time-to-target.

## 4. Filter KFAC Terms For Inversion Runs

### Problem

The joint-inversion test currently calls:

```python
lossf.kfac_residuals(current_params, batch)
```

with no term filter, so it includes every available KFAC residual term. For a
real joint inversion, ground-truth effective viscosity and basal friction data
are not available, and the KFAC objective may not need the full synthetic
regression data residual set.

The legacy regression script already has a term-filtering concept through
`KFAC_LOSS_TERMS`, `KFAC_ACTIVE_REGIONS`, and separate physical/data objective
weights.

### Proposed Change

Expose explicit KFAC term selection in the joint-inversion workflow, for
example:

```python
terms = ("eqn", "ct", "match")
lossf.kfac_eval(current_params, batch, terms=terms, regions=active_regions)
```

If data residuals are still wanted, keep them as a separate explicitly weighted
block:

```python
phys_residuals = lossf.kfac_eval(..., terms=("eqn", "ct", "match"))[3]
data_residuals = lossf.kfac_eval(..., terms=("data",))[3]
residuals = concat([
    sqrt(phys_weight) * phys_residuals,
    sqrt(data_weight) * data_residuals,
])
```

This should be treated as an objective change. It may accelerate each step and
may better match the real-data joint-inversion target, but it must be validated
against inversion quality.

### Validation

Compare at least these variants:

- Current all-term objective.
- Physical terms only: `("eqn", "ct", "match")`.
- Physical terms plus data terms with separate weights.
- Optional gPINN term included when the target workflow enables gPINN.

Record objective curves and final `mu_rel_mae` / `c_rel_mae`.

## Lower-Priority Follow-Up: Sampler Overhead

The sampler performs several un-jitted `random.choice(...)` operations per
iteration and RAD sampling periodically evaluates equation residuals over the
full collocation library. This is probably not the first bottleneck relative to
duplicated derivative work, but it should be profiled after the `kfac_eval`
refactor.

Possible follow-ups:

- JIT a fixed-shape sampler path for the common two-region joint-inversion case.
- Cache interface-collocation libraries and RAD probabilities when the adaptive
  sample is intentionally reused.
- Expose sample-count controls without mutating module globals.

## Recommended Order

1. Implement `lossf.kfac_eval(...)` and update the KFAC call sites.
2. Add equivalence tests for residuals, objectives, and diagnostics.
3. Replace the six-output gradient loop with one Jacobian and re-run equivalence
   tests.
4. Benchmark `inverse_update_period` values.
5. Benchmark term-filtered KFAC objectives against inversion metrics.
6. Profile sampler overhead only after the duplicated loss/derivative work is
   removed.
