# Reconciled XPINN Joint-Inversion Refactor And KFAC Plan

## Purpose

This plan reconciles `docs/20250612_refactor_plan.md` and
`docs/20260612_optimize_xpinn_joint_workflow.md`.

The target is a real-data XPINN joint inversion workflow for coupled grounded
and floating regions, while preserving the synthetic regression workflow as a
separate validation path. The refactor must also keep the KFAC stage fast enough
to be useful: KFAC should consume one weighted residual vector and should not
recompute expensive equation, calving-front, matching, gPINN, or mu-gradient
terms only to collect diagnostics.

## Reconciled Decisions

- Keep `LossConfig` as the public `DIFFICESolver` loss-selection API.
- Split real-data joint inversion from the regression workflow:
  - `joint_inversion` fits observed `u`, `v`, `h`, and `s`, plus equation,
    calving-front, matching, optional gPINN, and optional mu-gradient terms.
  - `joint_inversion` must not require ground-truth `mu` or `C`.
  - `regression` remains the synthetic supervised workflow that may compare
    `log(mu)` and `C` against ground truth.
- Remove dummy inverse-target injection from the real-data joint-inversion path.
- Introduce static, callable XPINN loss objects instead of continuing to grow
  closure-attached helper functions.
- Add `lossf.kfac_eval(...)` to compute scalar KFAC objective, diagnostics,
  regularization error lists, and the KFAC residual vector in one pass.
- Keep `lossf.kfac_residuals(...)` and `lossf.kfac_objective(...)` as
  compatibility wrappers around `kfac_eval(...)`.
- Treat KFAC term filtering and KFAC inverse update frequency as experiments,
  not default behavior changes, until inversion quality is validated.

## Main Conflict Resolution

The refactor plan is about API boundaries and scientific semantics. The
optimization plan is about removing duplicated KFAC work in the current
implementation. The reconciled order is:

1. First make the joint-inversion and regression data contracts explicit.
2. Then introduce a static loss-object API that includes `kfac_eval(...)`.
3. Then use residual blocks to make KFAC compute expensive terms once per step.
4. Only after equivalence checks pass, apply lower-level derivative and KFAC
   tuning changes.

This order avoids optimizing the legacy regression-shaped loss before the
real-data joint-inversion contract is fixed, while still preserving the highest
value KFAC optimization from the second plan.

## Source Plan Mapping

Kept from `20250612_refactor_plan.md`:

- public `LossConfig` selection through `DIFFICESolver`;
- separate internal configs and builders for joint inversion and the regression
  workflow;
- static callable loss objects with mutable `lref`;
- no dummy inverse targets for real-data joint inversion;
- joint-inversion KFAC residuals excluding inverse-target residuals;
- regression workflow residuals preserving synthetic `mu` and `C` supervision.

Kept from `20260612_optimize_xpinn_joint_workflow.md`:

- `kfac_eval(...)` as the main KFAC acceleration path;
- compatibility wrappers for `kfac_residuals(...)` and `kfac_objective(...)`;
- residual blocks that compute expensive derivative terms once;
- KFAC call-site updates in `DIFFICESolver._fit_kfac_stage` and the flatbed
  XPINN KFAC test;
- Jacobian-based `gradNN(...)` as a follow-up performance refactor;
- inverse-update-period, term-filtering, and sampler-overhead work as benchmarked
  follow-ups instead of unvalidated defaults.

## Target Loss API

Add internal frozen config dataclasses for concrete XPINN losses:

```python
@dataclass(frozen=True)
class DIFFICEJointInversionConfig:
    ...

@dataclass(frozen=True)
class DIFFICEXPINNRegressionConfig:
    ...
```

Add canonical factories:

```python
loss_joint_create(solNN, eqn, front_eqn, config)
loss_regression_create(solNN, eqn, front_eqn, config)
```

The returned loss object must provide:

```python
lossf(params, data) -> (loss_n, loss_info, reg_err_list)
lossf.kfac_eval(params, data, terms=None, regions=None)
lossf.kfac_residuals(params, data, terms=None, regions=None)
lossf.kfac_objective(params, data, terms=None, regions=None)
lossf.lref
```

`kfac_eval(...)` returns:

```python
loss_n, loss_info, reg_err_list, residuals
```

Legacy call signatures should remain as wrappers at public and test-facing
boundaries, especially the current positional `loss_regression_create(...)`
signature in `diffice_jax/model/xpinns/loss.py`.

## Static JIT Requirements

Concrete loss objects are static JIT arguments. They must:

- define `__hash__ = object.__hash__`;
- avoid dataclass-generated equality on the runtime loss object;
- avoid registering the loss object as a PyTree;
- store config, `idxgall`, `basal_mask`, static masks, default weights, and
  feature booleans as object fields;
- use `__slots__` on concrete runtime loss classes;
- keep `lref` mutable for current optimizer behavior;
- not mutate `lref` inside a jitted optimizer step;
- not pass config objects as dynamic JAX arguments.

Use a lightweight `Protocol` only for typing if needed. Do not rely on
inheritance dispatch inside traced numerical paths.

## Residual Blocks

Each active loss term should have a block helper that computes the expensive raw
term once and exposes both diagnostics and KFAC residual contributions:

- `data_block(...)`
- `eqn_block(...)`
- `calving_front_block(...)`
- `match_block(...)`
- `gpinn_block(...)`
- `mu_grad_block(...)`

Each block should return enough information to build:

- per-term diagnostic errors for `loss_info`;
- `reg_err_list` entries matching the existing optimizer/test expectations;
- weighted residuals for the KFAC residual vector.

`kfac_eval(...)` should:

1. Resolve active `terms` and `regions` outside traced dynamic branching where
   possible.
2. Compute each selected block once.
3. Build `loss_info` from the same errors used for residual construction.
4. Concatenate weighted residuals into the KFAC residual vector.
5. Return `loss_n = sum(residuals ** 2) / lossf.lref`.

Do not force the ordinary Adam/L-BFGS scalar path to call `kfac_eval(...)` in the
first implementation. First prove KFAC equivalence, then optionally route the
scalar path through shared blocks in a later cleanup.

## Solver Workflow

Keep `_prepare_xpinn` responsible for shared setup:

- load raw data;
- normalize data;
- infer sub-regions and `basal_mask`;
- build region scales;
- initialize params;
- create the XPINN solution.

Select loss behavior by `LossConfig.name`:

- `joint_inversion` uses the real-data sampler/data contract and
  `loss_joint_create(...)`;
- `regression` uses the regression workflow sampler/data contract and
  `loss_regression_create(...)`.

Builder responsibilities:

- create the correct sampler contract;
- build the internal DIFFICE loss config;
- return the callable loss object;
- keep legacy aliases only at API and data-loading boundaries.

`diffice_jax/core/loss_terms.py` should stop wrapping real-data batches with
`dataf_with_dummy_inverse_targets(...)` once `loss_joint_create(...)` no longer
requires `Mu_smp` or `C_smp`.

## KFAC Stage

Update `DIFFICESolver._fit_kfac_stage` and the flatbed XPINN KFAC test path to
prefer `kfac_eval(...)`:

```python
loss_n, loss_info, _, residuals = lossf.kfac_eval(current_params, batch)
residuals = residuals / jnp.sqrt(lossf.lref)
kfac_loss_functions.register_squared_error_loss(
    residuals,
    targets=jnp.zeros_like(residuals),
)
return loss_n, loss_info
```

Keep a fallback for losses that only expose scalar objectives.

Expose KFAC term and region selection through optimizer parameters first, rather
than expanding the public `LossConfig` immediately:

```python
OptimizerConfig(
    name="kfac",
    parameters={
        "kfac_terms": ("eqn", "ct", "match"),
        "kfac_active_regions": (0, 1),
    },
)
```

Default behavior should remain all active terms until validation shows a
filtered objective improves wall-time-to-target without damaging inversion
quality.

## Performance Follow-Ups

After the loss API and KFAC equivalence tests pass:

1. Replace the per-output `value_and_grad(...)` loop in
   `diffice_jax/model/xpinns/networks.py::gradf(...)` with a single Jacobian
   computation, preserving the existing output layout and scaling.
2. Benchmark `inverse_update_period=5` and `inverse_update_period=10` against
   the current `inverse_update_period=1`.
3. Benchmark KFAC objectives:
   - all active terms;
   - physical terms only: `("eqn", "ct", "match")`;
   - physical terms plus separately weighted data terms;
   - optional gPINN included when enabled.
4. Profile sampler overhead only after duplicated loss and derivative work is
   removed.

The Jacobian change is a pure performance refactor only if fixed-parameter
diagnostics and residuals match within tolerance. KFAC term filtering and inverse
update frequency are optimizer/objective experiments and must be judged by
wall-time-to-target and final inversion metrics.

## Validation Plan

Use `pyenv activate Metal-Env` before running checks. Because several old tests
are known to be stale for grounded ice domains, add focused tests first and only
run broader suites when they are relevant to the changed behavior.

Add deterministic tests for:

- `loss_joint_create(...)` on real-data-style batches without `Mu_smp` or
  `C_smp`;
- `loss_regression_create(...)` on regression workflow batches with true `mu`
  and `C`;
- joint-inversion KFAC residuals excluding inverse-target residuals;
- regression KFAC residuals preserving inverse-target residuals when requested;
- legacy `loss_regression_create(...)` signature compatibility;
- `DIFFICESolver.prepare()` with `LossConfig(name="joint_inversion")` and no
  dummy inverse targets;
- `kfac_eval(...)[3]` matching the old `kfac_residuals(...)` vector for fixed
  params and fixed batches before wrappers are replaced;
- `kfac_eval(...)[0]` matching the old `kfac_objective(...)` for fixed params
  and fixed batches;
- `kfac_eval(...)` diagnostics matching `lossf(...)` diagnostics for enabled
  terms;
- the Jacobian-based `gradNN(...)` matching the current implementation within
  tolerance before the old implementation is removed.

Add one smoke test for KFAC with fixed batch shape:

- two optimizer steps with the same loss object;
- no config object passed as a dynamic JAX argument;
- no NaNs;
- objective does not increase catastrophically.

For expensive validation, record:

- seconds per KFAC iteration;
- objective at fixed iteration count;
- objective at fixed wall time;
- `mu_rel_mae`;
- `c_rel_mae`.

The long-run target remains the two-region joint inversion: grounded basal
friction and floating-region effective viscosity should both be below the
accepted relative-error threshold for the validation data.

## Deferred Work

- Do not refactor standalone PINNs as part of this XPINN joint-inversion/KFAC
  plan.
- Do not make filtered KFAC terms the default until inversion metrics support
  the change.
- Do not change `inverse_update_period` defaults before benchmarking.
- Do not rewrite the sampler before profiling shows it remains a meaningful
  bottleneck after `kfac_eval(...)` and Jacobian work.
- Do not remove legacy regression workflow names at public boundaries; keep them
  as compatibility aliases while new internal modules use canonical names.
