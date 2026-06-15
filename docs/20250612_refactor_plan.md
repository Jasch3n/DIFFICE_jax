# Performance-Aware XPINN Loss Refactor Plan

## Summary

Refactor XPINN `joint_inversion` and `regression` losses around typed configs, loss-specific builders, and callable loss objects while preserving JAX performance. The loss object must remain a static JIT argument, with configuration resolved before tracing and no dynamic Python branching added inside traced numerical paths.

## Key Changes

- Keep `LossConfig` as the public `DIFFICESolver` API.
- Add internal frozen config dataclasses:
  - `DIFFICEJointInversionConfig`
  - `DIFFICEXPINNRegressionConfig`
- Add canonical factories:
  - `loss_joint_create(solNN, eqn, front_eqn, config)`
  - `loss_regression_create(solNN, eqn, front_eqn, config)`
- Return callable loss objects with:
  - `__call__(params, data)`
  - `kfac_residuals(params, data, terms=None, regions=None)`
  - `kfac_objective(params, data, terms=None, regions=None)`
  - mutable `lref`, matching current optimizer expectations.
- Keep legacy loss factory signatures through wrappers.

## Performance Rules

- Loss objects must be static-JIT friendly:
  - define `__hash__ = object.__hash__`,
  - avoid dataclass-generated equality on loss objects,
  - do not register loss objects as PyTrees,
  - store static config as object fields, not as dynamic function arguments.
- Use `__slots__` on concrete loss classes to reduce Python object overhead and prevent accidental attribute growth.
- Precompute static constants in `__init__`:
  - `idxgall` as tuple,
  - `basal_mask` as tuple,
  - active region/static masks where possible,
  - default term weights as `jnp.array`,
  - booleans such as `use_eqn`, `use_ct`, `use_gpinn`, `use_mu_grad`.
- Keep numerical helper functions JAX-pure and closure-based inside the object, so `jax.grad`, `jax.jit`, and KFAC see the same computation style as today.
- Do not pass config objects into jitted functions. The config is captured by the static `lossf` object.
- Avoid inheritance dispatch in hot methods. Use a `Protocol` or lightweight abstract interface for typing, but concrete classes should implement their own `__call__` and `kfac_residuals` directly.
- Preserve current `lref` behavior:
  - set `lossf.lref` after initial loss evaluation and before optimizer JIT tracing,
  - do not mutate `lref` during a jitted optimizer step.

## Solver Workflow

- `_prepare_xpinn` still performs shared setup:
  - load raw data,
  - normalize data,
  - infer regions,
  - build `basal_mask`,
  - build `scales`,
  - initialize params,
  - create `solution`.
- Select a loss builder by `LossConfig.name`:
  - `JointInversionLossBuilder`
  - `RegressionLossBuilder`
- Builder responsibilities:
  - create the correct sampler/data contract,
  - create the internal DIFFICE loss config,
  - return the callable loss object.
- Remove dummy inverse-target injection from the real-data joint inversion path.

## Loss Behavior

- `joint_inversion`:
  - data terms: `u, v, h, s`,
  - no supervised `mu` or `C`,
  - no dummy `Mu_smp` or `C_smp`,
  - keeps equation, calving-front, interface matching, optional gPINN, optional `mu_grad`.
- `regression`:
  - data terms: `u, v, h, s, log(mu), C`,
  - remains synthetic/test-only supervised inversion loss.
- KFAC residuals must exactly match the corresponding objective:
  - joint inversion excludes inverse-target residuals,
  - regression keeps inverse-target residuals.

## Tests

- Use `pyenv activate Metal-Env`.
- Add focused tests:
  - `loss_joint_create` evaluates on real-data-style batches without `Mu_smp`/`C_smp`.
  - `loss_regression_create` evaluates on regression batches with true `mu`/`C`.
  - `joint_inversion` KFAC residuals exclude inverse-target residuals.
  - legacy `loss_regression_create(...)` signature still works.
  - `DIFFICESolver.prepare()` works for `LossConfig(name="joint_inversion")` without dummy targets.
- Add a performance guard smoke test:
  - run two Adam minimizer calls with the same loss object and batch shape,
  - confirm no config object is passed as a dynamic JAX argument,
  - optionally inspect that the second call reuses the compiled path by timing or JAX compile logging when available.

## Assumptions

- First refactor covers XPINN `joint_inversion` and `regression` only.
- Callable loss objects are acceptable, but they must be identity-hashable static JIT arguments.
- The abstract loss interface is for typing and organization, not for runtime dynamic dispatch inside traced computations.
- Existing optimizer APIs stay unchanged.
