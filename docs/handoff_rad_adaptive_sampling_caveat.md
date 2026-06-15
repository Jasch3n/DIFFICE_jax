# Handoff

## Objective
- Document the important RAD sampling caveat discovered while troubleshooting the XPINN KFAC adaptive-sampling stall.
- Preserve the exact fix and verification path so future agents do not reintroduce full-library equation evaluation through `gov_eqn_iso(...)`.

## Status
- The solver-native RAD hang is fixed for config-driven XPINN KFAC workflows.
- `DIFFICESolver._adaptive_eval(...)` now treats `gov_eqn_iso(...)` as a pointwise SSA equation evaluator and batches it explicitly.
- `docs/known-issues.md` has been updated from an open issue to a fixed issue with root cause and smoke-run timing.

## Major Changes This Session
- Added YAML-stage controls for RAD/adaptive sampling:
  - `training.stages[].adaptive_sampling`
  - `training.stages[].adaptive_sampling_burn_in`
  - `training.stages[].adaptive_sampling_period`
- Kept adaptive-sampling controls out of `training.stages[].optimizer.parameters`; misplaced `optimizer.parameters.adaptive_sampling*` keys are rejected.
- Fixed the RAD stall in `diffice_jax/core/solver.py`:
  - Old behavior passed the full collocation-point library directly into `gov_eqn_iso(...)`.
  - New behavior evaluates `gov_eqn_iso(...)` pointwise via `jax.vmap(...)`.
  - Evaluation is chunked in blocks of 2048 collocation points to bound CPU memory and compile cost.
- Added KFAC logging around RAD windows:
  - `KFAC step N | adaptive_sampling=start | burn_in=... | period=...`
  - `KFAC step N | adaptive_sampling=done | elapsed=...`
- Added a focused regression test proving the solver RAD evaluator calls `gov_eqn_iso(...)` on one collocation point at a time.

## Unresolved Issues
- The forced one-step adaptive smoke run still spent 53.2s total on CPU because the first KFAC optimizer step includes JAX/KFAC compile and first-step execution cost. This is separate from the RAD probability evaluation, which completed in 3.9s.
- The old regression scripts under `tests/test_xpinn_regression/` and parts of `tests/test_xpinn_joint_inversion_kfac.py` still have their own legacy RAD paths. The solver fix protects `DIFFICESolver`, not every legacy script path.
- Do not assume passing a full `(N, 2)` collocation array into `gov_eqn_iso(...)` is safe. That function uses `jax.jacfwd` internally and should be treated as pointwise unless deliberately refactored and benchmarked.

## Relevant Commands
```bash
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m py_compile diffice_jax/core/solver.py diffice_jax/workflow/runner.py && python -m pytest tests/test_workflow_config.py tests/test_xpinn_loss_refactor.py tests/test_xpinn_floating_surface.py
```

```bash
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -c "from diffice_jax.workflow.config import load_workflow_config; from diffice_jax.workflow.runner import build_solver_from_config; cfg=load_workflow_config('examples/configs/xpinn_joint_flatbed_kfac_gpinn.yaml'); stage=cfg.training['stages'][0]; stage['iterations']=1; stage['adaptive_sampling']=True; stage['adaptive_sampling_burn_in']=0; stage['adaptive_sampling_period']=1; stage['optimizer']['parameters']['log_rate']=1; solver=build_solver_from_config(cfg).prepare(); solver.fit(); print('SMOKE_LOSS_HISTORY_LEN=' + str(len(solver.loss_history)))"
```

## Relevant Files
- `diffice_jax/core/solver.py`: `TrainingStage` adaptive controls, KFAC RAD logs, and `DIFFICESolver._adaptive_eval(...)` pointwise/chunked evaluator.
- `diffice_jax/workflow/runner.py`: YAML parsing and rejection of misplaced adaptive-sampling optimizer parameters.
- `diffice_jax/data/xpinns/sampling.py`: `eval_RAD_probs(...)` expects `eval_f(x_col_batch, idx, basal)` to return batched equation residuals and terms.
- `diffice_jax/equation/eqn_iso.py`: `gov_eqn(...)` is pointwise and uses `jax.jacfwd`; this is the critical caveat.
- `tests/test_workflow_config.py`: parser tests plus `test_solver_adaptive_eval_vectorizes_pointwise_equation`.
- `docs/known-issues.md`: fixed issue entry for the XPINN KFAC RAD stall.
- `examples/configs/xpinn_joint_flatbed_kfac_gpinn.yaml`: smoke-run source config used to force RAD at step 1.

## Next Steps
- If adaptive KFAC is used for long runs, first run a short forced-RAD smoke with `adaptive_sampling_burn_in: 0`, `adaptive_sampling_period: 1`, and `iterations: 1`.
- If future agents optimize RAD, preserve the shape contract: sampler-facing `eval_f` may accept a batch, but `gov_eqn_iso(...)` itself must remain pointwise unless equation differentiation is redesigned.
- Consider adding similar pointwise/chunked protection to legacy RAD paths if they are still used for production comparisons.
