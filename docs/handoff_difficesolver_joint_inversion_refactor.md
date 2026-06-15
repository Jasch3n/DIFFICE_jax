# Handoff

## Objective
- Refactor `tests/test_xpinn_joint_inversion_kfac.py` so the default training path uses the new `DIFFICESolver` API without changing numerical behavior, checkpoint compatibility, plotting outputs, or KFAC performance.
- Add a temporary `--legacy-runner` CLI flag so the current working implementation remains available as a behavioral baseline while the solver-native path is introduced and matched.

## Status
- `tests/test_xpinn_joint_inversion_kfac.py` currently does **not** use `DIFFICESolver`.
- The script uses the legacy XPINN regression helper path from `tests/test_xpinn_regression/xpinn_regression.py` plus a manual KFAC loop.
- The current script is the reference implementation for:
  - KFAC optimizer defaults.
  - gPINN interface collocation handling.
  - adaptive collocation reuse.
  - checkpoint resume.
  - plot-only checkpoint rendering.
  - cached plotting.
  - equation residual and x-equation term-ratio diagnostics.
- The most recent long run completed to step `100000` from checkpoint step `5000`; artifacts are organized under `tests/figures/joint_inversion__gpinn_100000/`.
- The proposed refactor has not been implemented yet.

## Major Changes This Session
- Confirmed that `tests/test_xpinn_joint_inversion_kfac.py` still imports and uses:
  - `KfacOptimizer` directly from `diffice_jax`.
  - legacy helper module `tests.test_xpinn_regression.xpinn_regression as xr`.
  - `xr.initialize_xpinn(...)` and `xr.initialize_loss(...)`.
- Confirmed that `DIFFICESolver._fit_kfac_stage` exists in `diffice_jax/core/solver.py`, but it does not yet match the working script:
  - different KFAC defaults;
  - no equivalent checkpoint continuation state;
  - no explicit `log_rate`;
  - no current equivalent of the script-local `_limit_batch`;
  - no current equivalent of `_attach_gpinn_interface_collocation`;
  - no solver-native replacement for `xr.attach_loss_weights`;
  - no adaptive collocation memory/reuse path matching the working script.
- Designed the migration approach:
  - keep the current runner as `--legacy-runner`;
  - introduce a solver-native runner beside it;
  - match objective and performance on compact cases before making solver-native the default;
  - keep plotting/reporting functions test-local for now.

## Unresolved Issues
- `DIFFICESolver._fit_kfac_stage` currently uses optimizer defaults that differ from the working script:
  - current solver default: `norm_constraint=1e-3`;
  - working script: `norm_constraint=1e-8`;
  - current solver default: `initial_damping=1e-3`;
  - working script: `initial_damping=1`;
  - current solver default: `curvature_ema=0.95`;
  - working script: `curvature_ema=0.997`;
  - current solver default: `inverse_update_period=1`;
  - working script: `inverse_update_period=10`;
  - current solver default: `damping_adaptation_decay=0.998`;
  - working script: `damping_adaptation_decay=0.997`.
- The solver path must convert XPINN params to a plain `dict` before KFAC unless `XPINNParams` is registered as a valid pytree for KFAC.
- The solver path must reproduce batch decoration exactly:
  - limit interface points only when requested;
  - limit calving-front points when requested;
  - attach gPINN interface collocation from the collocation tail;
  - attach the same region/loss weights currently produced by `xr.attach_loss_weights`.
- The solver path must preserve adaptive sampling behavior:
  - call `dataf(..., eval_adaptive=True, eval_f=...)` at the same cadence;
  - store `x_col_mem`;
  - reuse `x_col_mem` on non-adaptive steps after adaptation starts;
  - refresh `gpinn_col` when reused collocation changes.
- Checkpoint compatibility is required. Existing `plot_checkpoint()` expects current checkpoint keys:
  - `step`;
  - `params`;
  - `opt_state`;
  - `damping`;
  - `key`;
  - `loss_history`;
  - `mu_rel_mae`;
  - `c_rel_mae`;
  - `config`;
  - `x_col_mem`;
  - `adapted`.
- It is unknown whether a solver-native run will match the current objective bit-for-bit. Exact equivalence may be too strict because call ordering and PRNG splitting can differ. At minimum, the step-0 objective and residual-vector shape must match on the same prepared batch.

## Relevant Commands
```bash
# Syntax check for the current joint inversion script.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m py_compile tests/test_xpinn_joint_inversion_kfac.py'

# Current plot-only checkpoint rendering path; should keep working after the refactor.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --plot-checkpoint tests/figures/joint_inversion__gpinn_100000/checkpoints/KFAC_step_100000.pkl --tag gpinn_100000_replot'

# Current cache-only plotting path; should keep working after the refactor.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --plot-cache tests/figures/joint_inversion__gpinn_100000/test_xpinn_joint_inversion_flatbed_gpinn_100000_plot_cache.pkl --tag gpinn_100000_cache_replot'

# Proposed legacy baseline command after adding --legacy-runner.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --legacy-runner --iterations 5 --tag legacy_solver_match_5 --log-rate 1 --depth 2 --width 8 --sample-count 32 --interface-points 32 --calving-front-points 32 --interface-collocation 32 --no-adaptive-sampling'

# Proposed solver-native comparison command after adding the new runner.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --iterations 5 --tag solver_match_5 --log-rate 1 --depth 2 --width 8 --sample-count 32 --interface-points 32 --calving-front-points 32 --interface-collocation 32 --no-adaptive-sampling'

# Proposed gPINN-enabled compact comparison after step-0 matching is established.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --iterations 5 --tag solver_match_gpinn_5 --log-rate 1 --depth 2 --width 8 --sample-count 32 --interface-points 32 --calving-front-points 32 --interface-collocation 32 --use-gpinn --no-adaptive-sampling'
```

## Relevant Files
- `tests/test_xpinn_joint_inversion_kfac.py`
  - Current working legacy/manual KFAC implementation.
  - Add `--legacy-runner` here.
  - Keep plotting, dashboard, cache, checkpoint plotting, and artifact organization here for now.
- `diffice_jax/core/solver.py`
  - Contains `DIFFICESolver`, typed configs, and `_fit_kfac_stage`.
  - Main refactor target for solver-native KFAC behavior.
- `diffice_jax/core/loss_terms.py`
  - Contains `JointInversionLossBuilder` and `RegressionLossBuilder`.
  - Solver-native joint inversion should continue using these builders.
- `diffice_jax/model/xpinns/loss.py`
  - Provides `kfac_eval`, `kfac_residuals`, `kfac_objective`, region term weights, active-region handling, gPINN residuals, and mu-gradient residual support.
  - The solver-native KFAC path should call these methods rather than duplicating residual construction.
- `tests/test_xpinn_regression/xpinn_regression.py`
  - Behavioral reference for `attach_loss_weights`, adaptive sampling cadence, matching weights, and legacy XPINN setup.
  - Do not import this module from `diffice_jax/core`; copy or promote only the required behavior into package code.
- `diffice_jax/data/xpinns/sampling.py`
  - Contains XPINN sampling globals such as `N_INTERFACE_LIBRARY` and `N_INTERFACE_COLLOCATION`.
  - The refactor should minimize new global mutation; pass interface-collocation controls through solver config where possible.
- `tests/figures/joint_inversion__gpinn_100000/checkpoints/KFAC_step_100000.pkl`
  - Current successful checkpoint.
- `tests/figures/joint_inversion__gpinn_100000/test_xpinn_joint_inversion_flatbed_gpinn_100000_plot_cache.pkl`
  - Current plot cache; useful for regenerating figures without PINN evaluation.
- `tests/figures/joint_inversion__gpinn_100000/dashboard.html`
  - Current communication dashboard.
- `tests/figures/joint_inversion__gpinn_100000/dashboard.md`
  - Markdown companion to the dashboard.

## Next Steps
- Add `--legacy-runner` to `tests/test_xpinn_joint_inversion_kfac.py`.
  - When present, call the current `run_kfac_experiment(...)` implementation unchanged.
  - When absent, call a new solver-native implementation.
  - During transition, consider defaulting to legacy until matching checks pass; after matching, flip default to solver-native and keep `--legacy-runner` as fallback.
- Add a solver builder in the test script:
  - use `djax.DIFFICESolver`;
  - `DataConfig(source=DATA_PATH, regression_workflow=True, ...)`;
  - `ModelConfig(workflow="xpinn", regions=[RegionConfig("grounded", 0), RegionConfig("floating", 1)], network=NetworkConfig(depth=depth, width=width))`;
  - `LossConfig(name="joint_inversion", matching=True, calving_front=True, gpinn_weight=...)`;
  - `TrainingConfig` with one KFAC `TrainingStage`.
- Extend `DIFFICESolver._fit_kfac_stage` to accept script-equivalent controls through `stage.optimizer.parameters`:
  - `log_rate`;
  - `legacy_kfac_eval`;
  - `interface_points`;
  - `interface_collocation`;
  - `max_iterations`;
  - `checkpoint_dir`;
  - `resume_checkpoint`;
  - `target_c_rel_mae`;
  - `adaptive_sampling`.
- Change solver KFAC defaults to match `tests/test_xpinn_joint_inversion_kfac.py::kfac_config()` when the workflow is XPINN joint inversion, or provide an explicit named preset such as:
  - `parameters={"preset": "xpinn_joint_inversion_reference"}`.
- Promote the script-local batch helpers into package code:
  - `_limit_batch`;
  - `_attach_gpinn_interface_collocation`;
  - `_replace_collocation`;
  - a solver-native equivalent of `xr.attach_loss_weights`.
- Preserve the exact residual objective:
  - use `lossf.kfac_eval(current_params, batch)[0]` for objective when available and `legacy_kfac_eval=False`;
  - register `raw_residuals / sqrt(lossf.lref)` with KFAC;
  - for legacy residual mode, use `lossf.kfac_residuals(current_params, batch) / sqrt(lossf.lref)`;
  - keep `loss_n = sum(square(residuals))` after normalization.
- Add matching diagnostics before full replacement:
  - a helper that prepares one legacy batch and one solver batch with the same seed and reports residual-vector shape, `lref`, and objective;
  - require step-0 objective agreement before comparing optimizer steps.
- Preserve checkpoint format:
  - either keep writing the current dictionary schema from the test script;
  - or add `DIFFICESolver.save_checkpoint(...)` that writes the same schema plus optional solver metadata.
- Keep plot-only and cache-only commands independent from training runner choice.
- After compact matching succeeds, rerun the real continuation only with the solver-native path if the user explicitly wants a new long run; do not rerun the 100000-step inversion just for the refactor.
