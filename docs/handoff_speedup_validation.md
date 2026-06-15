# Handoff

## Objective
- Validate whether the XPINN/KFAC optimization work actually speeds up synthetic joint inversion workflows.
- Most recent user request: update this handoff after validation still failed to produce the promised speedup and the KFAC hot path was corrected.

## Status
- Implementation is complete enough for speed validation.
- Fast validation passed under `pyenv activate Cpu-Diffice-Env` after the correction: targeted KFAC/loss/network tests and broader cheap non-inversion tests.
- The user reported that validation runs with the previous `kfac_eval` hot path still failed to produce the promised speedup.
- User ran one legacy and one optimized no-gPINN validation using `tests/test_xpinn_joint_inversion_kfac.py` for 500 KFAC iterations under `Cpu-Diffice-Env`.
- That validation showed correctness parity but no meaningful speedup: optimized was about 1.024x faster by KFAC benchmark seconds, roughly 2.3% faster by real time.
- No broader synthetic inversion validation has been run after the residual-only correction.
- No timing baseline was preserved in code beyond the recorded legacy-vs-optimized NPZ artifacts listed below.
- Follow-up correction after failed speed validation: the synthetic KFAC hot path no longer uses full `kfac_eval` diagnostics every optimizer step. It now uses residual-only `kfac_residuals(...)`; full diagnostics are reserved for log intervals.
- `diffice_jax/core/solver.py` still prefers `kfac_eval(...)` in its generic KFAC path. Do not use that path for speed validation unless it is corrected similarly.

## Major Changes This Session
- Added `lossf.kfac_eval(params, data, terms=None, regions=None)` in `diffice_jax/model/xpinns/loss.py`; it computes scalar diagnostics, per-region diagnostics, and KFAC residuals in one traversal.
- Restored `lossf.kfac_residuals(...)` to a residual-only path after validation showed no speedup from using `kfac_eval(...)` in the KFAC step.
- Updated `tests/test_xpinn_regression/xpinn_regression.py` KFAC workflow to use residual-only evaluation inside `kfac_lossf(...)`; full loss diagnostics are recomputed only at `KFAC_LOG_RATE = 100`.
- In the synthetic workflow, KFAC optimizer aux now returns `jnp.array([loss_n, loss_n, loss_n])` during ordinary steps; the logged full `loss_info` replaces the last history entry at log intervals.
- `diffice_jax/core/solver.py` was previously updated to use `kfac_eval(...)`, but that is now known to be the wrong performance assumption for hot-path validation.
- Optimized `diffice_jax/model/xpinns/networks.py::gradf` by replacing repeated per-output `value_and_grad` calls with one `jax.jacfwd(lambda zz: f(params, zz, idx)[:6])` call.
- Added/updated tests for `kfac_eval` equivalence and the `gradf` derivative layout.

## Unresolved Issues
- Meaningful wall-clock speedup on the tested synthetic no-gPINN joint inversion case was not achieved.
- Actual speedup with gPINN enabled, larger sample limits, longer training, or the synthetic regression workflow remains unknown.
- The previous optimization hypothesis failed because full diagnostic assembly inside `kfac_eval(...)` can erase savings from sharing residual traversal.
- Full synthetic KFAC run remains expensive: `tests/test_xpinn_regression/xpinn_regression.py` defaults to `KFAC_MAXITER = 100000`, `KFAC_LOG_RATE = 100`, and adaptive sampling enabled.
- Baseline comparison for the no-gPINN 500-step joint inversion case is available in the two NPZ artifacts under `tests/figures/`.
- For any other benchmark, use prior logs, a pre-optimization checkout, or a temporary local revert.
- The generic `DIFFICESolver._fit_kfac_stage` path still needs the same residual-only treatment before it can be used as a performance validation path.
- The repo worktree was already dirty and contains many unrelated changes and tracked `__pycache__` files; do not use broad diff stats as evidence of this implementation's scope.
- `pyenv` emits `pyenv: cannot rehash: /Users/jasperchen/.pyenv/shims isn't writable`; this did not prevent tests from running.
- `ps` was blocked by sandbox permissions, so a previously launched broad pytest process could not be inspected directly.

## Relevant Commands
```bash
# Required environment for all validation commands in this repo.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env

# Syntax check that was run and passed.
python -m py_compile diffice_jax/model/xpinns/loss.py diffice_jax/model/xpinns/networks.py diffice_jax/core/solver.py tests/test_xpinn_regression/xpinn_regression.py tests/test_regression_kfac_terms.py tests/test_xpinn_network_config.py tests/test_xpinn_joint_inversion_kfac.py

# Targeted test command that was run and passed: 26 passed, 1 warning.
pytest tests/test_regression_kfac_terms.py tests/test_xpinn_network_config.py tests/test_xpinn_loss_refactor.py

# Broader cheap non-inversion command that was run and passed: 64 passed, 1 warning.
pytest tests/test_xpinn_loss_refactor.py tests/test_func.py tests/test_regression_sampling.py tests/test_regression_matching.py tests/test_regression_second_stage_loss.py tests/test_xpinn_floating_surface.py tests/test_xpinn_network_config.py tests/test_regression_kfac_terms.py tests/test_grounded_only_interface_mu_ct.py tests/test_calc_char.py

# User-run benchmark summary, not rerun by this agent:
# legacy --legacy-kfac-eval: 154.055801 KFAC seconds, 0.30811160 sec/iter, 167.77s real, objective 1.39430174e-02, mu rel MAE 0.37200406, C rel MAE 0.73339315
# optimized:                150.456184 KFAC seconds, 0.30091237 sec/iter, 164.38s real, objective 1.39430174e-02, mu rel MAE 0.37200406, C rel MAE 0.73339315
# observed speedup: about 1.024x by KFAC seconds, roughly 2.3% by real time

# Full synthetic KFAC entry point to validate absolute runtime and convergence behavior.
# This was not run after the speedup changes.
/usr/bin/time -p python tests/test_xpinn_regression/xpinn_regression.py --optimizer KFAC --mode full

# Optional branch-mode validation if the user wants staged synthetic checks.
# These were not run after the speedup changes.
/usr/bin/time -p python tests/test_xpinn_regression/xpinn_regression.py --optimizer KFAC --mode floating_region1_only
/usr/bin/time -p python tests/test_xpinn_regression/xpinn_regression.py --optimizer KFAC --mode grounded_only_interface_mu_ct
```

## Relevant Files
- `diffice_jax/model/xpinns/loss.py`: residual-only `kfac_residual_vector` starts around line 1173; diagnostic `kfac_eval` starts around line 1193; `kfac_residuals` delegates to the residual-only path around line 1331; both are attached around lines 1449-1450.
- `diffice_jax/model/xpinns/networks.py`: optimized `gradf` derivative calculation around line 131.
- `diffice_jax/core/solver.py`: generic solver KFAC path still uses `kfac_eval` around line 459; treat this as unresolved for speed validation.
- `tests/test_xpinn_regression/xpinn_regression.py`: synthetic KFAC workflow uses residual-only `kfac_eval_terms` around line 936; residual-only `kfac_lossf` starts around line 963; full diagnostics run at log intervals around line 1079.
- `tests/test_regression_kfac_terms.py`: KFAC residual/objective/equivalence tests, including `test_regression_kfac_eval_matches_residual_objective_and_diagnostics`.
- `tests/test_xpinn_network_config.py`: `test_xpinn_gradf_matches_legacy_per_output_value_and_grad_layout`.
- `docs/handoff_synthetic_inversion_validation.md`: earlier handoff for synthetic inversion validation context.
- `tests/figures/test_xpinn_joint_inversion_flatbed_limit200_nogpinn_legacy_validation.npz`: user-run 500-step no-gPINN legacy benchmark output.
- `tests/figures/test_xpinn_joint_inversion_flatbed_limit200_nogpinn_optimized_validation.npz`: user-run 500-step no-gPINN optimized benchmark output.
- Synthetic checkpoints and outputs are written under `tests/test_xpinn_regression/...` according to `CKPT_PATH` in `tests/test_xpinn_regression/xpinn_regression.py`.

## Next Steps
- Treat the 500-step no-gPINN joint inversion result as correctness parity, not a successful speedup.
- Before more micro-optimizations, profile one representative run to identify the dominant cost; likely candidates are KFAC optimizer internals, equation residuals, matching second derivatives, JAX compile/cache behavior, and logging/checkpointing overhead.
- Revalidate speed specifically with `tests/test_xpinn_joint_inversion_kfac.py` or `tests/test_xpinn_regression/xpinn_regression.py`, not `DIFFICESolver._fit_kfac_stage`, unless the generic solver KFAC path is corrected first.
- Establish a fair baseline before claiming any future speedup: use the recorded NPZ artifacts for the no-gPINN 500-step joint inversion case, the user's timing logs, a pre-optimization checkout, or a temporary local revert.
- Measure compile time separately from steady-state time; JAX first-step compile can dominate short runs.
- For each compared run, record backend, Python/JAX versions, mode, `KFAC_MAXITER`, `KFAC_LOG_RATE`, adaptive sampling settings, seed `8132002`, checkpoint path, and whether data residuals are included in KFAC.
- Compare at least: total wall time, time per logged 100 KFAC steps after first compile, loss trajectory, `kfac_phys`, `kfac_data`, and final checkpoint quality.
- If no speedup is visible after this correction, profile whether the remaining cost is dominated by equation/gPINN residuals, matching second derivatives, KFAC optimizer internals, adaptive sampling, or Python-side logging/checkpointing.
