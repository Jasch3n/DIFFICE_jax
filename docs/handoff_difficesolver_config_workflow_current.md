# Handoff

## Objective
- Continue the config-driven DIFFICESolver workflow work for synthetic PINN/XPINN inversion, especially `examples/configs/xpinn_joint_flatbed_kfac.yaml`.
- Make training configs user-facing and unambiguous: YAML-first, training-only configs; plotting/diagnostics separate; artifact parity with existing synthetic tests.
- Most recent request before this handoff: remove `continue_chunk` and add elapsed wall time plus seconds-per-iteration logging for `examples/run_inversion.py`.

## Status
- `examples/configs/xpinn_joint_flatbed_kfac.yaml` currently builds a `joint_inversion` XPINN workflow on CPU.
- The config no longer has optimizer-level `calving_front_points`, `continue_chunk`, or duplicated `per_region_networks`.
- `DIFFICESolver.predict()` now returns dimensional per-region fields and `DIFFICESolver.predict_equation_diagnostics()` returns per-region residuals/terms.
- `DIFFICESolver.save()` calls `predict()` exactly and saves its returned content to `predictions.pkl`.
- `examples/render_solver_xpinn_kfac_plots.py` can render the same plot products from saved solver artifacts using solver built-ins.
- Focused tests passed after the recent edits, but the final one-step KFAC timing smoke was interrupted before completion.

## Major Changes This Session
- Added `DIFFICESolver.load_params(path)` for loading plain `params.pkl` or checkpoint dictionaries containing `"params"`.
- Reworked `DIFFICESolver.predict()` to avoid the old heavy stitched predictor and return per-region fields: `u`, `v`, `h`, `s`, `mu`, `C`, velocity coordinates, and thickness coordinates.
- Added `DIFFICESolver.predict_equation_diagnostics(points="velocity"|"thickness"|"collocation")` for residuals, raw equation terms, and term magnitudes.
- Updated `examples/render_solver_xpinn_kfac_plots.py` to use `solver.predict()` and `solver.predict_equation_diagnostics()` instead of lower-level prediction helpers.
- Removed optimizer-level `calving_front_points`; calving-front sample counts now live only under `data.sampling_counts.calving_front`.
- Removed redundant `model.per_region_networks` from `xpinn_joint_flatbed_kfac.yaml`; `model.network.depth/width` are the default architecture.
- Added workflow parser normalization for optional per-region network aliases: `u`, `mu`, `c`/`c0` map to internal `net_u`, `net_mu`, `net_c`; `[depth, width]` list form is accepted.
- Removed `continue_chunk` from `xpinn_joint_flatbed_kfac.yaml`; config-driven continuation should use an absolute target `training.stages[].iterations`.
- KFAC now rejects stale `optimizer.parameters.calving_front_points` and `optimizer.parameters.continue_chunk` with explicit `ValueError`s.
- `use_gpinn` is a loss-level field (`loss.use_gpinn`), not an optimizer parameter.
- KFAC progress logs now include `elapsed=...s` and `seconds_per_iter=...`.
- `examples/run_inversion.py` now prints final KFAC start/final/trained iteration counts and `WORKFLOW_SECONDS_PER_ITER`.
- Updated focused tests in `tests/test_workflow_config.py`, `tests/test_xpinn_loss_refactor.py`, and legacy KFAC script compatibility points.

## Unresolved Issues
- The final one-step KFAC smoke for timing/log output was interrupted by the user before the first step completed. `py_compile` and focused tests had passed before the interrupted smoke, but the new live progress line was not observed in a completed run.
- The interrupted smoke may have left a Python process running; a sandboxed `ps` attempt returned `operation not permitted`, so process state is unknown from this session.
- Saved solver artifacts in `tests/figures/joint_inversion_kfac/solver` contain an older `config.json` with stale keys from previous runs. Regenerate artifacts after the config cleanup before treating saved config metadata as authoritative.
- Exact KFAC optimizer-state continuation requires a checkpoint containing `opt_state`, `damping`, and `key`. Current `solver.save()` artifacts include final `params.pkl`, `loss_history.pkl`, `config.json`, and predictions, but not full KFAC optimizer state.
- The old standalone `tests/test_xpinn_joint_inversion_kfac.py` still has its own local `continue_chunk` CLI logic. It is not part of the config-driven workflow, but it may confuse future readers.

## Relevant Commands
```bash
# Environment used for all verification in this session.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env

# Compile the recently touched files.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m py_compile diffice_jax/core/solver.py diffice_jax/workflow/runner.py examples/run_inversion.py examples/render_solver_xpinn_kfac_plots.py tests/test_workflow_config.py tests/test_xpinn_loss_refactor.py tests/test_xpinn_joint_inversion_kfac.py

# Focused tests that passed after the recent changes.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m pytest tests/test_workflow_config.py tests/test_xpinn_loss_refactor.py tests/test_xpinn_floating_surface.py

# Build-check the current XPINN config without training.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -c "import diffice_jax as djax; cfg=djax.load_workflow_config('examples/configs/xpinn_joint_flatbed_kfac.yaml'); solver=djax.build_solver_from_config(cfg); print(solver.model_config.network); print(solver.model_config.per_region_networks); print(solver.data_config.sampling_counts); print(solver.training_config.stages[0].optimizer.parameters)"

# Run the config-driven XPINN workflow. This is expensive on CPU.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python examples/run_inversion.py examples/configs/xpinn_joint_flatbed_kfac.yaml

# Run the same workflow without saving artifacts.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python examples/run_inversion.py examples/configs/xpinn_joint_flatbed_kfac.yaml --no-save

# Render plots from existing saved solver artifacts.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python examples/render_solver_xpinn_kfac_plots.py --config examples/configs/xpinn_joint_flatbed_kfac.yaml --solver-dir tests/figures/joint_inversion_kfac/solver --tag xpinn_joint_flatbed_kfac_config

# Solver-native prediction/diagnostic usage from saved params.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python - <<'PY'
import diffice_jax as djax

cfg = djax.load_workflow_config('examples/configs/xpinn_joint_flatbed_kfac.yaml')
solver = djax.build_solver_from_config(cfg).prepare().load_params('tests/figures/joint_inversion_kfac/solver/params.pkl')
fields = solver.predict()
diagnostics = solver.predict_equation_diagnostics(points='velocity')
print(len(fields['regions']), [r['kind'] for r in fields['regions']])
print(len(diagnostics['regions']), [r['raw_terms'].shape for r in diagnostics['regions']])
PY
```

## Relevant Files
- `examples/configs/xpinn_joint_flatbed_kfac.yaml`: current config; CPU runtime, `workflow: joint_inversion`, `iterations: 7500`, seed `8132002`, `network.depth: 6`, `network.width: 30`, no `per_region_networks`, no `continue_chunk`, no `calving_front_points`.
- `examples/run_inversion.py`: config-driven training CLI; now prints `WORKFLOW_SECONDS`, KFAC iteration range, trained iterations, and `WORKFLOW_SECONDS_PER_ITER`.
- `examples/render_solver_xpinn_kfac_plots.py`: renders field comparison, loss curve, equation residuals, and x-term ratios from saved solver artifacts using solver built-ins.
- `diffice_jax/core/solver.py`: `DIFFICESolver`, prediction/diagnostics/save behavior, KFAC controls, elapsed/per-iteration logging, stale-key validation.
- `diffice_jax/workflow/runner.py`: config-to-solver adapter, YAML/JSON normalization, sampling-count conversion, per-region network alias normalization.
- `tests/test_workflow_config.py`: focused config parser tests, including per-region network alias normalization and save/load params behavior.
- `tests/test_xpinn_loss_refactor.py`: focused XPINN loss and batch-limiting tests; calving-front truncation is no longer expected.
- `docs/handoff_synthetic_inversion_workflows_config.md`: existing broader handoff for config-driven synthetic workflows; partly updated but may lag current config values.
- `tests/figures/joint_inversion_kfac/solver/`: existing saved solver artifacts: `params.pkl`, `loss_history.pkl`, `config.json`.
- `tests/figures/joint_inversion__xpinn_joint_flatbed_kfac_config/`: rendered plot outputs from earlier successful render command.

## Next Steps
- Confirm no interrupted KFAC smoke process is still running before launching another expensive CPU training job.
- Run a very small completed config-driven KFAC smoke if feasible and verify the log line includes `elapsed=...s seconds_per_iter=...`.
- Use `examples/configs/xpinn_joint_flatbed_kfac_gpinn.yaml` for gPINN runs; the base `xpinn_joint_flatbed_kfac.yaml` has `loss.use_gpinn: false`.
- Regenerate `tests/figures/joint_inversion_kfac/solver` artifacts after the YAML cleanup so saved `config.json` no longer contains stale optimizer keys.
- For continuation from an existing run, prefer setting `training.stages[].iterations` to the absolute target iteration and using `resume_checkpoint` only when a full KFAC checkpoint is available.
