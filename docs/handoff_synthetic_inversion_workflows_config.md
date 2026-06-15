# Handoff

## Objective
- Continue abstracting synthetic inversion training into YAML workflow configs for standalone PINN ice-shelf inversion and XPINN grounded/floating joint inversion.
- Keep training configs training-only. Diagnostics, plotting, cache rendering, and figure styling are separate modules/scripts that should consume saved outputs or checkpoints.
- Preserve exact artifact parity with the existing synthetic scripts in `tests/` before replacing them.

## Status
- A config-driven training layer exists and can build `DIFFICESolver` objects from YAML or legacy JSON.
- Current public configs:
  - `examples/configs/pinn_synthetic_ice_shelf.yaml`
  - `examples/configs/xpinn_joint_flatbed_kfac.yaml`
- Supported top-level `workflow` names are:
  - `ice-shelf-only`
  - `joint-inversion`
  - `joint-inversion-regression`
- Underscore aliases such as `joint_inversion` and `joint_inversion_regression` are accepted and canonicalized internally.
- The current XPINN YAML example uses `workflow: joint_inversion`, not `joint_inversion_regression`; this builds a non-regression joint-inversion workflow with `DataConfig.regression_workflow=False`.
- Existing synthetic scripts under `tests/` remain the parity reference and are still the commands to reproduce current `.npz`, checkpoint, figure, and printed-output formats.
- Focused tests for config loading and XPINN loss behavior have passed in `Cpu-Diffice-Env`.
- Full synthetic training parity through `examples/run_inversion.py` has not been run or proven.

## Major Changes This Session
- Added YAML/JSON config loading in `diffice_jax/workflow/config.py`.
- Added config-to-solver construction and simple training execution in `diffice_jax/workflow/runner.py`.
- Added `examples/run_inversion.py` as the config-driven training CLI.
- Added public workflow exports in `diffice_jax/__init__.py`.
- Added `PyYAML >= 6.0` to `pyproject.toml`.
- Replaced public `data.regression_workflow` usage with top-level `workflow`. The workflow loader derives the internal `DataConfig.regression_workflow` flag from `workflow == "joint-inversion-regression"` only.
- Added public XPINN loss names for both workflow variants:
  - `joint_inversion`: non-regression joint inversion, used by `examples/configs/xpinn_joint_flatbed_kfac.yaml`
  - `joint_inversion_regression`: legacy regression-workflow compatibility path, still covered by JSON config tests
- Updated `data.sampling_counts` to use named YAML fields:
  - `velocity_data`: `(u, v)` data residual points
  - `thickness_data`: `h` data residual points
  - `surface_data`: `s` data residual points; currently must equal `thickness_data` when present because the sampler uses shared coordinates for `h` and `s`
  - `collocation`: equation residual points
  - `calving_front`: calving-front boundary points; scalar for PINN, per-region list for non-regression XPINN
  - `matching`: non-regression XPINN matching-boundary points for each interface between adjacent sub-regions
- Old positional `sampling_counts` lists are still accepted for legacy JSON/Python configs.
- Public YAML configs use `all` instead of `null` for unlimited point-count controls such as `interface_points`; the workflow normalizes `all` to internal `None`.
- `eqn_weight_regions` was removed from public YAML. All sub-regions receive equation loss by default.
- XPINN interface-collocation sampling is exposed under `data.interface_collocation`:
  - `library_size`: number of regular collocation points nearest to grounded/floating interfaces kept in the interface-collocation library per adjacent sub-region side
  - `sample_count`: number of points sampled from that interface-collocation library and appended to each region's regular collocation batch
- `training.stages` is the ordered list of optimizer phases to run. Each stage owns its optimizer, iteration count, and stage-level controls such as `adaptive_sampling`, `adaptive_sampling_burn_in`, and `adaptive_sampling_period`.
- Keep optimizer-specific knobs under `training.stages[].optimizer.parameters`; keep workflow controls such as adaptive sampling at the stage level. Misplaced `optimizer.parameters.adaptive_sampling*` keys are rejected.
- XPINN global loss weights are exposed under `training.global_weights`: `data`, `equation`, `calving_front`, `matching`, `gpinn`, and `mu_gradient`.
- If an explicitly configured XPINN global weight is `0.0`, the loss emits a `RuntimeWarning` and skips that term's scalar-loss and KFAC-residual computation.
- gPINN now evaluates on the full collocation batch `data["col"]`, not only `gpinn_col` interface-tail points. Do not concatenate `data["col"]` and `gpinn_col`, because interface collocation is already appended into `data["col"]`.

## Unresolved Issues
- Exact artifact parity is incomplete. `examples/run_inversion.py` saves generic solver outputs through `DIFFICESolver.save(...)`; it does not yet reproduce the old `.npz` metadata, checkpoint schema, plot-cache files, figure names, or printed key-value lines.
- Diagnostics still need to be moved out of test scripts into standalone modules/scripts that consume checkpoints or saved solver outputs.
- `runtime.jax_platform` is the only runtime field currently expected in training YAML. Plotting and Matplotlib config should stay out of training configs.
- The old `legacy.xpinn_regression_globals.interface_library` and `interface_collocation` fields are still accepted as compatibility fallback, but public configs should use `data.interface_collocation`.
- The worktree has many unrelated existing changes. Do not revert unrelated files.

## Relevant Commands
```bash
# Activate the requested environment in non-interactive shells.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env

# Focused verification for the new config/loss layer. This was run and passed.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m pytest tests/test_workflow_config.py tests/test_xpinn_loss_refactor.py

# Syntax-check the recently edited workflow/loss files. This was run and passed.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m py_compile diffice_jax/workflow/__init__.py diffice_jax/workflow/config.py diffice_jax/workflow/runner.py diffice_jax/core/solver.py diffice_jax/core/loss_terms.py diffice_jax/model/xpinns/loss.py examples/run_inversion.py tests/test_workflow_config.py tests/test_xpinn_loss_refactor.py

# Dry-build both example YAML configs without starting training. This has been used as a lightweight check.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python - <<'PY'
from pathlib import Path
import diffice_jax as djax
for path in [Path("examples/configs/pinn_synthetic_ice_shelf.yaml"), Path("examples/configs/xpinn_joint_flatbed_kfac.yaml")]:
    cfg = djax.load_workflow_config(path)
    solver = djax.build_solver_from_config(cfg)
    print(path.name, cfg.workflow, solver.model_config.workflow, solver.loss_config.name, solver.data_config.regression_workflow, solver.data_config.sampling_counts)
PY

# Config-driven standalone PINN training. This has not been validated for artifact parity yet.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python examples/run_inversion.py examples/configs/pinn_synthetic_ice_shelf.yaml

# Config-driven XPINN joint-inversion training. This has not been validated for artifact parity yet.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python examples/run_inversion.py examples/configs/xpinn_joint_flatbed_kfac.yaml

# One-step smoke run for the XPINN YAML without saving artifacts. This was run and completed one KFAC step.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python - <<'PY'
import diffice_jax as djax
cfg = djax.load_workflow_config('examples/configs/xpinn_joint_flatbed_kfac.yaml')
cfg.training['stages'][0]['iterations'] = 1
result = djax.run_training_workflow(cfg, save=False)
print('elapsed_seconds', round(result.elapsed_seconds, 3))
print('loss_history', result.solver.loss_history)
print('sampling_counts', result.solver.data_config.sampling_counts)
print('regression_workflow', result.solver.data_config.regression_workflow)
PY

# Reproduce the current standalone PINN Adam synthetic test case and artifacts.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python tests/test_pinn_synthetic_ice_shelf.py --iterations 5000 --tag 5000_adam

# Reproduce the current standalone PINN KFAC synthetic test case and artifacts.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python tests/test_pinn_synthetic_ice_shelf_kfac.py --iterations 1000 --tag 1000_kfac

# Reproduce the current XPINN joint-inversion KFAC synthetic test case and artifacts.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python tests/test_xpinn_joint_inversion_kfac.py --iterations 5000 --tag 5000_kfac --use-gpinn --no-adaptive-sampling

# Pytest entry points for the same synthetic test cases. These are expensive.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m pytest tests/test_pinn_synthetic_ice_shelf.py::test_pinn_synthetic_ice_shelf_viscosity_inference
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m pytest tests/test_pinn_synthetic_ice_shelf_kfac.py::test_pinn_synthetic_ice_shelf_kfac_viscosity_inference
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m pytest tests/test_xpinn_joint_inversion_kfac.py::test_xpinn_joint_inversion_kfac_loss_decreases

# Render XPINN diagnostics from an existing checkpoint, separate from training.
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python tests/test_xpinn_joint_inversion_kfac.py --plot-checkpoint tests/figures/joint_inversion__5000_kfac/checkpoints/KFAC_step_5000.pkl --tag 5000_kfac
```

## Relevant Files
- `examples/configs/pinn_synthetic_ice_shelf.yaml`: standalone PINN config, `workflow: ice-shelf-only`, seed `1234`, Adam `1e-3`, `5000` iterations, named sampling counts `[1024, 1024, 2048, 256]` after conversion.
- `examples/configs/xpinn_joint_flatbed_kfac.yaml`: XPINN joint inversion config, `workflow: joint_inversion`, `loss.name: joint_inversion`, `loss.use_gpinn: false`, derived `DataConfig.regression_workflow=False`, seed `8132002`, CPU runtime, KFAC reference preset, `7500` iterations, `adaptive_sampling: false`, `adaptive_sampling_burn_in: 1000`, `adaptive_sampling_period: 50`, `interface_points: all`, `collocation: [1028, 1028]`, `calving_front: [500, 500]`, `matching: 500`, interface-collocation `library_size: 800`, `sample_count: 800`, default network `depth: 6`, `width: 30`.
- `examples/run_inversion.py`: config-driven training CLI.
- `diffice_jax/workflow/config.py`: YAML/JSON workflow config loader.
- `diffice_jax/workflow/runner.py`: config-to-solver builder, path normalization, named sampling-count conversion, `all` normalization, and training-level global-weight mapping.
- `diffice_jax/core/solver.py`: `DIFFICESolver`, `LossConfig.global_weights`, KFAC stage controls, and all-region default equation loss weighting.
- `diffice_jax/model/xpinns/loss.py`: XPINN loss construction, global-weight overrides, zero-weight skip warnings, and gPINN full-collocation behavior.
- `tests/test_workflow_config.py`: focused config tests.
- `tests/test_xpinn_loss_refactor.py`: focused XPINN loss tests, including zero-weight skip behavior and gPINN full-collocation coverage.
- Parity-reference training scripts:
  - `tests/test_pinn_synthetic_ice_shelf.py`
  - `tests/test_pinn_synthetic_ice_shelf_kfac.py`
  - `tests/test_xpinn_joint_inversion_kfac.py`
- Synthetic data:
  - `tests/data_pinns_test.mat`
  - `tests/test_xpinn_regression/flatbed_data_xpinns_regression_test.mat`
- Existing output roots:
  - `tests/figures/ice_shelf_only__{tag}/`
  - `tests/figures/joint_inversion__{tag}/`

## Next Steps
- Add artifact-parity wrappers so `examples/run_inversion.py` can emit the same `.npz` fields, checkpoint files, figure paths, and printed metrics as the existing `tests/` scripts.
- Extract diagnostics/plotting from `tests/test_pinn_synthetic_ice_shelf.py` and `tests/test_xpinn_joint_inversion_kfac.py` into standalone analysis modules that consume checkpoints or saved outputs.
- Run old and new workflows with short tags first, compare output schemas and key metrics, then run the longer synthetic parity cases.
- Keep public training YAML free of plotting/Matplotlib settings and low-level internal fields such as `eqn_weight_regions`.
