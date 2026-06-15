# Handoff

## Objective
- Next agent should run and interpret the deferred expensive synthetic inversion validation cases after the XPINN joint-inversion loss refactor and failed-test cleanup.

## Status
- Core refactor and cheap validation have passed.
- Expensive synthetic inversion validation was intentionally skipped per user instruction.
- Tests must be run under `pyenv activate Cpu-Diffice-Env`; in this shell, pyenv needs explicit initialization first.
- `pyenv` emits `pyenv: cannot rehash: /Users/jasperchen/.pyenv/shims isn't writable`; this warning did not prevent pytest from running.

## Major Changes This Session
- Implemented separate real-data XPINN joint-inversion loss path with no dummy `Mu_smp` / `C_smp` requirements.
- Added static callable XPINN loss object, frozen internal loss configs, and canonical `loss_joint_create(...)` / config-compatible `loss_regression_create(...)`.
- Added literal `JointInversionLossBuilder` and `RegressionLossBuilder`; `DIFFICESolver` now routes by `LossConfig.name` and no longer wraps joint-inversion batches with dummy inverse targets.
- Added `tests/test_xpinn_loss_refactor.py` for the new joint/regression loss contracts.
- Updated failed cheap tests to match current behavior:
  - interface-collocation tests now use `N_INTERFACE_COLLOCATION`;
  - matching tests assert symmetric interface viscosity matching and current 19-component diagnostics;
  - KFAC/gPINN tests avoid stale `loss_info` offsets;
  - `calc_char.py` has lightweight compatibility wrappers for smoke tests.

## Unresolved Issues
- Expensive synthetic inversion tasks have not been rerun after the refactor:
  - standalone PINN Adam synthetic ice-shelf inversion;
  - standalone PINN KFAC synthetic ice-shelf inversion;
  - XPINN flatbed joint-inversion KFAC workflow.
- The full suite was previously interrupted while running expensive cases; do not assume full-suite status.
- Existing worktree is very dirty with many unrelated modified/untracked files; do not revert unrelated changes.
- KFAC may require CPU backend. `tests/test_xpinn_joint_inversion_kfac.py` sets CPU env vars internally, but use `Cpu-Diffice-Env`.
- The synthetic inversion quality gates are numerical, not just unit-test pass/fail. Record `rel_mae`, wall time, and saved figures/NPZ outputs.

## Relevant Commands
```bash
cd /Users/jasperchen/Academics/Software/diffice/DIFFICE_jax
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env

# Cheap validation already passed: 62 passed, 1 warning.
pytest tests/test_xpinn_loss_refactor.py tests/test_func.py tests/test_regression_sampling.py tests/test_regression_matching.py tests/test_regression_second_stage_loss.py tests/test_xpinn_floating_surface.py tests/test_xpinn_network_config.py tests/test_regression_kfac_terms.py tests/test_grounded_only_interface_mu_ct.py tests/test_calc_char.py

# Deferred expensive synthetic inversion validation.
pytest tests/test_pinn_synthetic_ice_shelf.py
pytest tests/test_pinn_synthetic_ice_shelf_kfac.py
pytest tests/test_xpinn_joint_inversion_kfac.py

# Optional direct benchmark entry points with explicit tags.
python tests/test_pinn_synthetic_ice_shelf.py --iterations 5000 --tag 5000_adam
python tests/test_pinn_synthetic_ice_shelf_kfac.py --iterations 1000 --tag 1000_kfac

# Static checks for files touched in this work.
python -m py_compile diffice_jax/model/xpinns/loss.py diffice_jax/core/loss_terms.py diffice_jax/core/solver.py tests/test_xpinn_loss_refactor.py tests/test_xpinn_regression/calc_char.py
git diff --check -- diffice_jax/__init__.py diffice_jax/model/xpinns/loss.py tests/test_xpinn_regression/calc_char.py tests/test_regression_sampling.py tests/test_regression_matching.py tests/test_regression_kfac_terms.py tests/test_grounded_only_interface_mu_ct.py
```

## Relevant Files
- `docs/20260613_reconciled_xpinn_joint_inversion_plan.md`: controlling plan used for implementation.
- `diffice_jax/model/xpinns/loss.py`: new configs/factories/static XPINN loss object and joint/regression split.
- `diffice_jax/core/loss_terms.py`: `JointInversionLossBuilder`, `RegressionLossBuilder`, legacy `loss_joint_inversion_xpinn` wrapper.
- `diffice_jax/core/solver.py`: builder routing and removal of dummy inverse-target injection.
- `tests/test_xpinn_loss_refactor.py`: focused contract tests for joint/regression loss split.
- `tests/test_pinn_synthetic_ice_shelf.py`: Adam synthetic PINN validation; gate is `rel_mae <= 0.05`.
- `tests/test_pinn_synthetic_ice_shelf_kfac.py`: KFAC synthetic PINN validation; gate is `rel_mae <= 0.12405031`.
- `tests/test_xpinn_joint_inversion_kfac.py`: expensive flatbed XPINN joint-inversion KFAC validation.
- Expected outputs under `tests/figures/`, including `.png`, `.npz`, and KFAC checkpoint folders.

## Next Steps
- Run the three deferred expensive pytest files one at a time under `Cpu-Diffice-Env`.
- For each run, capture wall time, final assertion result, `rel_mae` metrics, and output artifact paths.
- If a synthetic inversion fails numerically, inspect whether failure is a scientific/regression-quality issue or a data-contract break from the joint/regression loss split.
- Do not change KFAC defaults (`inverse_update_period=1`, `curvature_block_type="naive_full"`) during validation unless explicitly benchmarking an alternative.
- If `tests/test_xpinn_joint_inversion_kfac.py` is too slow, first run its smallest available smoke configuration by importing its helper functions rather than changing the production loss code.
