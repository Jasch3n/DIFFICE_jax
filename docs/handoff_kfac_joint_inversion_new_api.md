# Handoff

## Objective
- Continue development of the KFAC joint-inversion scheme in the new `DIFFICESolver` API for XPINN grounded/floating workflows.
- Use the legacy regression harness `tests/test_xpinn_regression/xpinn_regression.py` as the behavioral reference, especially its KFAC residual-vector objective for joint ice-stream/ice-shelf inversion.
- The most recent requested test target is the flatbed two-region XPINN dataset at `tests/test_xpinn_regression/flatbed_data_xpinns_regression_test.mat`; no effective-viscosity or basal-friction MAE target is required, only that the KFAC objective goes down.

## Status
- Added `tests/test_xpinn_joint_inversion_kfac.py` as a standalone reference-style KFAC test/CLI script.
- The new script does not yet use `DIFFICESolver`; it intentionally mirrors the legacy `xpinn_regression.py` KFAC path so the next agent can compare behavior before porting into the new API.
- The script uses full active two-region XPINN training with data, equation, calving-front, and matching residuals. gPINN is disabled for this test.
- The script defaults to `3000` KFAC iterations, default JAX precision, CPU backend, compact network size `depth=2`, `width=8`, and compact sample counts for runtime.
- Verified short smoke runs showed objective reduction:
  - 5-step compact-network run: initial objective `9.58212713e-01`, final objective `1.30699299e-01`, output `tests/figures/test_xpinn_joint_inversion_flatbed_smoke5_smallnet_no_gpinn.npz`.
- A full 3000-step run was not completed in this session because KFAC is still slow even after disabling gPINN and reducing sample/network sizes.

## Major Changes This Session
- Created `tests/test_xpinn_joint_inversion_kfac.py`.
- The script forces `JAX_PLATFORMS=cpu` / `JAX_PLATFORM_NAME=cpu` before importing JAX to avoid Metal plugin initialization for KFAC.
- Removed explicit `jax_enable_x64`; this test runs in default precision.
- Converted `XPINNParams` to a plain `dict` inside the test before passing it to KFAC, because JAX/KFAC treated the custom `XPINNParams` dict subclass as a non-array leaf.
- Disabled gPINN in the script with:
  - `xr.USE_GPINN = False`
  - `xr.USE_GPINN_IN_KFAC = False`
  - `KFAC_LOSS_TERMS = ("eqn", "ct", "match")`
- Added compact test defaults:
  - `DEFAULT_DEPTH = 2`
  - `DEFAULT_WIDTH = 8`
  - `DEFAULT_SAMPLE_COUNT = 32`
  - `DEFAULT_INTERFACE_POINTS = 32`
  - `DEFAULT_CALVING_FRONT_POINTS = 32`
  - `DEFAULT_INTERFACE_COLLOCATION = 32`
- Added CLI controls for iterations, log rate, network depth/width, and sample sizes.

## Unresolved Issues
- `DIFFICESolver._fit_kfac_stage` does not yet reproduce the legacy XPINN regression KFAC objective exactly. It currently calls `lossf.kfac_residuals(current_params, batch)` without term filtering, active-region controls, or separate physical/data objective weights.
- The new API does not expose the legacy KFAC controls currently hard-coded in `xpinn_regression.py`, such as:
  - `KFAC_LOSS_TERMS`
  - `KFAC_PHYS_OBJECTIVE_WEIGHT`
  - `KFAC_DATA_OBJECTIVE_WEIGHT`
  - active regions/data regions
  - matching and equation region weights
  - compact/limited batch controls for cheap smoke tests
- Full reference architecture `depth=6`, `width=30` with KFAC remains too slow for ordinary test use in this environment.
- KFAC is not available in `Metal-Env` here because `kfac_jax` is missing. The CPU environment has `kfac_jax` and was used for verification.
- The repository worktree is very dirty with many pre-existing unrelated changes and generated caches. Do not revert unrelated changes.
- Generated smoke artifacts under `tests/figures/test_xpinn_joint_inversion_flatbed_*.npz` are untracked and should be treated as diagnostics, not required source.

## Relevant Commands
```bash
# Syntax check used after the final edit.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m py_compile tests/test_xpinn_joint_inversion_kfac.py'

# Verified compact 5-step smoke run, no gPINN, default precision.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --iterations 5 --tag smoke5_smallnet_no_gpinn --log-rate 1'

# Intended long gate from the current script defaults. This was not completed in this session.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --iterations 3000 --tag 3000_kfac --log-rate 100'

# Run the pytest entry point. This will run the default 3000-step KFAC test and may take a long time.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python -m pytest tests/test_xpinn_joint_inversion_kfac.py -q'

# Restore closer-to-reference architecture for manual comparison; expected to be slow.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --iterations 5 --tag ref_arch_no_gpinn --log-rate 1 --depth 6 --width 30 --sample-count 64 --interface-points 64 --calving-front-points 64 --interface-collocation 64'
```

## Relevant Files
- `tests/test_xpinn_joint_inversion_kfac.py`: New reference-style KFAC joint-inversion test/CLI. This is the immediate comparison target for a solver-native implementation.
- `tests/test_xpinn_regression/xpinn_regression.py`: Legacy behavioral reference. Important functions/settings include `kfac_optimize`, `kfac_loss_terms`, `attach_loss_weights`, `REGRESSION_N_PT_BY_NSUB`, and `MATCH_COMPONENT_WEIGHTS_ON`.
- `diffice_jax/core/solver.py`: New API orchestration. `DIFFICESolver._fit_kfac_stage` is the main porting target.
- `diffice_jax/core/loss_terms.py`: New API XPINN loss wrapper around `loss_regression_create`.
- `diffice_jax/model/xpinns/loss.py`: Provides `loss_regression_create`, `kfac_residuals`, `kfac_objective`, term filters, region filters, matching residuals, gPINN residuals, and region term weights.
- `diffice_jax/model/xpinns/initialization.py`: Defines `XPINNParams`; the custom dict subclass is not currently treated as a pytree by KFAC/JAX in the new script.
- `diffice_jax/data/xpinns/sampling.py`: XPINN and regression samplers, including interface collocation constants `N_INTERFACE_LIBRARY` and `N_INTERFACE_COLLOCATION`.
- `tests/test_xpinn_regression/flatbed_data_xpinns_regression_test.mat`: Flatbed grounded/floating dataset used by the new script.
- `tests/figures/test_xpinn_joint_inversion_flatbed_smoke5_smallnet_no_gpinn.npz`: Verified smoke artifact with objective decrease.

## Next Steps
- Port the tested residual-vector objective into `DIFFICESolver._fit_kfac_stage` for XPINN regression/joint-inversion workflows:
  - Add solver/API controls for KFAC loss terms, physical/data objective weights, active regions, and data regions.
  - Preserve separate residual blocks equivalent to legacy `kfac_residual_vector`: physical residuals plus data residuals, each weighted before concatenation.
  - Use `lossf.lref` normalization exactly as the reference path does.
- Decide how the new API should expose compact smoke-test batch controls without mutating global constants in `diffice_jax.data.xpinns.sampling`.
- Fix or register `XPINNParams` as a proper pytree, or convert it to a plain dict before KFAC in the solver path.
- Add a solver-native XPINN KFAC script/test beside `tests/test_xpinn_joint_inversion_kfac.py` and compare initial/final KFAC objectives against the reference-style script on the same seed and compact settings.
- Once the solver-native path matches the reference-style objective behavior, decide whether the ordinary pytest should remain a 3000-step long gate or use a shorter smoke assertion with a separate manual long-run command.
