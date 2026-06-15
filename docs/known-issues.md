# Known Issues

## XPINN KFAC RAD Sampling Can Stall After Burn-In

Observed on 2026-06-13 while running the solver-native synthetic XPINN joint-inversion comparison in `Cpu-Diffice-Env`:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python tests/test_xpinn_joint_inversion_kfac.py --iterations 50000 --tag solver_gpinn_50000 --log-rate 100 --use-gpinn
```

The run progressed normally through KFAC step 1000:

```text
KFAC step 1000: objective=5.5289e-03 scalar_loss=5.4665e-03 damping=4.97e-02
```

After step 1000, the script entered the first adaptive sampling / RAD evaluation window and produced no further stdout for an extended period. The process later exited with code `-1` and wrote no checkpoint or result artifact under `tests/figures/joint_inversion__solver_gpinn_50000/`.

Fixed on 2026-06-14:

- Root cause: `DIFFICESolver._adaptive_eval(...)` passed the full collocation-point library directly into `gov_eqn_iso(...)`. The SSA equation function uses `jax.jacfwd` and is written as a pointwise equation evaluator, so full-library input made JAX trace a huge batch-to-batch Jacobian during the first RAD probability evaluation.
- Fix: the solver-native RAD evaluator now accepts a batch, evaluates `gov_eqn_iso(...)` pointwise with `jax.vmap(...)`, and chunks the collocation library in blocks of 2048 points to keep CPU memory/compile cost bounded.
- Added explicit KFAC logs before and after RAD sampling:

```text
KFAC step 1 | adaptive_sampling=start | burn_in=0 | period=1
KFAC step 1 | adaptive_sampling=done | elapsed=3.9s
```

- Verified in `Cpu-Diffice-Env` by forcing adaptive sampling at step 1 from `examples/configs/xpinn_joint_flatbed_kfac_gpinn.yaml`. RAD completed in 3.9s on CPU; total one-step KFAC compile/execution was 53.2s.

## Config-Driven XPINN KFAC Save Fails on PosixPath Serialization

Observed on 2026-06-13 while running the config-driven XPINN joint-inversion KFAC workflow in `Cpu-Diffice-Env`:

```bash
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python examples/run_inversion.py examples/configs/xpinn_joint_flatbed_kfac.yaml
```

Training completed all configured KFAC iterations:

```text
KFAC step 5000: objective=4.8314e-05 scalar_loss=6.5585e-05 damping=9.98e-07
```

The workflow then failed during artifact saving:

```text
TypeError: Object of type PosixPath is not JSON serializable
diffice_jax/core/solver.py:360
```

Current working assumption: `DIFFICESolver.save(...)` writes solver/config metadata with `json.dump(...)`, but at least one config field has already been normalized to a `pathlib.Path`, likely `DataConfig.source` or an optimizer/artifact path.

Fixed on 2026-06-13:

- `diffice_jax.core.adapters.to_builtin(...)` now converts `Path`, JAX arrays, NumPy arrays, and NumPy scalar dtypes before writing JSON metadata.
- `tests/test_workflow_config.py::test_solver_save_serializes_workflow_paths_and_arrays` covers the save path.
- A one-step `xpinn_joint_flatbed_kfac.yaml` smoke run with `save=True` completed and wrote `/private/tmp/diffice_xpinn_config_save_smoke/solver/config.json`.
