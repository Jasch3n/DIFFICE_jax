# Handoff

## Objective
- Refactor and validate the new `DIFFICESolver` API for standalone PINN and XPINN workflows, with emphasis on the legacy synthetic ice-shelf PINN fixture `tests/data_pinns_test.mat`.
- Troubleshoot why standalone PINN KFAC training underperformed Adam, using the XPINN regression workflow and `loss_regression_create` as the reference for KFAC residual registration.
- Add reproducible synthetic ice-shelf scripts that train effective viscosity with Adam and KFAC, save spatial effective-viscosity plots, save loss histories, and compare optimizer cost on CPU.

## Status
- `DIFFICESolver` is available as the high-level orchestration API and is used by `tests/test_pinn_synthetic_ice_shelf.py` and `tests/test_pinn_synthetic_ice_shelf_kfac.py`.
- The standalone synthetic ice-shelf scripts use the updated solver interface:
  - `DataConfig(source=raw, sampling_counts=..., collocation_library_size="full")`
  - `ModelConfig(workflow="pinn", network=NetworkConfig(depth=6, width=30))`
  - `EquationConfig(name="ssa_iso")`
  - `LossConfig(name="iso", weights=(1.0, 0.05, 0.1))`
  - `TrainingConfig(stages=[TrainingStage(OptimizerConfig(...), iterations=...)])`
- KFAC now registers one weighted residual vector with `kfac_jax`, matching the XPINN regression pattern more closely than the earlier multi-registration attempt.
- KFAC is not available on the JAX Metal backend through `diffice_jax.optimizer.optimization.KfacOptimizer`; use CPU for KFAC.
- For a fair cost comparison, both KFAC and Adam benchmark runs were executed on CPU in `Cpu-Diffice-Env`.
- The worktree is dirty and contains many pre-existing unrelated changes. Do not revert unrelated files without explicit user direction.

## Major Changes This Session
- Added project terminology discipline through `docs/ubiquitous-language.md`; use "effective viscosity", "calving front", "dynamic boundary condition", "collocation points", and "standalone PINN" in new code/docs.
- Added or updated `DIFFICESolver` support for standalone PINN runs:
  - `diffice_jax/core/solver.py` supports PINN preparation, Adam stages, KFAC stages, prediction, and solver state.
  - `DataConfig.collocation_library_size` accepts `None`, an integer, or `"full"`.
  - `DIFFICESolver._fit_kfac_stage` prints KFAC progress every 100 steps and uses:
    ```python
    optim = KfacOptimizer(loss_fn=kfac_lossf, **config).get_optimizer()
    ```
- Updated standalone PINN KFAC loss plumbing:
  - `diffice_jax/model/pinns/loss.py` exposes `kfac_residual_terms`, `kfac_residuals`, and `kfac_objective`.
  - KFAC residual composition was checked against the scalar loss at initialization; the initial scalar-vs-residual objective difference was about `5.96e-08`.
  - Current solver KFAC registration uses a single concatenated residual vector rather than separate registrations for data/equation/calving-front groups.
- Fixed synthetic calving-front handling:
  - `diffice_jax/data/pinns/preprocessing.py` filters synthetic calving-front strips to the actual outer boundary when duplicate inward strips share the same normal.
  - For `tests/data_pinns_test.mat`, the normalized calving-front set is reduced to `301` points at the single true calving-front `x` location, with normals `[[1, 0]]`.
- Fixed floating PINN calving-front dynamic boundary condition indexing:
  - `diffice_jax/equation/eqn_iso.py` now reads effective viscosity from `sol[3]` for vanilla floating PINNs and from `sol[4]` when a surface-elevation output is present.
- Added flexible collocation library sizing:
  - `diffice_jax/data/pinns/sampling.py` no longer hard-codes `M=6000`.
  - `collocation_library_size="full"` uses the full observed normalized `(x, y)` set as the collocation library.
  - `None` preserves the historical expanded-library behavior with `M = 4 * n_data`.
- Added synthetic ice-shelf benchmark scripts:
  - `tests/test_pinn_synthetic_ice_shelf.py` runs Adam, computes relative effective-viscosity MAE, saves spatial plots, saves `.npz` diagnostics, and can generate optimizer-comparison plots.
  - `tests/test_pinn_synthetic_ice_shelf_kfac.py` runs KFAC with the requested XPINN-regression-style config.
- Benchmark results saved in `tests/figures/`:
  - KFAC 5000 CPU: `5000` steps, `441.888238 s`, `0.08837765 s/iter`, relative effective-viscosity MAE `0.05770980`, final normalized loss `4.219438e-05`.
  - Adam 50000 CPU: requested `50000`, actual `51258` steps due to Adam's built-in "continue until minimum" tail, `2517.968281 s`, `0.04912342 s/iter`, relative effective-viscosity MAE `0.06050862`, final normalized loss `3.209921e-05`.

## Updated API Usage
- Use `DIFFICESolver` as the high-level entry point. `tests/test_pinn_synthetic_ice_shelf.py` is the current best standalone PINN example.
- Minimal standalone PINN pattern:
  ```python
  from pathlib import Path
  import jax.numpy as jnp
  from scipy.io import loadmat
  import diffice_jax as djax

  raw = loadmat(Path("tests/data_pinns_test.mat"))
  solver = djax.DIFFICESolver(
      data=djax.DataConfig(
          source=raw,
          sampling_counts=jnp.array([1024, 1024, 2048, 256], dtype="int32"),
          collocation_library_size="full",
      ),
      model=djax.ModelConfig(
          workflow="pinn",
          network=djax.NetworkConfig(depth=6, width=30),
      ),
      equation=djax.EquationConfig(name="ssa_iso"),
      loss=djax.LossConfig(name="iso", weights=(1.0, 0.05, 0.1)),
      training=djax.TrainingConfig(stages=[]),
      seed=1234,
  )
  solver.prepare()
  solver.training_config = djax.TrainingConfig(
      stages=[
          djax.TrainingStage(
              djax.OptimizerConfig("adam", learning_rate=1e-3),
              iterations=5000,
          )
      ]
  )
  solver.fit()
  predictions = solver.predict()
  mu_normalized = predictions["named"]["mu"]
  ```
- To run KFAC through the same API, keep the optimizer config unchanged from the XPINN regression reference:
  ```python
  import jax.numpy as jnp
  import diffice_jax as djax

  kfac_config = dict(
      learning_rate=None,
      momentum=None,
      damping=jnp.nan,
      norm_constraint=1e-8,
      initial_damping=1,
      min_damping=1e-6,
      curvature_block_type="naive_full",
      damping_adaptation_decay=0.997,
      curvature_ema=0.997,
      inverse_update_period=1,
      num_burnin_steps=0,
      always_use_exact_qmodel_for_damping_adjustment=True,
      include_norms_in_stats=True,
  )
  solver.training_config = djax.TrainingConfig(
      stages=[
          djax.TrainingStage(
              djax.OptimizerConfig("kfac", learning_rate=None, parameters=kfac_config),
              iterations=5000,
          )
      ]
  )
  solver.fit()
  ```
- `solver.predict()["named"]` returns normalized named predictions. The synthetic ice-shelf tests multiply `mu` by `solver.state.scales[0].dynamic_scale.mu0` before comparing to `raw["mud"]`.
- Relative effective-viscosity MAE in the current scripts is:
  ```python
  mean(abs(mu_pred - mu_true)) / mean(abs(mu_true))
  ```
  It is not MSE-normalized.

## Unresolved Issues
- `tests/test_pinn_synthetic_ice_shelf_kfac.py` still has a pytest assertion intended for an earlier 1000-step KFAC expectation. In observed runs, 1000 KFAC iterations give poor effective-viscosity MAE, while 2000 and 5000 iterations improve sharply. The assertion should be revisited before relying on this as an ordinary test.
- `tests/test_pinn_synthetic_ice_shelf.py` uses a 5000-step Adam pytest path with a strict `rel_mae <= 0.05` assertion. Earlier observed 5000-step Adam results were around `0.0808` relative MAE on one run. This threshold may be too strict for a cheap deterministic test and should be recalibrated or marked as a long scientific gate.
- KFAC at 1000 iterations does not yet match the original expectation that it should beat 2500-step Adam. Longer KFAC works well, but the early-iteration discrepancy still needs deeper investigation.
- The standalone PINN KFAC residual scaling is closer to XPINN regression than before, but it is not identical in scientific weighting. In XPINN regression, data and physics residuals are assembled with explicit term-count normalization and physics/data objective weights; standalone PINN currently follows the PINN scalar loss weights.
- Adam's optimizer wrapper adds extra "burn out until minimum" iterations after the requested epoch count. Benchmark artifacts now record actual steps, but this behavior makes requested-step comparisons less clean.
- Many files in the worktree are modified, deleted, or untracked outside this session's narrow changes. Treat `git status --short` as required context before editing.
- Python bytecode caches were generated by runs and appear in `git status`; they should not be committed.
- CUDA compatibility has not been explicitly verified. The code avoids Metal-only paths for KFAC and uses JAX primitives, but future CUDA runs should verify device placement, `kfac_jax` compatibility, and memory use.

## Relevant Commands
```bash
# CPU KFAC benchmark used in this session.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && MPLCONFIGDIR=/private/tmp python tests/test_pinn_synthetic_ice_shelf_kfac.py --iterations 5000 --tag 5000_kfac'

# CPU Adam benchmark used in this session, including comparison plot generation.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && MPLCONFIGDIR=/private/tmp python tests/test_pinn_synthetic_ice_shelf.py --iterations 50000 --tag 50000_adam --compare-kfac tests/figures/test_pinn_synthetic_ice_shelf_5000_kfac.npz'

# Rebuild the optimizer comparison plot from saved .npz files without retraining.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && MPLCONFIGDIR=/private/tmp python -c "from pathlib import Path; from tests.test_pinn_synthetic_ice_shelf import save_optimizer_comparison; fig=Path(\"tests/figures\"); save_optimizer_comparison(fig/\"test_pinn_synthetic_ice_shelf_50000_adam.npz\", fig/\"test_pinn_synthetic_ice_shelf_5000_kfac.npz\", fig/\"test_pinn_synthetic_ice_shelf_optimizer_comparison.png\")"'

# Read saved benchmark metrics.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -c "from pathlib import Path; import numpy as np; fig=Path(\"tests/figures\"); names=[\"5000_kfac\",\"50000_adam\"]; [print(name, (d:=np.load(fig/f\"test_pinn_synthetic_ice_shelf_{name}.npz\"))[\"optimizer\"].item(), int(d[\"iterations\"]), float(d[\"elapsed_seconds\"]), float(d[\"seconds_per_iteration\"]), float(d[\"rel_mae\"]), float(d[\"loss_history\"][-1,0])) for name in names]"'

# Focused cheap regression that passed during the session.
zsh -lc 'eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env && python -m pytest tests/test_func.py -q'
```

## Relevant Files
- `diffice_jax/core/solver.py`: `DIFFICESolver`, typed configs, PINN/XPINN preparation, KFAC stage registration and logging.
- `diffice_jax/model/pinns/loss.py`: standalone PINN scalar loss and KFAC residual helpers.
- `diffice_jax/data/pinns/preprocessing.py`: synthetic calving-front filtering for the legacy PINN fixture.
- `diffice_jax/data/pinns/sampling.py`: flexible collocation library sizing, including `"full"`.
- `diffice_jax/equation/eqn_iso.py`: floating PINN effective-viscosity indexing in `front_eqn`.
- `tests/test_pinn_synthetic_ice_shelf.py`: Adam benchmark, solver API example, effective-viscosity MAE calculation, spatial plot generation, optimizer comparison plotting.
- `tests/test_pinn_synthetic_ice_shelf_kfac.py`: KFAC benchmark and exact KFAC config.
- `docs/ubiquitous-language.md`: project glossary for terminology.
- `tests/figures/test_pinn_synthetic_ice_shelf_5000_kfac.png`: KFAC spatial effective-viscosity plot.
- `tests/figures/test_pinn_synthetic_ice_shelf_50000_adam.png`: Adam spatial effective-viscosity plot.
- `tests/figures/test_pinn_synthetic_ice_shelf_optimizer_comparison.png`: loss-vs-iteration and loss-vs-time comparison.
- `tests/figures/test_pinn_synthetic_ice_shelf_5000_kfac.npz`: KFAC loss history, timing, and MAE.
- `tests/figures/test_pinn_synthetic_ice_shelf_50000_adam.npz`: Adam loss history, timing, and MAE.

## Next Steps
- Decide which synthetic ice-shelf runs are tests versus long scientific gates. The 5000-step Adam and 1000-step KFAC pytest assertions likely need relaxed thresholds, longer iteration counts, or `pytest.mark.slow`.
- Investigate why KFAC needs more than 1000 iterations for standalone PINN effective-viscosity inversion despite strong XPINN regression performance. Start by comparing standalone PINN residual scaling against `loss_regression_create.loss_eqn_res_sub`, `loss_ct_res_sub`, and the XPINN script's `KFAC_PHYS_OBJECTIVE_WEIGHT` / `KFAC_DATA_OBJECTIVE_WEIGHT`.
- Consider adding optional `terms` or objective-weight controls to standalone PINN KFAC residuals, so physics-only or physics-upweighted KFAC experiments can be run without changing scalar Adam loss behavior.
- Replace or disable Adam's post-epoch "burn out until minimum" behavior for benchmark mode, or expose actual iteration count consistently in optimizer history.
- Add a CUDA smoke run once a CUDA backend is available:
  - Adam should run under CUDA through JAX normally.
  - KFAC should be verified separately because `kfac_jax` curvature blocks can be memory intensive.
- Clean generated caches and unrelated dirty files before any commit or archival step.
