# XPINN Joint-Inversion Profiling And Ablation Plan

## Objective

Identify the dominant runtime costs in the XPINN joint-inversion KFAC workflow before attempting more performance work. The prior residual/diagnostic refactor produced correctness parity but only about a 1.024x speedup on a 500-step no-gPINN benchmark, so the next optimization must be driven by measured cost centers.

## Working Assumptions

- Validation and profiling should run under `Cpu-Diffice-Env`; `Metal-Env` is not usable for this benchmark because `kfac_jax` is missing there.
- The primary benchmark entry point is `tests/test_xpinn_joint_inversion_kfac.py`.
- The known parity benchmark is the 500-step flatbed no-gPINN case with `sample_count=1028`, `interface_collocation=500`, `calving_front_points=200`, default depth 6, default width 30, and seed `8132002`.
- The benchmark already reports `KFAC_SECONDS`, `KFAC_SECONDS_PER_ITER`, objective, `MU_REL_MAE`, `C_REL_MAE`, and NPZ output paths.
- Treat speedups below roughly 5% as noise unless repeated runs show low variance and identical numerical behavior.

## Measurement Rules

- Always record both `/usr/bin/time -p` real time and script-reported `KFAC_SECONDS`.
- Use unique `--tag` values for every run so output artifacts are not overwritten.
- Run at least 3 repeats for any candidate performance claim.
- Compare quality as well as runtime: final objective, loss history, `MU_REL_MAE`, and `C_REL_MAE`.
- Keep compile/setup effects visible. Do not claim steady-state speedup from a single short run unless interval timing or profiler traces confirm it.
- Prefer one-factor-at-a-time ablations first. Interactions can be tested after the dominant term is identified.

## Environment Setup

```bash
eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" && pyenv activate Cpu-Diffice-Env
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export MPLCONFIGDIR=/tmp/matplotlib-diffice-jax
```

## Phase 1: Reproduce The Known Result

Purpose: verify that the local environment reproduces the user-reported 500-step no-gPINN result before adding more runs.

```bash
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py \
  --iterations 500 \
  --tag profile_p1_legacy_nogpinn_r1 \
  --legacy-kfac-eval

/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py \
  --iterations 500 \
  --tag profile_p1_optimized_nogpinn_r1
```

Expected reference result from user run:

| Run | KFAC seconds | sec/iter | real time | final objective | MU rel MAE | C rel MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy `--legacy-kfac-eval` | 154.055801 | 0.30811160 | 167.77s | 1.39430174e-02 | 0.37200406 | 0.73339315 |
| optimized | 150.456184 | 0.30091237 | 164.38s | 1.39430174e-02 | 0.37200406 | 0.73339315 |

Artifacts already produced:

- `tests/figures/test_xpinn_joint_inversion_flatbed_limit200_nogpinn_legacy_validation.npz`
- `tests/figures/test_xpinn_joint_inversion_flatbed_limit200_nogpinn_optimized_validation.npz`

## Phase 2: Repeatability And Noise Floor

Purpose: estimate runtime variance. A claimed optimization must exceed this noise floor.

Run the optimized no-gPINN benchmark three times:

```bash
for r in 1 2 3; do
  /usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py \
    --iterations 500 \
    --tag profile_p2_optimized_nogpinn_r${r}
done
```

Run the legacy path three times only if the optimized variance is low enough to make a 2-5% difference meaningful:

```bash
for r in 1 2 3; do
  /usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py \
    --iterations 500 \
    --tag profile_p2_legacy_nogpinn_r${r} \
    --legacy-kfac-eval
done
```

Decision rule:

- If optimized `KFAC_SECONDS_PER_ITER` varies by more than 2%, do not interpret the prior 2.4% speedup as meaningful.
- If variance is below 1%, keep using 500-step runs for ablations.
- If variance is high, increase to 1000 iterations for timing-only ablations.

## Phase 3: Cost-Center Ablation Table

Purpose: identify which term or input size controls runtime.

Use optimized mode only unless explicitly testing legacy parity. Run each case once first, then repeat the most informative cases 3 times.

| ID | Purpose | Command delta | Expected interpretation |
| --- | --- | --- | --- |
| A0 | Baseline no-gPINN | `--iterations 500` | Reference cost and quality |
| A1 | Data/collocation scale | `--sample-count 512` | If sec/iter drops strongly, residual/data size matters |
| A2 | Larger data/collocation scale | `--sample-count 2048` | If superlinear increase, KFAC curvature/residual dimension dominates |
| A3 | Interface matching scale down | `--interface-points 100` | If faster, matching residuals or second derivatives are expensive |
| A4 | Interface matching scale up | `--interface-points 1000` | Confirms sensitivity to matching point count |
| A5 | Calving front scale down | `--calving-front-points 50` | Tests calving-front residual cost |
| A6 | Calving front disabled by size proxy | `--calving-front-points 0` | Tests lower bound when front residuals are empty |
| A7 | gPINN enabled | `--use-gpinn` | Quantifies gPINN derivative cost |
| A8 | gPINN collocation down | `--use-gpinn --interface-collocation 100` | Tests gPINN sensitivity to interface collocation count |
| A9 | gPINN collocation up | `--use-gpinn --interface-collocation 1000` | Confirms gPINN scaling |
| A10 | Network size down | `--depth 4 --width 20` | If much faster, network eval/AD derivative dominates |
| A11 | Network size up | `--depth 8 --width 40` | Confirms network-size scaling |
| A12 | Legacy residual path | `--legacy-kfac-eval` | Quantifies residual-only correction effect |

Concrete commands:

```bash
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a0_baseline
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a1_sample512 --sample-count 512
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a2_sample2048 --sample-count 2048
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a3_iface100 --interface-points 100
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a4_iface1000 --interface-points 1000
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a5_front50 --calving-front-points 50
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a6_front0 --calving-front-points 0
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a7_gpinn --use-gpinn
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a8_gpinn_ic100 --use-gpinn --interface-collocation 100
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a9_gpinn_ic1000 --use-gpinn --interface-collocation 1000
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a10_net_small --depth 4 --width 20
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a11_net_large --depth 8 --width 40
/usr/bin/time -p python tests/test_xpinn_joint_inversion_kfac.py --iterations 500 --tag profile_a12_legacy --legacy-kfac-eval
```

Record the table in this format:

| ID | Tag | Iterations | gPINN | sample count | interface points | front points | interface collocation | depth x width | KFAC seconds | sec/iter | real seconds | final objective | MU rel MAE | C rel MAE | delta sec/iter vs A0 | quality parity? | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |

Quality parity rule:

- For pure performance variants expected to preserve objective semantics, final objective and MAE should match baseline within run-to-run noise.
- For intentional problem-size ablations, quality may change; use runtime scaling only, not final metrics, to identify cost centers.

## Phase 4: Profiling Trace

Purpose: once Phase 3 identifies a likely hot area, capture a profiler trace for one representative baseline and one stressed case.

Recommended first traces:

- Baseline no-gPINN: `profile_a0_baseline`
- Most expensive gPINN or matching case from Phase 3

Use JAX profiler if available in the environment. If adding code is acceptable, instrument `run_kfac_experiment` with:

```python
jax.profiler.start_trace("/tmp/diffice_jax_profile")
# run 20-50 KFAC steps after one warmup step
jax.profiler.stop_trace()
```

Trace rules:

- Keep the traced segment short: 20-50 steps.
- Avoid tracing initialization and plotting.
- Use the same shapes as the benchmark case.
- Inspect whether time is dominated by XLA computations, KFAC inverse/curvature updates, residual evaluation, or host callbacks/logging.

If JAX profiler is not practical, add coarse timers around:

- data sampling and `_limit_batch`
- adaptive sampling branch
- `optim.step(...)`
- `objective_value(...)` at log intervals
- checkpoint saving
- final field collection and plotting

Coarse timing should be reported per interval, not just at process end.

## Phase 5: KFAC-Specific Ablations

Purpose: test whether KFAC internals dominate more than residual evaluation.

These require small code changes or temporary config overrides in `tests/test_xpinn_joint_inversion_kfac.py::kfac_config`.

| ID | Config change | Why it matters |
| --- | --- | --- |
| K1 | `inverse_update_period=5` | Tests whether inverse updates dominate every-step cost |
| K2 | `inverse_update_period=10` | Stronger version of K1 |
| K3 | `always_use_exact_qmodel_for_damping_adjustment=False` | Tests exact damping model overhead |
| K4 | `include_norms_in_stats=False` | Tests stats overhead |
| K5 | higher `num_burnin_steps` | Tests whether early exact curvature work is avoidable |
| K6 | alternate `curvature_block_type` if supported | Tests curvature representation cost |

Acceptance criteria for a KFAC config change:

- At least 10% sec/iter improvement on 3 repeats.
- Loss trajectory remains qualitatively similar over 500 steps.
- Final objective and MAE do not regress materially for the same iteration budget.
- If convergence slows but sec/iter improves, compare time-to-objective, not only sec/iter.

## Phase 6: Candidate Optimization Decisions

Use the ablation outcome to choose one path:

| Finding | Next implementation target |
| --- | --- |
| Runtime scales with `sample_count` | Reduce residual vector dimension for KFAC curvature or subsample residuals for KFAC only |
| Runtime scales with `interface_points` | Optimize matching residuals and second derivative computation |
| Runtime scales strongly with `--use-gpinn` | Reduce gPINN frequency, collocation count, or derivative order in KFAC |
| Runtime barely changes with residual sizes | Focus on KFAC internals, inverse update frequency, or backend limitations |
| Runtime scales with depth/width | Optimize network derivative evaluation or architecture for inversion |
| Runtime dominated by logging/checkpointing | Move diagnostics/checkpoints out of hot intervals |

## Deliverables

- A completed ablation table with all Phase 3 fields.
- Raw stdout logs for every run.
- NPZ artifacts for every run under `tests/figures/`.
- A short conclusion with one of:
  - dominant cost identified and next implementation target selected,
  - benchmark too noisy and needs longer runs,
  - KFAC internals dominate and residual refactors are not worth further effort.

## Stop Conditions

- Stop micro-optimizing residual wrappers if Phase 3 shows less than 5% sensitivity to residual/data/matching sizes.
- Stop comparing legacy vs optimized if repeated no-gPINN runs remain near 1.02x; correctness parity has already been established.
- Stop using `DIFFICESolver._fit_kfac_stage` for speed validation until its KFAC hot path is changed away from `kfac_eval(...)`.
