# Config-driven inversion workflow

`examples/run_inversion.py` is the config-driven entry point for running `DIFFICE_jax` inversions through the
high-level `DIFFICESolver` workflow API.  It replaces hand-edited training scripts with a YAML file that specifies
the data source, model type, loss terms, optimizer, runtime settings, and output location.

The main reference example is
`examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml`, which runs a two-region XPINN
joint inversion on the synthetic flatbed data using KFAC.
The YAML examples below follow the same settings, with relative paths written for configs stored in
`examples/synthetic_data/configs/`.

<br />

# What the workflow does

When called with a YAML config, `run_inversion.py`:

1. reads the config and applies optional JAX runtime settings before importing the full solver stack;
2. loads the `.mat` training data specified by `data.source`;
3. prints dataset and batch point counts for velocity, thickness, surface, collocation, calving-front, and interface data;
4. builds a `DIFFICESolver` from the `data`, `model`, `equation`, `loss`, and `training` sections;
5. runs each optimizer stage in order;
6. saves solver artifacts unless `--no-save` is passed.

Relative paths in the YAML file are resolved relative to the directory containing that YAML file.  For example, a
config stored in `examples/synthetic_data/configs/` should refer to the flatbed dataset as:

```yaml
data:
  source: ../flatbed_data_xpinns_regression_test.mat
```

<br />

# Requirements

Use a Python environment with `diffice_jax`, JAX, SciPy, NumPy, and PyYAML installed.  For KFAC runs, use an
environment that has `kfac_jax` available.

For GPU KFAC runs on Sherlock, use the GPU environment and submit through the provided SLURM script:

```bash
sbatch examples/submit_run_inversion.sbatch \
  examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml
```

For CPU or quick syntax/smoke runs, activate the CPU environment and call the script directly:

```bash
source /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_cpu_env/bin/activate
python examples/run_inversion.py examples/synthetic_data/configs/pinn_synthetic_ice_shelf.yaml --no-save
```

For GPU runs outside SLURM, activate the GPU environment and use a config with `runtime.jax_platform: cuda`:

```bash
source /oak/stanford/groups/cyaolai/JasperChen/VirtualEnv/DIFFICE_gpu_env/bin/activate
python examples/run_inversion.py \
  examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml
```

<br />

# YAML structure

A workflow config has these main sections:

- `name`: label printed in logs and used as the default artifact tag.
- `workflow`: public workflow name. Common values are `ice-shelf-only` for PINNs and `joint_inversion` for XPINNs.
- `runtime`: optional JAX platform and compilation-cache settings.
- `data`: input `.mat` path and sampling counts.
- `model`: PINN or XPINN model structure and network size.
- `equation`: PDE and boundary-condition choice.
- `loss`: loss type and active loss terms.
- `training`: random seed, global weights, and one or more optimizer stages.
- `artifacts`: output directory for saved parameters, predictions, and loss history.

<br />

# Example: XPINN joint inversion with KFAC

This is a compact version of the flatbed KFAC workflow.  It is based on
`examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml`.

```yaml
name: xpinn_joint_flatbed_kfac_col2056_100k_gpu
workflow: joint_inversion

runtime:
  jax_platform: cuda
  jax_compilation_cache_dir: /oak/stanford/groups/cyaolai/JasperChen/Software/Cache_jax
  jax_enable_compilation_cache: true
  jax_persistent_cache_min_compile_time_secs: 0
  jax_persistent_cache_min_entry_size_bytes: -1
  jax_persistent_cache_enable_xla_caches: xla_gpu_per_fusion_autotune_cache_dir
  jax_compilation_cache_include_metadata_in_key: false

data:
  source: ../flatbed_data_xpinns_regression_test.mat
  interface_collocation:
    library_size: 600
    sample_count: 600
  sampling_counts:
    velocity_data: [1028, 1028]
    thickness_data: [1028, 1028]
    surface_data: [1028, 1028]
    collocation: [2056, 2056]
    calving_front: [500, 500]
    matching: 500

model:
  workflow: xpinn
  regions:
    - index: 0
      kind: grounded
    - index: 1
      kind: floating
  network:
    depth: 6
    width: 30

equation:
  name: ssa_iso

loss:
  name: joint_inversion
  matching: true
  calving_front: true
  use_gpinn: false
  active_regions: [0, 1]

training:
  seed: 8132002
  global_weights:
    data: 1.0
    equation: 0.01
    calving_front: 0.05
    matching: 1.0
    gpinn: 0.0
    mu_gradient: 0.0
  stages:
    - optimizer:
        name: kfac
        learning_rate: null
        damping: .nan
        parameters:
          preset: xpinn_joint_inversion_reference
          log_rate: 200
          interface_points: all
          active_regions: [0, 1]
      iterations: 100000
      adaptive_sampling: false

artifacts:
  output_dir: ../../tests/figures/joint_inversion_kfac_col2056_100k_gpu
```

Run it with:

```bash
python examples/run_inversion.py \
  examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml
```

Submit it to SLURM with:

```bash
sbatch examples/submit_run_inversion.sbatch \
  examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml
```

<br />

# Example: short XPINN smoke run

For checking that the workflow builds, use fewer iterations and skip saving:

```yaml
name: xpinn_joint_flatbed_kfac_smoke
workflow: joint_inversion

runtime:
  jax_platform: cuda

data:
  source: ../flatbed_data_xpinns_regression_test.mat
  interface_collocation:
    library_size: 600
    sample_count: 600
  sampling_counts:
    velocity_data: [1028, 1028]
    thickness_data: [1028, 1028]
    surface_data: [1028, 1028]
    collocation: [1028, 1028]
    calving_front: [500, 500]
    matching: 500

model:
  workflow: xpinn
  regions:
    - index: 0
      kind: grounded
    - index: 1
      kind: floating
  network:
    depth: 6
    width: 30

equation:
  name: ssa_iso

loss:
  name: joint_inversion
  matching: true
  calving_front: true
  use_gpinn: true
  gpinn_weight: 0.001
  active_regions: [0, 1]

training:
  seed: 8132002
  global_weights:
    data: 1.0
    equation: 0.01
    calving_front: 0.05
    matching: 1.0
    gpinn: 0.001
    mu_gradient: 0.0
  stages:
    - optimizer:
        name: kfac
        learning_rate: null
        damping: .nan
        parameters:
          preset: xpinn_joint_inversion_reference
          log_rate: 20
          interface_points: all
          active_regions: [0, 1]
      iterations: 100
      adaptive_sampling: true
      adaptive_sampling_burn_in: 1500
      adaptive_sampling_period: 50

artifacts:
  output_dir: ../../tests/figures/joint_inversion_kfac_smoke
```

Run without writing artifacts:

```bash
python examples/run_inversion.py path/to/smoke_config.yaml --no-save
```

<br />

# Example: PINN ice-shelf-only workflow

For a regular PINN, use `workflow: ice-shelf-only`, `model.workflow: pinn`, and scalar sampling counts:

```yaml
name: pinn_synthetic_ice_shelf
workflow: ice-shelf-only

data:
  source: ../../tests/data_pinns_test.mat
  sampling_counts:
    velocity_data: 1024
    thickness_data: 1024
    surface_data: null
    collocation: 2048
    calving_front: 256
  collocation_library_size: full

model:
  workflow: pinn
  network:
    depth: 6
    width: 30

equation:
  name: ssa_iso

loss:
  name: iso
  weights: [1.0, 0.05, 0.1]

training:
  seed: 1234
  stages:
    - optimizer:
        name: adam
        learning_rate: 1.0e-3
      iterations: 5000

artifacts:
  output_dir: ../../tests/figures/ice_shelf_only__{tag}/solver
```

Run it with:

```bash
python examples/run_inversion.py examples/synthetic_data/configs/pinn_synthetic_ice_shelf.yaml
```

<br />

# Outputs

During execution, `run_inversion.py` prints machine-readable summary lines such as:

```text
DATASET_VELOCITY_DATA_POINTS=...
BATCH_COLLOCATION_POINTS=...
WORKFLOW_NAME=...
WORKFLOW_SECONDS=...
WORKFLOW_OUTPUT_DIR=...
```

For KFAC stages it also prints:

```text
WORKFLOW_START_ITERATION=...
WORKFLOW_FINAL_ITERATION=...
WORKFLOW_TRAINED_ITERATIONS=...
WORKFLOW_SECONDS_PER_ITER=...
```

If `artifacts.output_dir` is set and `--no-save` is not passed, the solver writes trained parameters, predictions,
loss history, and workflow metadata to the configured output directory.

<br />

# Notes

- `data.source`, optimizer checkpoint paths, and `artifacts.output_dir` are resolved relative to the YAML file.
- XPINN sampling counts can be per-region lists, for example `[1028, 1028]`.
- PINN sampling counts are scalar.
- Current samplers require `surface_data` to match `thickness_data` when surface data are used.
- KFAC runs should use the CPU or GPU environment that has `kfac_jax`; GPU configs should set `runtime.jax_platform: cuda`.
- Use `--no-save` for build checks, smoke runs, and timing runs where saved artifacts are not needed.
