# ISSMAdjoint Shelf Inversion Workflow

This directory contains MATLAB workflows for running ISSM adjoint inversions on
real Antarctic ice shelves. The main experiment is a rheology-B inversion: ISSM
uses observed velocity and regularization to infer the shelf ice rigidity field.

## What to Edit First

Start with a config file in `configs/`. Each YAML file defines one shelf
inversion experiment:

- `configs/amery.yaml`
- `configs/larsenc.yaml`
- `configs/rnflch.yaml`
- `configs/ross.yaml`
- `configs/larsend.yaml`

Shared defaults live in `shared/config/config_defaults.m`. Copy an existing
YAML file when trying a new setting, then change the copied file instead of
editing shared MATLAB code.

Common experiment settings:

- `paths.results_dir` - where outputs are written
- `data.bedmachine_file` - BedMachine NetCDF source path
- `data.measures_file` - MEaSURES velocity NetCDF source path
- `data.bedmachine_bounds` - EPSG:3031 box that defines the BedMachine domain
- `data.bedmachine_clip` - optional `xmin/xmax/ymin/ymax` clipping inside the domain
- `mesh.hmax`, `mesh.hmin`, `mesh.adaptation_error` - mesh controls
- `inversion.regularization_weight` - single inversion alpha
- `inversion.maxsteps`, `inversion.maxiter` - production solve budget
- `smoke.maxsteps`, `smoke.maxiter` - short smoke-test solve budget

## Basic Workflow

Run MATLAB from this directory:

```matlab
cd examples/real_data/ISSMAdjoint
addpath(genpath('shared'))
config = shelf_config('configs/amery.yaml');
```

Then run the minimal single-weight template:

```matlab
run_inversion
```

Change `config_name` in `run_inversion.m` to choose another config. The script
runs preprocessing and then one rheology-B inversion using
`inversion.regularization_weight`.

Preprocessing step meanings:

- `0` - build the BedMachine-derived outline file
- `1` - build the BAMG mesh
- `2` - parameterize the model from BedMachine and MEaSURES

For a cheap check after a parameterized model already exists, set
`preprocessing_steps = []` in `run_inversion.m` or run:

```matlab
[md, smoke_result] = run_rheology_smoke_inversion(config);
```

To smoke-test the main configured shelves:

```matlab
smoke_test
```

`smoke_test.m` runs short rheology-B inversions for Amery, Larsen C,
Ronne-Filchner, and Ross. Set `include_larsend = true` before running it if you
also want the Larsen D config.

To compare regularization weights, create a series of copied config files with
different `inversion.regularization_weight` values and run `run_inversion.m`
once per config. The template saves outputs using the config filename stem, so
copied configs do not overwrite each other.

## Where the Core Logic Lives

The top-level workflow is:

- `shared/preprocess/preprocess_inversion_data.m` - prepares BedMachine and
  MEaSURES data into the parameterized ISSM model used by inversion
- `shared/preprocess/run_shelf_inversion_steps.m` - readable step dispatcher
- `shared/config/shelf_config.m` - config loader and compatibility entry point
- `shared/config/config_paths.m` - path resolution and derived artifact names
- `shared/config/print_config_summary.m` - resolved settings printed before runs

The core rheology-B inversion logic is:

- `shared/inversion/rheology_b_inversion_setup.m` - defines floating-shelf control
  vertices, active velocity-cost vertices, and mode-specific setup
- `shared/inversion/run_rheology_single_inversion.m` - one-weight inversion used
  by the intern-facing template
- `shared/inversion/run_rheology_smoke_inversion.m` - short regression-style
  inversion

If you are changing the scientific inversion behavior, inspect
`shared/inversion/rheology_b_inversion_setup.m` and
`shared/inversion/run_rheology_single_inversion.m` first.

## Data and Artifacts

Inputs are configured in YAML and defaults:

- `data.bedmachine_file` - BedMachine Antarctica v4 NetCDF
- `data.measures_file` - MEaSURES velocity NetCDF

The working inversion domain is identified from BedMachine v4 using
`data.bedmachine_bounds` and optional `data.bedmachine_clip`. There is no
separate `.mat` domain input; meshes are constructed from BedMachine only.

`data.bedmachine_clip` is an extra filter applied inside the BedMachine bounds.
Use it when the bounding box contains more than one shelf or a nearby floating
region. It accepts any subset of `xmin`, `xmax`, `ymin`, and `ymax`. For
example, Larsen C and Larsen D use the same `bedmachine_bounds`, then split the
region at `x = -2.0e6`: Larsen C uses `xmax: -2.0e6`, while Larsen D uses
`xmin: -2.0e6`.

Generated files are written under each shelf's `Geometry/` and `Results/`
directories. Full inversions can write large `.mat` files and PNG diagnostics;
do not commit regenerated artifacts unless the project owner asks for them.

## Validation Checklist

Before a full inversion:

1. Load the config and read the printed summary.
2. Run steps `[0 1]` for outline and mesh changes.
3. Run step `[2]` and inspect parameterization warnings.
4. Run `run_rheology_smoke_inversion(config)` from an existing parameterized
   model.
5. Run `run_inversion.m` with the intended single-weight config only after the
   smoke inversion reduces the objective.

The key scientific assumptions are EPSG:3031 coordinates, BedMachine v4 mask
codes, MEaSURES velocity observations in m/yr, and ISSM `cuffey` conversion for
rheology B limits.
