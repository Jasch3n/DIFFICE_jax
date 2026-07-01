# ISSMAdjoint Instructional Refactor Plan

## Goal

Refactor `examples/real_data/ISSMAdjoint/` into a teaching-oriented MATLAB
workflow where students can understand and modify one concept at a time. The
public interface should be explicit YAML config files, and the main config
loader should take paths to those YAML files:

```matlab
config = shelf_config("configs/amery.yaml");
outputs = run_shelf_inversion_steps(config, [0 1 2 3 4]);
```

The refactor should prefer a few deeper, well-named modules over many shallow
files. Entry-point scripts should remain thin and should not contain inversion
logic, numerical constants, or data-source paths.

## Current Friction Points

- `shared/shelf_config.m` mixes shelf selection, filesystem layout, hard-coded
  machine paths, numerical defaults, shelf-specific bounds, and generated
  artifact paths.
- User-editable settings are MATLAB assignments rather than an explicit config
  file contract.
- L-curve and smoke wrappers duplicate option assembly for
  `invert_rheology_b_lcurve_core`.
- Plotting routines still live behind string-dispatched `helpers(...)`, which
  makes them harder for students to discover.
- `helpers.m` contains unrelated utilities, diagnostics, and plotting code.
  This keeps the directory narrow, but hides concepts that students should be
  able to inspect directly.

## Target Layout

Keep the shared directory narrow, but make each module deeper and conceptually
clear:

```text
examples/real_data/ISSMAdjoint/
  configs/
    amery.yaml
    larsenc.yaml
    larsend.yaml
    rnflch.yaml
    ross.yaml
    README.md
  shared/
    shelf_config.m
    config_defaults.m
    config_paths.m
    print_config_summary.m
    run_shelf_inversion_steps.m
    build_bedmachine_outline.m
    build_mesh.m
    parameterize_from_bedmachine_measures.m
    set_shelf_boundary_conditions.m
    inversion_options_from_config.m
    invert_rheology_b_lcurve_core.m
    run_rheology_lcurve_inversion.m
    run_rheology_smoke_inversion.m
    plot_fem_field.m
    plot_inversion_diagnostics.m
    helpers.m
```

`helpers.m` should remain only for small generic utilities such as directory
creation, NetCDF subset reads, simple masks, and ISSM path bootstrapping.

## YAML Config Contract

Each shelf gets one YAML file. Paths inside the YAML file should be resolved
relative to the YAML file location unless they are absolute. This matches the
Python workflow config convention in `diffice_jax/workflow/config.py`.

Recommended schema:

```yaml
name: Amery

runtime:
  issm_dir: /Users/jiapchen/Software/ISSM
  np: 2

data:
  bedmachine_file: /Users/jiapchen/Research/Data/BedMachineAntarctica-v4.nc
  measures_file: /Users/jiapchen/Research/Data/MEaSURES-ice-vel/insar_antarctica_ice_velocity_450m_v2.nc
  roi_input_mat: ../../data_pinns_Amery.mat
  has_roi_input: true
  bedmachine_bounds: [1.63e6, 2.29e6, 0.56e6, 0.89e6]
  bedmachine_clip: {}

paths:
  shelf_dir: ../Amery
  geometry_dir: ../Amery/Geometry
  results_dir: ../Amery/Results

outline:
  minimum_contour_length: 10000
  minimum_contour_points: 5
  minimum_hole_area: 1.0e6
  closed_contour_tolerance: 750
  minimum_grounded_hole_fraction: 0.5
  outline_cleanup_radius: 1000
  front_probe_distances: [500, 1000, 2000, 4000, 8000]
  grounding_line_tolerance: 5000

mesh:
  initial_hmax: 10000
  hmax: 10000
  hmin: 1000
  gradation: 1.5
  adaptation_error: [0.20, 0.20]
  maxnbv: 1000000
  data_padding: 20000
  roi_padding: 50000

physics:
  initial_temperature: 263.15
  rheology_min_temperature: 273
  rheology_max_temperature: 240
  minimum_ice_thickness: 20
  grounded_friction_coefficient: 30

inversion:
  min_speed_for_cost: 1
  velocity_abs_weight: 1000
  lcurve_regularization_weights:
    logspace: [-20, -14, 9]
  initial_shelf_b_scale: 1.10
  maxsteps: 40
  maxiter: 40
  solver_residue_threshold: .nan

smoke:
  regularization_weight:
    use_lcurve_middle: true
  maxsteps: 10
  maxiter: 10
```

The loader should normalize this nested schema into the flat fields currently
expected by existing functions. That keeps the first refactor small and avoids
touching every workflow function at once.

## Public Interface

Change `shelf_config` from a shelf-name factory into a path-based loader:

```matlab
config = shelf_config(config_path)
```

Responsibilities:

1. Read `.yaml` or `.yml`.
2. Record `config.config_path` and `config.base_dir`.
3. Apply defaults from `config_defaults.m`.
4. Resolve relative input/output paths using `config_paths.m`.
5. Expand compact values such as `logspace: [-20, -14, 9]`.
6. Derive standard artifact paths such as `<Shelf>_Mesh.mat`.
7. Validate required fields and print actionable errors.

For a transition period, `shelf_config("Amery")` can emit a warning and map to
`configs/amery.yaml`, but all examples and docs should use paths.

Thin shelf wrappers should become:

```matlab
script_dir = fileparts(mfilename("fullpath"));
addpath(fullfile(fileparts(script_dir), "shared"));

if ~exist("steps", "var")
    steps = [1 2 3 4];
end

config = shelf_config(fullfile(fileparts(script_dir), "configs", "amery.yaml"));
run_shelf_inversion_steps(config, steps);
```

Alternatively, replace shelf wrappers with one generic runner:

```matlab
run_issm_adjoint("configs/amery.yaml", [1 2 3 4]);
```

Keep the shelf wrappers if they are useful for students who are new to MATLAB.

## Module Responsibilities

### Config Modules

- `shelf_config.m`: public loader; no shelf-specific hard-coded constants.
- `config_defaults.m`: default values grouped as `runtime`, `outline`, `mesh`,
  `physics`, `inversion`, and `smoke`.
- `config_paths.m`: path resolution and artifact path derivation.
- `print_config_summary.m`: display shelf name, data files, mesh settings,
  regularization grid, solver iteration counts, and output paths before long
  runs.

### Workflow Module

`run_shelf_inversion_steps.m` should stay as the readable top-level narrative:

```text
0  build outlines
1  build mesh
2  parameterize model
3  run initial stress balance
4  run rheology-B L-curve inversion
```

Add a `print_config_summary(config)` call before executing non-empty steps.

### Inversion Modules

Add `inversion_options_from_config(config, mode)` so both production and smoke
wrappers use one option mapping:

```matlab
options = inversion_options_from_config(config, "lcurve");
options = inversion_options_from_config(config, "smoke");
```

This makes the teaching interface explicit: students can inspect one file to
see exactly how YAML fields affect ISSM inversion settings.

Keep `invert_rheology_b_lcurve_core.m` as the main algorithm study file. Improve
comments inside it by labeling the major phases:

1. Choose controllable rheology-B entries.
2. Scale the initial shelf B field.
3. Build ISSM inversion parameters for one alpha.
4. Solve stress balance with m1qn3.
5. Extract objective terms and velocity diagnostics.
6. Select the L-curve corner.

Do not put file IO, plotting, or config parsing into this core file.

### Plotting Modules

Move these string-dispatched helper actions into
`plot_inversion_diagnostics.m`:

- velocity speed comparison
- velocity misfit components
- L-curve plot

Keep `plot_fem_field.m` as the low-level FEM plotting primitive. This gives
students two clear layers: one generic plotting function and one diagnostic
plotting workflow.

## Implementation Stages

### Stage 1: Introduce YAML Without Changing Behavior

1. Add `configs/*.yaml` files that reproduce the current values in
   `shelf_config.m`.
2. Add `config_defaults.m`, `config_paths.m`, and path-based `shelf_config.m`.
3. Keep output field names identical to the current flat struct.
4. Update the handoff and shelf wrappers to call `shelf_config(config_path)`.
5. Verify that `which shelf_config` and all shared function names resolve in
   MATLAB.

Recommended checks:

```bash
matlab -batch "cd('/Users/jiapchen/Software/DIFFICE_jax/examples/real_data/ISSMAdjoint'); addpath('shared'); config=shelf_config('configs/amery.yaml'); disp(config.shelf_name); disp(config.mesh_hmax)"
```

### Stage 2: Make Inversion Options Explicit

1. Add `inversion_options_from_config.m`.
2. Replace option assembly in `run_rheology_lcurve_inversion.m`.
3. Replace option assembly in `run_rheology_smoke_inversion.m`.
4. Add comments in `invert_rheology_b_lcurve_core.m` that map code blocks to
   the teaching phases above.

Recommended checks:

```bash
matlab -batch "cd('/Users/jiapchen/Software/DIFFICE_jax/examples/real_data/ISSMAdjoint'); addpath('shared'); config=shelf_config('configs/ross.yaml'); options=inversion_options_from_config(config,'smoke'); disp(options.maxsteps); disp(options.regularization_weights)"
```

### Stage 3: Split Plotting From Helpers

1. Add `plot_inversion_diagnostics.m`.
2. Move velocity diagnostic and L-curve plotting out of `helpers.m`.
3. Update `run_rheology_lcurve_inversion.m` and `regenerate_smoke_plots.m`.
4. Leave `helpers('summarize_velocity_misfit', ...)` temporarily if moving it
   would enlarge the diff too much; otherwise promote it to
   `summarize_velocity_misfit.m`.

Recommended checks:

```bash
matlab -batch "cd('/Users/jiapchen/Software/DIFFICE_jax/examples/real_data/ISSMAdjoint'); addpath('shared'); which plot_inversion_diagnostics; which plot_fem_field"
```

### Stage 4: Add Instructional Smoke Scripts

Add one manual script under `tests/manual/`, for example:

```text
tests/manual/issm_adjoint_instructional_smoke.m
```

It should run only:

1. outline generation,
2. mesh generation,
3. parameterization,
4. 10-step smoke inversion,
5. plot regeneration.

Use one shelf by default, probably Ross or Amery, and make the config path a
single variable at the top of the script.

## Validation Policy

After each stage, run a parser-level MATLAB check before any expensive solve.
After Stage 1, run `steps = [0 1]` for one shelf to confirm path resolution and
artifact derivation. After Stage 2, run one smoke inversion from an existing
parameterized model. After any change to outline, mesh, boundary conditions, or
plotting, regenerate production artifacts before using figures in reports.

Generated `.mat` and PNG outputs are large artifacts. Do not assume they should
be committed unless the project owner explicitly wants regenerated outputs
tracked.

## Open Decisions

- Whether to keep backward compatibility for `shelf_config("Amery")` or switch
  immediately to path-only usage.
- Whether MATLAB's built-in YAML support is available in the target teaching
  environment. If not, use a small documented dependency or a simple local YAML
  reader that supports only this schema.
- Whether shelf wrappers should remain as beginner-friendly entry points or be
  replaced by a single generic `run_issm_adjoint(config_path, steps)` function.
- Whether diagnostics should become separate first-class functions now, or only
  after plotting is split from `helpers.m`.
