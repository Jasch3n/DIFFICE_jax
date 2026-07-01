# ISSMAdjoint Handoff

## Purpose

`examples/real_data/ISSMAdjoint/` is now a generic MATLAB workflow for ISSM
adjoint inversions on Antarctic ice shelves. It currently supports:

- `Amery`
- `LarsenC`
- `LarsenD`
- `RnFlch`
- `Ross`

The next development stage should make this module easier to read and teach.
The main goals are:

1. Make the core inversion logic obvious to students.
2. Make inversion logic easy to tweak without touching wrappers.
3. Make mesh, refinement, regularization, and solver settings easy to change.
4. Make plotting and visualization reliable, simple, and easy to inspect.

## Current Layout

Each shelf has a thin folder:

```text
examples/real_data/ISSMAdjoint/<Shelf>/
  <Shelf>_Inversion.m
  <Shelf>_GL.m
  Geometry/
    BM2_<Shelf>_Outline.exp       # if a BM2/data-PINNs ROI exists
    <Shelf>_Outline.exp           # BedMachine-v4 BAMG outline
    <Shelf>_GL_preview.png
    <Shelf>_mesh.png
  Results/
    <Shelf>_Mesh.mat
    <Shelf>_Parameterization.mat
    <Shelf>_Smoke_Control_B.mat
    <Shelf>_smoke_rheology_B.mat
    <Shelf>_velocity_speed_comparison.png
    <Shelf>_velocity_misfit_components.png
```

Amery also has production L-curve artifacts:

```text
Results/Amery_Control_B.mat
Results/Amery_lcurve_rheology_B.mat
Results/Amery_lcurve_rheology_B.png
Results/Amery_velocity_misfit_diagnostics.mat
```

Shared code lives in:

```text
examples/real_data/ISSMAdjoint/shared/
```

Important shared files:

- `shelf_config.m` - all shelf paths and user-facing numerical settings.
- `run_shelf_inversion_steps.m` - step dispatcher used by shelf entrypoints.
- `build_bedmachine_outline.m` - direct BM4 mask outline and hole extraction.
- `build_mesh.m` - BAMG mesh creation and adaptation.
- `parameterize_from_bedmachine_measures.m` - BedMachine and MEaSURES interpolation.
- `set_shelf_boundary_conditions.m` - GL, island, and ice-front boundary conditions.
- `invert_rheology_b_lcurve_core.m` - core inversion logic.
- `run_rheology_lcurve_inversion.m` - production L-curve wrapper.
- `run_rheology_smoke_inversion.m` - 10-step stability smoke test.
- `regenerate_smoke_plots.m` - rebuild diagnostic plots from smoke outputs.
- `plot_fem_field.m` - shared FEM plotting helper.
- `helpers.m` - small utilities collected into one file.

## How To Run

From a shelf directory:

```matlab
cd examples/real_data/ISSMAdjoint/Amery
steps = [0 1 2 3 4];
Amery_Inversion
```

Step meanings:

```text
0  write ROI outline, BM4 shelf outline, holes, and GL preview
1  build/adapt BAMG mesh and save Geometry/<Shelf>_mesh.png
2  parameterize from BedMachine v4 and MEaSURES velocity
3  run initial stress balance
4  run production rheology-B L-curve inversion
```

Outline and mesh only:

```matlab
cd examples/real_data/ISSMAdjoint/Ross
steps = [0 1];
Ross_Inversion
```

Smoke inversion after mesh and parameterization:

```matlab
cd examples/real_data/ISSMAdjoint
addpath('shared')
config = shelf_config('Ross');
run_shelf_inversion_steps(config, 2);
run_rheology_smoke_inversion(config);
regenerate_smoke_plots(config);
```

Regenerate smoke plots for all shelves:

```matlab
cd examples/real_data/ISSMAdjoint
addpath('shared')
shelves = {'Amery','LarsenC','LarsenD','RnFlch','Ross'};
for i = 1:numel(shelves)
    config = shelf_config(shelves{i});
    regenerate_smoke_plots(config);
end
```

## Configuration

Most knobs students should edit are in `shared/shelf_config.m`.

Paths:

```matlab
config.issm_dir
config.bedmachine_file
config.measures_file
config.bedmachine_bounds
config.bedmachine_clip
```

Mesh controls:

```matlab
config.mesh_initial_hmax = 10000;
config.mesh_hmax = 10000;
config.mesh_hmin = 1000;
config.mesh_gradation = 1.5;
config.mesh_adaptation_error = [0.20 0.20];
config.mesh_maxnbv = 1000000;
```

Current mesh adaptation fields are MEaSURES speed and BedMachine thickness.
The current `hmax` target is 10 km, with adaptation errors loose enough that
some elements reach that maximum.

Outline controls:

```matlab
config.minimum_contour_length
config.minimum_contour_points
config.minimum_hole_area
config.minimum_grounded_hole_fraction
config.outline_cleanup_radius
```

Inversion controls:

```matlab
config.lcurve_regularization_weights = logspace(-20, -14, 9);
config.initial_shelf_b_scale = 1.10;
config.velocity_abs_weight = 1000;
config.invert_maxsteps = 40;
config.invert_maxiter = 40;
config.smoke_regularization_weight
config.smoke_invert_maxsteps = 10;
config.smoke_invert_maxiter = 10;
```

LarsenC and LarsenD are split with `bedmachine_clip` at `x = -2.0e6`:

- LarsenC: `x <= -2.0e6`
- LarsenD: `x >= -2.0e6`

## Outline And Coordinate Contract

There is no local coordinate transform in the current BM4 outline workflow.
The working outlines are built directly from BedMachine v4 EPSG:3031 grids.

BedMachine v4 mask codes are treated as:

```text
0  ocean
1  ice-free land
2  grounded ice
3  floating ice
4  Lake Vostok
```

The BAMG shelf domain is generated from `mask == 3`. Solid boundaries and
grounded-island holes are based on mask codes `1`, `2`, and `4`.

Important files:

```text
Geometry/BM2_<Shelf>_Outline.exp
Geometry/<Shelf>_Outline.exp
```

`BM2_<Shelf>_Outline.exp` is only an ROI/data-PINNs artifact. It is not the
BAMG domain. `Geometry/<Shelf>_Outline.exp` is the BedMachine-v4 outline used
by BAMG.

## Core Inversion Logic

The core inversion implementation is intentionally separated into:

```text
shared/invert_rheology_b_lcurve_core.m
```

That file is the best place for students to study or modify the actual
inversion algorithm. It mirrors the terminology in `aashray_amery.m` where
possible:

- `md.inversion = m1qn3inversion(...)`
- `md.inversion.iscontrol = 1`
- `md.inversion.control_parameters = {'MaterialsRheologyBbar'}`
- `md.inversion.cost_functions = [101 502]`
- `101` is absolute velocity misfit
- `502` is rheology-B regularization
- `md.inversion.min_parameters` and `max_parameters` use `cuffey(...)`
- ISSM solves `Stressbalance`

High-level flow inside `invert_rheology_b_lcurve_core.m`:

1. Build a control mask over floating shelf elements.
2. Scale the initial shelf rheology_B by `initial_shelf_b_scale`.
3. Loop over `regularization_weights`.
4. For each alpha, configure `m1qn3inversion`.
5. Activate velocity cost only on valid, unconstrained floating vertices.
6. Activate rheology regularization on shelf vertices.
7. Run `solve(md, 'Stressbalance')`.
8. Record objective terms, RMSE diagnostics, and loss decrease.
9. Select an L-curve corner if at least three weights succeed.
10. Return the selected model and the L-curve table.

Production wrapper:

```text
shared/run_rheology_lcurve_inversion.m
```

Smoke-test wrapper:

```text
shared/run_rheology_smoke_inversion.m
```

The wrappers should stay thin. Future teaching-oriented improvements should
put inversion changes in `invert_rheology_b_lcurve_core.m`, not in shelf
entrypoints.

## Boundary Conditions

Boundary-condition logic lives in:

```text
shared/set_shelf_boundary_conditions.m
```

Current behavior:

- Ice-ocean front vertices keep Neumann stress-balance conditions.
- Grounding-line vertices are constrained to observed MEaSURES velocity.
- Grounded-island boundary vertices are also constrained to observed velocity.
- The active velocity cost excludes constrained vertices.

This separation matters: boundary constraints are not the same as inversion
cost-function activity.

## Plotting

Main plotting helper:

```text
shared/plot_fem_field.m
```

Current behavior:

- Uses ISSM `md.mesh.elements` directly.
- Does not use `delaunay`, so it does not invent triangles across holes.
- Uses kilometers on axes.
- Labels axes as `x_{ps} [km]` and `y_{ps} [km]`.
- Uses `axis equal`.
- Turns grid off.
- Uses a top-level `FONT_SIZE = 28`.
- Omits triangles with any non-finite vertex value.

Velocity diagnostic plots are generated by `helpers('plot_velocity_diagnostics', ...)`.

Speed comparison plot:

1. Observed speed.
2. Modeled speed.
3. Relative absolute error.

RAE plotting:

- Color scale is capped at `[0, 1]`.
- Statistics show min, max, mean, and median.
- Stats text uses a monospace font.
- Stats background is semi-transparent with alpha `0.4`.

Layout:

- Amery and LarsenD are stacked vertically because they are elongated.
- LarsenC, RnFlch, and Ross are stacked horizontally because they are more square.

Recent plotting experiments to avoid:

- Do not globally keep only the largest connected component. That removed valid
  triangles inside the domain.
- Do not fill all plot NaNs with nearest-neighbor interpolation unless this is
  explicitly documented as a visualization-only smoothing step. It can hide
  missing-data structure.

## Verification Status

The current meshes and smoke inversions were regenerated after the BM4 outline
and hole-handling changes. Smoke inversions used 10 m1qn3 steps and a single
regularization weight.

Last recorded smoke-test outcomes:

```text
Amery    initial J 359.682   final J 56.6587   decrease 303.023   pass 1
LarsenC  initial J 428.5     final J 54.2001   decrease 374.3     pass 1
LarsenD  initial J 139.239   final J 34.405    decrease 104.834   pass 1
RnFlch   initial J 70459     final J 7560.71   decrease 62898.3   pass 1
Ross     initial J 20011.7   final J 503.189   decrease 19508.5   pass 1
```

Observed parameterization counts from the current meshes:

```text
Amery:
  floating vertices used by ISSM: 11263 of 12465
  ice-edge/front boundary vertices: 936
  GL boundary vertices: 4476
  grounded-island boundary vertices: 612

LarsenC:
  floating vertices used by ISSM: 11054 of 12558
  ice-edge/front boundary vertices: 1006
  GL boundary vertices: 4293
  grounded-island boundary vertices: 0

LarsenD:
  floating vertices used by ISSM: 6562 of 8068
  ice-edge/front boundary vertices: 1148
  GL boundary vertices: 3281
  grounded-island boundary vertices: 0

RnFlch:
  floating vertices used by ISSM: 40962 of 43099
  ice-edge/front boundary vertices: 1839
  GL boundary vertices: 14893
  grounded-island boundary vertices: 4927

Ross:
  floating vertices used by ISSM: 34200 of 37393
  ice-edge/front boundary vertices: 2292
  GL boundary vertices: 12663
  grounded-island boundary vertices: 3317
```

## Known Issues

Ross GL jaggedness:

- The jagged GL visible in Ross diagnostic plots is also visible in
  `Geometry/Ross_mesh.png`.
- That means it is not primarily a plotting artifact.
- It comes from the direct BedMachine mask contour used in
  `Geometry/Ross_Outline.exp`.
- A future fix should smooth or simplify the BM4-derived contour before BAMG,
  while preserving true holes and not merging floating ice through grounded
  islands.

Plotting:

- `plot_fem_field.m` currently omits triangles whose plotted field has any
  non-finite vertex. This is conservative and avoids partial triangles.
- Missing velocity samples can therefore create small white triangular gaps.
- If visualization-only filling is reintroduced, keep it explicit and do not
  let it affect RMSE or inversion statistics.

Production inversion:

- Smoke outputs exist for all shelves.
- Production L-curve outputs are complete only where explicitly generated.
- Before using figures in a report, rerun the relevant production `steps = 4`
  after any outline, mesh, BC, or plotting change.

Generated artifacts:

- `.mat` model outputs and PNGs are large generated artifacts.
- Do not assume they should be committed unless the project owner wants them
  tracked.

## Recommended Next Refactor

The next pass should focus on readability and teaching value rather than new
features.

Recommended file organization:

```text
shared/
  shelf_config.m
  build_bedmachine_outline.m
  build_mesh.m
  parameterize_from_bedmachine_measures.m
  set_shelf_boundary_conditions.m
  invert_rheology_b_lcurve_core.m
  run_rheology_lcurve_inversion.m
  run_rheology_smoke_inversion.m
  plot_fem_field.m
  plot_inversion_diagnostics.m        # suggested future split from helpers.m
  helpers.m
```

Suggested improvements:

- Add comments in `invert_rheology_b_lcurve_core.m` that map each block to the
  matching section of `aashray_amery.m`.
- Make a small `inversion_options_from_config(config)` function so students can
  see exactly which config fields affect the inversion.
- Move plotting logic out of `helpers.m` into a named plotting file.
- Add a lightweight `describe_config(config)` or `print_config_summary(config)`
  utility that prints mesh resolution, adaptation fields, alpha grid, cost
  functions, and max iteration counts before a run.
- Add a manual test script that runs only:
  1. outline generation,
  2. mesh generation,
  3. parameterization,
  4. 10-step smoke inversion,
  5. plot regeneration.
- Keep shelf entrypoints thin. A new shelf should mostly require adding one
  case in `shelf_config.m` and one small `<Shelf>_Inversion.m` wrapper.

## Useful Commands

MATLAB parser check:

```bash
matlab -batch "cd('/Users/jiapchen/Software/DIFFICE_jax/examples/real_data/ISSMAdjoint'); addpath('shared'); which shelf_config; which invert_rheology_b_lcurve_core; which plot_fem_field"
```

Regenerate Ross outline and mesh:

```matlab
cd examples/real_data/ISSMAdjoint/Ross
steps = [0 1];
Ross_Inversion
```

Run Ross smoke inversion:

```matlab
cd examples/real_data/ISSMAdjoint
addpath('shared')
config = shelf_config('Ross');
run_shelf_inversion_steps(config, 2);
run_rheology_smoke_inversion(config);
regenerate_smoke_plots(config);
```

Run Amery production inversion:

```matlab
cd examples/real_data/ISSMAdjoint/Amery
steps = [2 3 4];
Amery_Inversion
```
