# Shared MATLAB Modules

This folder is organized by where changes belong. Interns should usually edit
`../configs/*.yaml` first and `../run_inversion.m` second. Only edit these
shared modules when changing workflow behavior.

## Folder Map

- `config/` - config loading, defaults, path resolution, and run summaries
- `preprocess/` - BedMachine/MEaSURES preprocessing, outlines, mesh, and model
  parameterization
- `inversion/` - rheology-B inversion setup and solve logic
- `plotting/` - reusable plotting primitives
- `utils/` - small generic helper functions

## Intern Edit Path

For normal experiments:

1. Copy a YAML file in `../configs/`.
2. Change `inversion.regularization_weight`, mesh settings, or output paths in
   that copied config.
3. Change `config_name` in `../run_inversion.m`.
4. Run `../run_inversion.m`.

For inversion behavior changes:

- Start in `inversion/rheology_b_inversion_setup.m` for active masks.
- Use `inversion/run_rheology_single_inversion.m` for the single-weight solve.

For preprocessing changes:

- Start in `preprocess/preprocess_inversion_data.m` for the public
  preprocessing flow and dataset path assumptions.
- Start in `preprocess/parameterize_from_bedmachine_measures.m` for BedMachine
  and MEaSURES interpolation.
- Use `preprocess/build_mesh.m` for mesh adaptation.
- Use `preprocess/build_bedmachine_outline.m` for domain outlines.

Avoid editing `utils/helpers.m` for experiment behavior. If a helper starts
containing domain behavior, promote that behavior into `preprocess/`,
`inversion/`, or `plotting/`.
