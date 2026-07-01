function outputs = run_shelf_inversion_steps(config, steps)
%RUN_SHELF_INVERSION_STEPS Dispatch outline, mesh, parameterization, and solves.
%
% Syntax:
%   config = shelf_config('configs/amery.yaml');
%   outputs = run_shelf_inversion_steps(config, [1 2 3 4]);
%
% Required inputs:
%   config - struct from shelf_config.
%   steps  - numeric vector. Step 0 writes the BedMachine outline;
%       step 1 builds mesh; step 2 parameterizes; step 3 solves initial
%       stress balance; step 4 runs rheology-B L-curve inversion.
%
% Output:
%   outputs - struct with fields populated by the requested steps.
%
% Saved artifacts:
%   Step 0 writes Geometry/<Shelf>_Outline.exp. Steps 1-4 write
%   Results/<Shelf>_* artifacts.
%
% Assumptions:
%   Geometry/<Shelf>_Outline.exp is generated directly from BM4 in EPSG:3031.
%   BedMachine mask codes are 0 ocean, 1 ice-free land, 2 grounded ice,
%   3 floating ice, and 4 Lake Vostok. ISSM is available for steps 1-4.
%
% Examples:
%   cd examples/real_data/ISSMAdjoint/LarsenC
%   steps = [1 2];
%   LarsenC_Inversion

if nargin < 2
    steps = [1 2 3 4];
end

outputs = struct();
helpers('ensure_directory', config.shelf_dir);
helpers('ensure_directory', config.geometry_dir);
helpers('ensure_directory', config.model_dir);

if isempty(steps)
    fprintf('No %s ISSMAdjoint steps requested.\n', config.shelf_name);
    return;
end

if isfield(config, 'config_path')
    print_config_summary(config);
end

if any(steps == 0)
    outputs.outline = build_bedmachine_outline(config);
end

if any(steps == 1)
    outputs.mesh = build_mesh(config);
end

if any(steps == 2)
    helpers('bootstrap_issm_path', config.issm_dir);
    md = loadmodel(config.mesh_path);
    [outputs.parameterized, outputs.bedmachine_data, ...
        outputs.valid_velocity] = parameterize_from_bedmachine_measures( ...
        config, md);
end

if any(steps == 3)
    outputs.stressbalance = run_initial_stressbalance(config);
end

if any(steps == 4)
    [outputs.inversion, outputs.lcurve, outputs.velocity_diagnostics] = ...
        run_rheology_lcurve_inversion(config);
end
end
