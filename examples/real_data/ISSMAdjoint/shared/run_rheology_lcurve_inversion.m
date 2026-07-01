function [md, lcurve, velocity_diagnostics] = run_rheology_lcurve_inversion(config)
%RUN_RHEOLOGY_LCURVE_INVERSION Run IO and plots for rheology-B inversion.
%
% Syntax:
%   config = shelf_config('Amery');
%   [md, lcurve, diagnostics] = run_rheology_lcurve_inversion(config);
%
% Required input:
%   config - struct from shelf_config with parameterized_path, inversion and
%       diagnostic output paths, L-curve weights, and solver settings.
%
% Outputs:
%   md - selected inverted ISSM model.
%   lcurve - struct array for every regularization alpha.
%   velocity_diagnostics - RMSE summary over active velocity-cost vertices.
%
% Saved artifacts:
%   Results/<Shelf>_Control_B.mat, Results/<Shelf>_lcurve_rheology_B.mat,
%   Results/<Shelf>_lcurve_rheology_B.png,
%   Results/<Shelf>_velocity_speed_comparison.png,
%   Results/<Shelf>_velocity_misfit_components.png, and
%   Results/<Shelf>_velocity_misfit_diagnostics.mat.
%
% Assumptions:
%   The parameterized model is in EPSG:3031 with BedMachine mask codes already
%   converted to ISSM levelsets. Cost functions are [101 502]. ISSM m1qn3 and
%   Stressbalance solve are available.
%
% Examples:
%   cd examples/real_data/ISSMAdjoint/Amery
%   steps = [1 2 3 4];
%   Amery_Inversion

helpers('bootstrap_issm_path', config.issm_dir);
fprintf('Step 4: Invert for MaterialsRheologyBbar on %s\n', config.shelf_name);

md = loadmodel(config.parameterized_path);

% Inversion setup, following the same high-level block order as
% aashray_amery.m: observations are already on md.inversion, then masks choose
% where velocity misfit and rheology_B regularization are active.
shelf_vertices = md.mask.ocean_levelset < 0 & md.mask.ice_levelset <= 0;
fprintf('Floating-shelf vertices controlled in B inversion: %d of %d\n', ...
    nnz(shelf_vertices), md.mesh.numberofvertices);

valid_velocity = shelf_vertices & isfinite(md.inversion.vel_obs) & ...
    md.inversion.vel_obs >= config.min_speed_for_cost & ...
    isnan(md.stressbalance.spcvx) & isnan(md.stressbalance.spcvy);
fprintf('Velocity-cost vertices active in full-model shelf inversion: %d\n', ...
    nnz(valid_velocity));

options = struct();
options.regularization_weights = config.lcurve_regularization_weights;
options.initial_shelf_b_scale = config.initial_shelf_b_scale;
options.velocity_abs_weight = config.velocity_abs_weight;
options.np = config.np;
options.solver_residue_threshold = config.solver_residue_threshold;
options.maxsteps = config.invert_maxsteps;
options.maxiter = config.invert_maxiter;
options.rheology_min_temperature = config.rheology_min_temperature;
options.rheology_max_temperature = config.rheology_max_temperature;

[md, lcurve] = invert_rheology_b_lcurve_core(md, shelf_vertices, ...
    valid_velocity, options);

velocity_diagnostics = helpers('summarize_velocity_misfit', md, valid_velocity);
velocity_diagnostics.lcurve = lcurve;
fprintf('Velocity RMSE over active inversion vertices:\n');
fprintf('  vector RMSE: %.6g m/yr\n', velocity_diagnostics.vector_rmse);
fprintf('  speed RMSE:  %.6g m/yr\n', velocity_diagnostics.speed_rmse);
fprintf('  Vx RMSE:     %.6g m/yr\n', velocity_diagnostics.vx_rmse);
fprintf('  Vy RMSE:     %.6g m/yr\n', velocity_diagnostics.vy_rmse);
fprintf('  active vertices: %d\n', velocity_diagnostics.active_vertices);

helpers('plot_velocity_diagnostics', config, md, valid_velocity, ...
    velocity_diagnostics);
helpers('plot_lcurve', config, lcurve);
save(config.lcurve_path, 'lcurve', '-v7.3');
save(config.diagnostics_path, 'velocity_diagnostics', '-v7.3');

if isfield(md.results.StressbalanceSolution, 'MaterialsRheologyBbar')
    rheology_b_plot = md.results.StressbalanceSolution.MaterialsRheologyBbar;
else
    rheology_b_plot = md.materials.rheology_B;
end
plotmodel(md, 'figure', 2, ...
    'data', rheology_b_plot, ...
    'title', 'Inferred rheology B', ...
    'data', md.results.StressbalanceSolution.Vel, ...
    'title', 'Modeled velocity');

save(config.inversion_path, 'md', '-v7.3');
end
