function [md, lcurve, velocity_diagnostics] = run_rheology_lcurve_inversion(config)
%RUN_RHEOLOGY_LCURVE_INVERSION Run IO and plots for rheology-B inversion.
%
% Syntax:
%   config = shelf_config('configs/amery.yaml');
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

setup = rheology_b_inversion_setup(config, md, 'lcurve');
shelf_vertices = setup.shelf_vertices;
valid_velocity = setup.valid_velocity;
fprintf('Floating-shelf vertices controlled in B inversion: %d of %d\n', ...
    setup.control_vertices, md.mesh.numberofvertices);
fprintf('Velocity-cost vertices active in full-model shelf inversion: %d\n', ...
    setup.active_velocity_vertices);

[md, lcurve] = invert_rheology_b_lcurve_core(md, shelf_vertices, ...
    valid_velocity, setup.options);

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
