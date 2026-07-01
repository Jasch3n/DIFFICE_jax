function [md, smoke_result] = run_rheology_smoke_inversion(config)
%RUN_RHEOLOGY_SMOKE_INVERSION Run a non-production 10-step inversion smoke test.
%
% Syntax:
%   config = shelf_config('Ross');
%   [md, smoke_result] = run_rheology_smoke_inversion(config);
%
% Required input:
%   config - struct from shelf_config with parameterized_path and smoke output
%       paths. The parameterized model must already exist.
%
% Outputs:
%   md - solved smoke-test model.
%   smoke_result - struct with alpha, objective endpoints, loss decrease, and
%       pass/fail status.
%
% Saved artifacts:
%   Results/<Shelf>_Smoke_Control_B.mat and
%   Results/<Shelf>_smoke_rheology_B.mat.
%
% Assumptions:
%   This is a stability smoke test, not a production L-curve inversion. It uses
%   one regularization weight and 10 m1qn3 steps.

helpers('bootstrap_issm_path', config.issm_dir);
fprintf('Smoke inversion: 10-step MaterialsRheologyBbar solve on %s\n', ...
    config.shelf_name);

md = loadmodel(config.parameterized_path);
shelf_vertices = md.mask.ocean_levelset < 0 & md.mask.ice_levelset <= 0;
valid_velocity = shelf_vertices & isfinite(md.inversion.vel_obs) & ...
    md.inversion.vel_obs >= config.min_speed_for_cost & ...
    isnan(md.stressbalance.spcvx) & isnan(md.stressbalance.spcvy);

if ~any(valid_velocity)
    error('No active velocity-cost vertices for %s smoke inversion.', ...
        config.shelf_name);
end

options = struct();
options.regularization_weights = config.smoke_regularization_weight;
options.initial_shelf_b_scale = config.initial_shelf_b_scale;
options.velocity_abs_weight = config.velocity_abs_weight;
options.np = config.np;
options.solver_residue_threshold = config.solver_residue_threshold;
options.maxsteps = config.smoke_invert_maxsteps;
options.maxiter = config.smoke_invert_maxiter;
options.rheology_min_temperature = config.rheology_min_temperature;
options.rheology_max_temperature = config.rheology_max_temperature;

[md, lcurve] = invert_rheology_b_lcurve_core(md, shelf_vertices, ...
    valid_velocity, options);
smoke_result = lcurve([lcurve.selected]);
smoke_result.shelf_name = config.shelf_name;
smoke_result.pass = ~smoke_result.failed && ...
    isfinite(smoke_result.loss_decrease) && smoke_result.loss_decrease > 0;
smoke_result.active_velocity_vertices = nnz(valid_velocity);

fprintf(['Smoke %s: initial J %.6g, final J %.6g, decrease %.6g, ', ...
    'pass %d\n'], config.shelf_name, smoke_result.initial_total_J, ...
    smoke_result.final_total_J, smoke_result.loss_decrease, ...
    smoke_result.pass);

save(config.smoke_lcurve_path, 'lcurve', 'smoke_result', '-v7.3');
save(config.smoke_inversion_path, 'md', '-v7.3');
end
