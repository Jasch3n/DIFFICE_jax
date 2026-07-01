function [md, smoke_result] = run_rheology_smoke_inversion(config)
%RUN_RHEOLOGY_SMOKE_INVERSION Run a non-production 10-step inversion smoke test.
%
% Syntax:
%   config = shelf_config('configs/ross.yaml');
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
setup = rheology_b_inversion_setup(config, md, 'smoke');

[md, lcurve] = invert_rheology_b_lcurve_core(md, setup.shelf_vertices, ...
    setup.valid_velocity, setup.options);
smoke_result = lcurve([lcurve.selected]);
smoke_result.shelf_name = config.shelf_name;
smoke_result.pass = ~smoke_result.failed && ...
    isfinite(smoke_result.loss_decrease) && smoke_result.loss_decrease > 0;
smoke_result.active_velocity_vertices = setup.active_velocity_vertices;

fprintf(['Smoke %s: initial J %.6g, final J %.6g, decrease %.6g, ', ...
    'pass %d\n'], config.shelf_name, smoke_result.initial_total_J, ...
    smoke_result.final_total_J, smoke_result.loss_decrease, ...
    smoke_result.pass);

save(config.smoke_lcurve_path, 'lcurve', 'smoke_result', '-v7.3');
save(config.smoke_inversion_path, 'md', '-v7.3');
end
