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
%   smoke_result - struct with regularization weight, objective endpoints,
%       loss decrease, and pass/fail status.
%
% Saved artifacts:
%   Results/<Shelf>_Smoke_Control_B.mat and
%   Results/<Shelf>_smoke_rheology_B.mat.
%
% Assumptions:
%   This is a stability smoke test. It uses one regularization weight and
%   the smoke m1qn3 step budget from the config.

fprintf('Smoke inversion: 10-step MaterialsRheologyBbar solve on %s\n', ...
    config.shelf_name);

[md, smoke_result] = run_rheology_single_inversion(config, ...
    config.smoke_regularization_weight, config.smoke_invert_maxsteps, ...
    config.smoke_invert_maxiter);
smoke_result.pass = ...
    isfinite(smoke_result.loss_decrease) && smoke_result.loss_decrease > 0;

fprintf(['Smoke %s: initial J %.6g, final J %.6g, decrease %.6g, ', ...
    'pass %d\n'], config.shelf_name, smoke_result.initial_total_J, ...
    smoke_result.final_total_J, smoke_result.loss_decrease, ...
    smoke_result.pass);

save(config.smoke_result_path, 'smoke_result', '-v7.3');
save(config.smoke_inversion_path, 'md', '-v7.3');
end
