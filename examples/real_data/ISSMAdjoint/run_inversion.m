% Template: run one config-driven ISSM adjoint rheology-B inversion.
%
% Change config_name below for a new experiment. The config file supplies the
% single inversion.regularization_weight value.
% This template intentionally does not perform L-curve analysis. If you need
% an L-curve, write a separate driver that calls the single inversion for each
% candidate regularization weight and then compares the results.

clear;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(genpath(fullfile(script_dir, 'shared')));

% Intern-editable settings.
config_name = 'amery.yaml';
inversion_maxsteps = 10;
inversion_maxiter = 10;

% Preprocessing steps needed before inversion:
%   0 - build the BedMachine-derived outline
%   1 - build mesh
%   2 - parameterize from BedMachine and MEaSURES
%
% Set preprocessing_steps = [] if Results/<Shelf>_Parameterization.mat already
% exists and you only want to rerun the inversion.
preprocessing_steps = [0 1 2];

config_path = fullfile(script_dir, 'configs', config_name);
config = shelf_config(config_path);
[~, output_tag] = fileparts(config_path);

if ~isempty(preprocessing_steps)
    preprocess_inversion_data(config, preprocessing_steps);
end

[md, inversion_result] = run_rheology_single_inversion(config, ...
    config.regularization_weight, inversion_maxsteps, inversion_maxiter);

output_path = fullfile(config.results_dir, ...
    sprintf('%s_%s_Control_B.mat', config.shelf_name, output_tag));
result_path = fullfile(config.results_dir, ...
    sprintf('%s_%s_result.mat', config.shelf_name, output_tag));

save(output_path, 'md', '-v7.3');
save(result_path, 'inversion_result', '-v7.3');

fprintf('\nSaved single-inversion outputs:\n');
fprintf('  %s\n', output_path);
fprintf('  %s\n', result_path);
