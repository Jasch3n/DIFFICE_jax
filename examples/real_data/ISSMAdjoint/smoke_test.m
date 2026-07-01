% Short ISSM adjoint rheology-B smoke inversion across configured shelves.
%
% This script is intended for validating large-scale code changes without
% running full production inversions. It requires existing
% Results/<Shelf>_Parameterization.mat files and writes smoke inversion
% artifacts under each shelf's Results directory.
%
% Usage from MATLAB:
%   cd examples/real_data/ISSMAdjoint
%   smoke_test
%
% Optional overrides before running:
%   include_larsend = true;   % add the fifth configured shelf
%   regenerate_plots = true;  % rebuild diagnostic plots from smoke outputs

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(genpath(fullfile(script_dir, 'shared')));

if ~exist('include_larsend', 'var')
    include_larsend = false;
end
if ~exist('regenerate_plots', 'var')
    regenerate_plots = false;
end

config_files = {'amery.yaml', 'larsenc.yaml', 'rnflch.yaml', 'ross.yaml'};
if include_larsend
    config_files = {'amery.yaml', 'larsenc.yaml', 'larsend.yaml', ...
        'rnflch.yaml', 'ross.yaml'};
end

results = repmat(struct( ...
    'shelf_name', '', ...
    'pass', false, ...
    'active_velocity_vertices', NaN, ...
    'loss_decrease', NaN, ...
    'message', ''), numel(config_files), 1);

fprintf('Running ISSMAdjoint smoke inversions for %d shelf config(s).\n', ...
    numel(config_files));

for k = 1:numel(config_files)
    config_path = fullfile(script_dir, 'configs', config_files{k});
    config = shelf_config(config_path);
    results(k).shelf_name = config.shelf_name;

    fprintf('\n[%d/%d] %s smoke inversion\n', k, numel(config_files), ...
        config.shelf_name);
    try
        if ~isfile(config.parameterized_path)
            error(['Missing parameterized model: %s\nRun steps = [1 2] ', ...
                'for %s before smoke_test.'], config.parameterized_path, ...
                config.shelf_name);
        end

        [~, smoke_result] = run_rheology_smoke_inversion(config);
        results(k).pass = logical(smoke_result.pass);
        results(k).active_velocity_vertices = ...
            smoke_result.active_velocity_vertices;
        results(k).loss_decrease = smoke_result.loss_decrease;
        if ~results(k).pass
            results(k).message = 'Smoke inversion did not pass.';
        end

        if regenerate_plots
            regenerate_smoke_plots(config);
        end
    catch smoke_error
        results(k).pass = false;
        results(k).message = smoke_error.message;
        fprintf('Smoke inversion failed for %s: %s\n', ...
            config.shelf_name, smoke_error.message);
    end
end

fprintf('\nISSMAdjoint smoke inversion summary:\n');
for k = 1:numel(results)
    status = 'FAIL';
    if results(k).pass
        status = 'PASS';
    end
    fprintf('  %-8s %s  active vertices: %8g  loss decrease: %.6g\n', ...
        results(k).shelf_name, status, ...
        results(k).active_velocity_vertices, results(k).loss_decrease);
    if ~isempty(results(k).message)
        fprintf('           %s\n', results(k).message);
    end
end

if ~all([results.pass])
    error('One or more ISSMAdjoint smoke inversions failed.');
end
