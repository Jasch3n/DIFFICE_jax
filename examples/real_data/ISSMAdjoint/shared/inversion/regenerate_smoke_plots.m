function diagnostics = regenerate_smoke_plots(config)
%REGENERATE_SMOKE_PLOTS Rebuild velocity diagnostic plots from smoke outputs.
%
% Syntax:
%   config = shelf_config('configs/ross.yaml');
%   diagnostics = regenerate_smoke_plots(config);
%
% Required input:
%   config - struct from shelf_config with smoke_inversion_path and plot paths.
%
% Output:
%   diagnostics - RMSE summary used by the regenerated plots.
%
% Saved artifacts:
%   Results/<Shelf>_velocity_speed_comparison.png and
%   Results/<Shelf>_velocity_misfit_components.png.

helpers('bootstrap_issm_path', config.issm_dir);
md = loadmodel(config.smoke_inversion_path);
setup = rheology_b_inversion_setup(config, md, 'diagnostics');
diagnostics = helpers('summarize_velocity_misfit', md, setup.valid_velocity);
helpers('plot_velocity_diagnostics', config, md, setup.valid_velocity, ...
    diagnostics);
fprintf('Regenerated smoke plots for %s.\n', config.shelf_name);
end
