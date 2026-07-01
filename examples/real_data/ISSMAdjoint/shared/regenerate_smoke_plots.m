function diagnostics = regenerate_smoke_plots(config)
%REGENERATE_SMOKE_PLOTS Rebuild velocity diagnostic plots from smoke outputs.
%
% Syntax:
%   config = shelf_config('Ross');
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
shelf_vertices = md.mask.ocean_levelset < 0 & md.mask.ice_levelset <= 0;
valid_velocity = shelf_vertices & isfinite(md.inversion.vel_obs) & ...
    md.inversion.vel_obs >= config.min_speed_for_cost & ...
    isnan(md.stressbalance.spcvx) & isnan(md.stressbalance.spcvy);
diagnostics = helpers('summarize_velocity_misfit', md, valid_velocity);
helpers('plot_velocity_diagnostics', config, md, valid_velocity, diagnostics);
fprintf('Regenerated smoke plots for %s.\n', config.shelf_name);
end
