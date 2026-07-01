function setup = rheology_b_inversion_setup(config, md, mode)
%RHEOLOGY_B_INVERSION_SETUP Resolve masks and options for a rheology-B run.
%
% Syntax:
%   setup = rheology_b_inversion_setup(config, md, 'smoke');
%   setup = rheology_b_inversion_setup(config, md, 'diagnostics');
%
% Output:
%   setup.shelf_vertices - floating shelf vertices controlled in B inversion.
%   setup.valid_velocity - active velocity-cost vertices.
%   setup.options        - reserved for mode-specific solver options.
%
% Assumptions:
%   Floating shelf vertices have md.mask.ocean_levelset < 0 and
%   md.mask.ice_levelset <= 0. Velocity-cost vertices must be floating shelf,
%   finite observed speed, at least config.min_speed_for_cost, and unconstrained
%   by stressbalance.spcvx/spcvy.

mode = validatestring(char(mode), {'smoke', 'diagnostics'});

setup = struct();
setup.shelf_vertices = md.mask.ocean_levelset < 0 & md.mask.ice_levelset <= 0;
setup.valid_velocity = setup.shelf_vertices & ...
    isfinite(md.inversion.vel_obs) & ...
    md.inversion.vel_obs >= config.min_speed_for_cost & ...
    isnan(md.stressbalance.spcvx) & isnan(md.stressbalance.spcvy);
setup.active_velocity_vertices = nnz(setup.valid_velocity);
setup.control_vertices = nnz(setup.shelf_vertices);
setup.mode = mode;

if ~any(setup.shelf_vertices)
    error('No floating-shelf vertices found for %s.', config.shelf_name);
end
if ~any(setup.valid_velocity)
    error('No active velocity-cost vertices for %s %s run.', ...
        config.shelf_name, mode);
end

setup.options = struct();
end
