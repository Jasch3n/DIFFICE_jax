function options = inversion_options_from_config(config, mode)
%INVERSION_OPTIONS_FROM_CONFIG Map config fields to rheology-B solve options.
%
% Syntax:
%   options = inversion_options_from_config(config, 'lcurve');
%   options = inversion_options_from_config(config, 'smoke');
%
% The returned struct is the interface expected by
% invert_rheology_b_lcurve_core. This module keeps the config-to-ISSM mapping
% in one place so smoke and production runs cannot drift.

mode = validatestring(char(mode), {'lcurve', 'smoke'});

options = struct();
switch mode
    case 'lcurve'
        options.regularization_weights = config.lcurve_regularization_weights;
        options.maxsteps = config.invert_maxsteps;
        options.maxiter = config.invert_maxiter;
    case 'smoke'
        options.regularization_weights = config.smoke_regularization_weight;
        options.maxsteps = config.smoke_invert_maxsteps;
        options.maxiter = config.smoke_invert_maxiter;
end

options.initial_shelf_b_scale = config.initial_shelf_b_scale;
options.velocity_abs_weight = config.velocity_abs_weight;
options.np = config.np;
options.solver_residue_threshold = config.solver_residue_threshold;
options.rheology_min_temperature = config.rheology_min_temperature;
options.rheology_max_temperature = config.rheology_max_temperature;
end
