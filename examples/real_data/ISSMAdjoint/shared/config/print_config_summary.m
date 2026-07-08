function print_config_summary(config)
%PRINT_CONFIG_SUMMARY Print the resolved shelf inversion experiment settings.
%
% The summary appears before long ISSM runs so students can verify the loaded
% config path, data sources, mesh controls, inversion controls, and outputs.

fprintf('\nISSM adjoint shelf inversion experiment: %s\n', config.shelf_name);
fprintf('  config:    %s\n', config.config_path);
fprintf('  shelf dir: %s\n', config.shelf_dir);
fprintf('  geometry:  %s\n', config.geometry_dir);
fprintf('  results:   %s\n', config.results_dir);
fprintf('  BedMachine: %s\n', config.bedmachine_file);
fprintf('  MEaSURES:   %s\n', config.measures_file);
fprintf('  mesh hmax/hmin: %.6g / %.6g m\n', ...
    config.mesh_hmax, config.mesh_hmin);
fprintf('  regularization weight: %.6g\n', config.regularization_weight);
fprintf('  inversion steps/iters: %d / %d\n', ...
    config.invert_maxsteps, config.invert_maxiter);
fprintf('  smoke steps/iters: %d / %d\n\n', ...
    config.smoke_invert_maxsteps, config.smoke_invert_maxiter);
end
