function md = run_initial_stressbalance(config)
%RUN_INITIAL_STRESSBALANCE Solve and save the initial stress-balance model.
%
% Syntax:
%   md = run_initial_stressbalance(config);
%
% Required input:
%   config - struct from shelf_config with parameterized_path,
%       stressbalance_path, np, and solver_residue_threshold fields.
%
% Output:
%   md - solved ISSM model.
%
% Saved artifacts:
%   Results/<Shelf>_Stressbalance_initial.mat.
%
% Assumptions:
%   The parameterized model uses EPSG:3031 coordinates and BedMachine-derived
%   masks. ISSM solve, generic, verbose, and loadmodel are available.

helpers('bootstrap_issm_path', config.issm_dir);
fprintf('Step 3: Initial stress-balance solve for %s\n', config.shelf_name);

md = loadmodel(config.parameterized_path);
md.cluster = generic('name', oshostname, 'np', config.np);
md.verbose = verbose('solution', true);
md.stressbalance.restol = 0.01;
md.stressbalance.reltol = 0.1;
md.stressbalance.abstol = NaN;
md.settings.solver_residue_threshold = config.solver_residue_threshold;
md = solve(md, 'Stressbalance');

save(config.stressbalance_path, 'md', '-v7.3');
end
