function outputs = preprocess_inversion_data(config, preprocessing_steps)
%PREPROCESS_INVERSION_DATA Prepare BedMachine/MEaSURES data for inversion.
%
% Syntax:
%   outputs = preprocess_inversion_data(config);
%   outputs = preprocess_inversion_data(config, [0 1 2]);
%
% This is the intern-facing preprocessing module. It turns the configured
% BedMachine and MEaSURES datasets into the parameterized ISSM model consumed
% by run_rheology_single_inversion.
%
% Step meanings:
%   0 - write the BedMachine-derived working outline file
%   1 - build the BAMG mesh
%   2 - interpolate BedMachine/MEaSURES fields onto the mesh and save the
%       parameterized model at config.parameterized_path
%
% Dataset paths:
%   Specify data.bedmachine_file and data.measures_file in the YAML config.
%   The working inversion domain is identified by data.bedmachine_bounds and
%   optional data.bedmachine_clip.
%
% Assumptions:
%   Mesh coordinates and gridded data are EPSG:3031. BedMachine v4 mask codes
%   are 0 ocean, 1 ice-free land, 2 grounded ice, 3 floating ice, and
%   4 Lake Vostok. MEaSURES velocity observations are in m/yr.

if nargin < 2
    preprocessing_steps = [0 1 2];
end

unsupported_steps = setdiff(preprocessing_steps, [0 1 2]);
if ~isempty(unsupported_steps)
    error(['preprocess_inversion_data only supports steps [0 1 2]. ', ...
        'Unsupported step(s): %s'], mat2str(unsupported_steps));
end

outputs = struct();
helpers('ensure_directory', config.shelf_dir);
helpers('ensure_directory', config.geometry_dir);
helpers('ensure_directory', config.model_dir);

fprintf('Preprocessing inversion data for %s\n', config.shelf_name);
fprintf('  BedMachine: %s\n', config.bedmachine_file);
fprintf('  MEaSURES:   %s\n', config.measures_file);
fprintf('  output parameterized model: %s\n', config.parameterized_path);

if isempty(preprocessing_steps)
    fprintf('No preprocessing steps requested for %s.\n', config.shelf_name);
    return;
end

if any(preprocessing_steps == 0)
    outputs.outline = build_bedmachine_outline(config);
end

if any(preprocessing_steps == 1)
    outputs.mesh = build_mesh(config);
end

if any(preprocessing_steps == 2)
    helpers('bootstrap_issm_path', config.issm_dir);
    md = loadmodel(config.mesh_path);
    [outputs.parameterized, outputs.bedmachine_data, ...
        outputs.valid_velocity] = parameterize_from_bedmachine_measures( ...
        config, md);
end
end
