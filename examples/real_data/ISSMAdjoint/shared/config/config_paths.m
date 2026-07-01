function config = config_paths(spec, config_path, shared_dir)
%CONFIG_PATHS Normalize nested YAML into the flat workflow config struct.
%
% Paths in the YAML file are resolved relative to the YAML file location
% unless they are absolute. Generated artifact paths keep the historical
% Results/<Shelf>_* and Geometry/<Shelf>_* names.

base_dir = fileparts(config_path);
adjoint_dir = fileparts(shared_dir);
real_data_dir = fileparts(adjoint_dir);
shelf_name = canonicalShelfName(spec.name);
if isempty(shelf_name)
    error('Config field "name" is required.');
end

shelf_dir = resolveOptionalPath(base_dir, spec.paths.shelf_dir, ...
    fullfile(adjoint_dir, shelf_name));
geometry_dir = resolveOptionalPath(base_dir, spec.paths.geometry_dir, ...
    fullfile(shelf_dir, 'Geometry'));
results_dir = resolveOptionalPath(base_dir, spec.paths.results_dir, ...
    fullfile(shelf_dir, 'Results'));

config = struct();
config.config_path = config_path;
config.base_dir = base_dir;
config.shelf_name = shelf_name;
config.shared_dir = shared_dir;
config.adjoint_dir = adjoint_dir;
config.real_data_dir = real_data_dir;
config.shelf_dir = shelf_dir;
config.geometry_dir = geometry_dir;
config.results_dir = results_dir;
config.model_dir = results_dir;

config.mesh_domain_file = fullfile(geometry_dir, ...
    sprintf('%s_Outline.exp', shelf_name));
config.gl_preview_file = fullfile(geometry_dir, ...
    sprintf('%s_GL_preview.png', shelf_name));
config.mesh_preview_file = fullfile(geometry_dir, ...
    sprintf('%s_mesh.png', shelf_name));

config.issm_dir = resolvePath(base_dir, spec.runtime.issm_dir);
config.bedmachine_file = resolvePath(base_dir, spec.data.bedmachine_file);
config.measures_file = resolvePath(base_dir, spec.data.measures_file);
config.bedmachine_bounds = spec.data.bedmachine_bounds;
config.bedmachine_clip = spec.data.bedmachine_clip;

config.np = spec.runtime.np;
config.mesh_initial_hmax = spec.mesh.initial_hmax;
config.mesh_hmax = spec.mesh.hmax;
config.mesh_hmin = spec.mesh.hmin;
config.mesh_gradation = spec.mesh.gradation;
config.mesh_adaptation_error = spec.mesh.adaptation_error;
config.mesh_maxnbv = spec.mesh.maxnbv;
config.data_padding = spec.mesh.data_padding;
config.minimum_contour_length = spec.outline.minimum_contour_length;
config.minimum_contour_points = spec.outline.minimum_contour_points;
config.minimum_hole_area = spec.outline.minimum_hole_area;
config.closed_contour_tolerance = spec.outline.closed_contour_tolerance;
config.minimum_grounded_hole_fraction = ...
    spec.outline.minimum_grounded_hole_fraction;
config.outline_cleanup_radius = spec.outline.outline_cleanup_radius;
config.front_probe_distances = spec.outline.front_probe_distances;
config.grounding_line_tolerance = spec.outline.grounding_line_tolerance;

config.initial_temperature = spec.physics.initial_temperature;
config.rheology_min_temperature = spec.physics.rheology_min_temperature;
config.rheology_max_temperature = spec.physics.rheology_max_temperature;
config.minimum_ice_thickness = spec.physics.minimum_ice_thickness;
config.grounded_friction_coefficient = ...
    spec.physics.grounded_friction_coefficient;

config.min_speed_for_cost = spec.inversion.min_speed_for_cost;
config.velocity_abs_weight = spec.inversion.velocity_abs_weight;
config.regularization_weight = spec.inversion.regularization_weight;
if ~isscalar(config.regularization_weight) || ...
        ~isfinite(config.regularization_weight) || ...
        config.regularization_weight <= 0
    error('inversion.regularization_weight must be a positive finite scalar.');
end
config.initial_shelf_b_scale = spec.inversion.initial_shelf_b_scale;
config.invert_maxsteps = spec.inversion.maxsteps;
config.invert_maxiter = spec.inversion.maxiter;
config.solver_residue_threshold = spec.inversion.solver_residue_threshold;
config.smoke_regularization_weight = expandSmokeWeight( ...
    spec.smoke.regularization_weight, config.regularization_weight);
config.smoke_invert_maxsteps = spec.smoke.maxsteps;
config.smoke_invert_maxiter = spec.smoke.maxiter;

config.mesh_path = fullfile(results_dir, sprintf('%s_Mesh.mat', shelf_name));
config.parameterized_path = fullfile(results_dir, ...
    sprintf('%s_Parameterization.mat', shelf_name));
config.stressbalance_path = fullfile(results_dir, ...
    sprintf('%s_Stressbalance_initial.mat', shelf_name));
config.inversion_path = fullfile(results_dir, ...
    sprintf('%s_Control_B.mat', shelf_name));
config.smoke_inversion_path = fullfile(results_dir, ...
    sprintf('%s_Smoke_Control_B.mat', shelf_name));
config.diagnostics_path = fullfile(results_dir, ...
    sprintf('%s_velocity_misfit_diagnostics.mat', shelf_name));
config.smoke_result_path = fullfile(results_dir, ...
    sprintf('%s_smoke_rheology_B.mat', shelf_name));
config.speed_plot_path = fullfile(results_dir, ...
    sprintf('%s_velocity_speed_comparison.png', shelf_name));
config.misfit_plot_path = fullfile(results_dir, ...
    sprintf('%s_velocity_misfit_components.png', shelf_name));
end

function canonical_name = canonicalShelfName(shelf_name)
normalized = lower(regexprep(strtrim(char(shelf_name)), '[^a-zA-Z0-9]', ''));
switch normalized
    case 'amery'
        canonical_name = 'Amery';
    case {'larsenc', 'larsen'}
        canonical_name = 'LarsenC';
    case 'larsend'
        canonical_name = 'LarsenD';
    case {'rnflch', 'ronnefilchner', 'ronnefilchnerice'}
        canonical_name = 'RnFlch';
    case 'ross'
        canonical_name = 'Ross';
    otherwise
        canonical_name = strtrim(char(shelf_name));
end
end

function path = resolveOptionalPath(base_dir, configured_path, fallback)
if isempty(configured_path)
    path = fallback;
else
    path = resolvePath(base_dir, configured_path);
end
end

function path = resolvePath(base_dir, path)
path = char(path);
if isempty(path) || isAbsolutePath(path)
    return;
end
path = char(java.io.File(fullfile(base_dir, path)).getCanonicalPath());
end

function tf = isAbsolutePath(path)
if ispc
    tf = ~isempty(regexp(path, '^[A-Za-z]:[\\/]', 'once')) || ...
        startsWith(path, '\\');
else
    tf = startsWith(path, filesep);
end
end

function weight = expandSmokeWeight(value, regularization_weight)
if isnumeric(value)
    weight = value;
    return;
end
if isstruct(value) && isfield(value, 'use_inversion_weight') && ...
        value.use_inversion_weight
    weight = regularization_weight;
    return;
end
error(['smoke.regularization_weight must be numeric or ', ...
    '{use_inversion_weight: true}.']);
end
