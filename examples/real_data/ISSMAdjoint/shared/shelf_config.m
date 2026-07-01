function config = shelf_config(shelf_name)
%SHELF_CONFIG Return paths and parameters for an ISSM adjoint shelf run.
%
% Syntax:
%   config = shelf_config('Amery');
%   config = shelf_config('Ross');
%
% Required input:
%   shelf_name - shelf key, currently 'Amery', 'LarsenC', 'LarsenD',
%       'RnFlch', or 'Ross'. New shelves can be added by extending the
%       input-file switch.
%
% Output:
%   config - struct with shelf paths, BedMachine/MEaSURES paths, ROI and
%       mesh outline filenames, translation settings, mesh controls,
%       inversion controls, and generated artifact paths.
%
% Saved artifacts:
%   None. Downstream functions write Geometry/BM2_<Shelf>_Outline.exp,
%   Geometry/<Shelf>_Outline.exp, Results/<Shelf>_*.mat, and diagnostic PNGs.
%
% Assumptions:
%   BedMachine v4 outlines are generated directly in EPSG:3031 from the mask
%   field. BedMachine mask codes are 0 ocean, 1 ice-free land, 2 grounded ice,
%   3 floating ice, and 4 Lake Vostok. ISSM is available at config.issm_dir
%   before mesh/solve steps.
%
% Examples:
%   cd examples/real_data/ISSMAdjoint/Amery
%   steps = [1 2 3 4];
%   Amery_Inversion
%
%   config = shelf_config('Ross');
%   create_roi_outline_from_mat(config);
%   build_bedmachine_outline(config);

if nargin < 1 || isempty(shelf_name)
    error('shelf_config requires a shelf name.');
end

canonical_name = canonicalShelfName(shelf_name);
shared_dir = fileparts(mfilename('fullpath'));
adjoint_dir = fileparts(shared_dir);
real_data_dir = fileparts(adjoint_dir);
shelf_dir = fullfile(adjoint_dir, canonical_name);
geometry_dir = fullfile(shelf_dir, 'Geometry');
results_dir = fullfile(shelf_dir, 'Results');

switch lower(canonical_name)
    case 'amery'
        input_mat_name = 'data_pinns_Amery.mat';
        bedmachine_bounds = [1.63e6 2.29e6 0.56e6 0.89e6];
        bedmachine_clip = struct();
        has_roi_input = true;
    case 'larsenc'
        input_mat_name = 'data_pinns_LarsenC.mat';
        bedmachine_bounds = [-2.38e6 -1.55e6 0.88e6 1.34e6];
        bedmachine_clip = struct('xmax', -2.0e6);
        has_roi_input = true;
    case 'larsend'
        input_mat_name = '';
        bedmachine_bounds = [-2.38e6 -1.55e6 0.88e6 1.34e6];
        bedmachine_clip = struct('xmin', -2.0e6);
        has_roi_input = false;
    case 'rnflch'
        input_mat_name = 'data_xpinns_RnFlch.mat';
        bedmachine_bounds = [-1.56e6 -0.48e6 0.08e6 1.10e6];
        bedmachine_clip = struct();
        has_roi_input = true;
    case 'ross'
        input_mat_name = 'data_xpinns_Ross.mat';
        bedmachine_bounds = [-0.65e6 0.45e6 -1.40e6 -0.38e6];
        bedmachine_clip = struct();
        has_roi_input = true;
    otherwise
        error('Unsupported shelf "%s". Extend shelf_config.m for new shelves.', ...
            shelf_name);
end

config = struct();
config.shelf_name = canonical_name;
config.shared_dir = shared_dir;
config.adjoint_dir = adjoint_dir;
config.real_data_dir = real_data_dir;
config.shelf_dir = shelf_dir;
config.geometry_dir = geometry_dir;
config.results_dir = results_dir;
config.model_dir = results_dir;
if isempty(input_mat_name)
    config.input_mat_file = '';
else
    config.input_mat_file = fullfile(real_data_dir, input_mat_name);
end
config.has_roi_input = has_roi_input;
config.roi_outline_file = fullfile(geometry_dir, ...
    sprintf('BM2_%s_Outline.exp', canonical_name));
config.mesh_domain_file = fullfile(geometry_dir, ...
    sprintf('%s_Outline.exp', canonical_name));
config.gl_preview_file = fullfile(geometry_dir, ...
    sprintf('%s_GL_preview.png', canonical_name));
config.mesh_preview_file = fullfile(geometry_dir, ...
    sprintf('%s_mesh.png', canonical_name));

config.issm_dir = '/Users/jiapchen/Software/ISSM';
config.bedmachine_file = ...
    '/Users/jiapchen/Research/Data/BedMachineAntarctica-v4.nc';
config.measures_file = ...
    ['/Users/jiapchen/Research/Data/MEaSURES-ice-vel/', ...
     'insar_antarctica_ice_velocity_450m_v2.nc'];

config.bedmachine_bounds = bedmachine_bounds;
config.bedmachine_clip = bedmachine_clip;

config.np = 2;
config.mesh_initial_hmax = 10000;
config.mesh_hmax = 10000;
config.mesh_hmin = 1000;
config.mesh_gradation = 1.5;
config.mesh_adaptation_error = [0.20 0.20];
config.mesh_maxnbv = 1000000;
config.data_padding = 20000;
config.roi_padding = 50000;
config.minimum_contour_length = 10000;
config.minimum_contour_points = 5;
config.minimum_hole_area = 1e6;
config.closed_contour_tolerance = 750;
config.minimum_grounded_hole_fraction = 0.5;
config.outline_cleanup_radius = 1000;
config.front_probe_distances = [500 1000 2000 4000 8000];
config.grounding_line_tolerance = 5000;

config.initial_temperature = 263.15;
config.rheology_min_temperature = 273;
config.rheology_max_temperature = 240;
config.minimum_ice_thickness = 20;
config.min_speed_for_cost = 1;
config.invert_maxsteps = 40;
config.invert_maxiter = 40;
config.velocity_abs_weight = 1000;
config.grounded_friction_coefficient = 30;
config.solver_residue_threshold = NaN;
config.lcurve_regularization_weights = logspace(-20, -14, 9);
config.initial_shelf_b_scale = 1.10;
config.smoke_regularization_weight = ...
    config.lcurve_regularization_weights(ceil(numel(config.lcurve_regularization_weights) / 2));
config.smoke_invert_maxsteps = 10;
config.smoke_invert_maxiter = 10;

config.mesh_path = fullfile(results_dir, sprintf('%s_Mesh.mat', canonical_name));
config.parameterized_path = fullfile(results_dir, ...
    sprintf('%s_Parameterization.mat', canonical_name));
config.stressbalance_path = fullfile(results_dir, ...
    sprintf('%s_Stressbalance_initial.mat', canonical_name));
config.inversion_path = fullfile(results_dir, ...
    sprintf('%s_Control_B.mat', canonical_name));
config.smoke_inversion_path = fullfile(results_dir, ...
    sprintf('%s_Smoke_Control_B.mat', canonical_name));
config.diagnostics_path = fullfile(results_dir, ...
    sprintf('%s_velocity_misfit_diagnostics.mat', canonical_name));
config.lcurve_path = fullfile(results_dir, ...
    sprintf('%s_lcurve_rheology_B.mat', canonical_name));
config.smoke_lcurve_path = fullfile(results_dir, ...
    sprintf('%s_smoke_rheology_B.mat', canonical_name));
config.lcurve_plot_path = fullfile(results_dir, ...
    sprintf('%s_lcurve_rheology_B.png', canonical_name));
config.speed_plot_path = fullfile(results_dir, ...
    sprintf('%s_velocity_speed_comparison.png', canonical_name));
config.misfit_plot_path = fullfile(results_dir, ...
    sprintf('%s_velocity_misfit_components.png', canonical_name));
end

function canonical_name = canonicalShelfName(shelf_name)
normalized = lower(regexprep(strtrim(shelf_name), '[^a-zA-Z0-9]', ''));
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
        canonical_name = shelf_name;
end
end
