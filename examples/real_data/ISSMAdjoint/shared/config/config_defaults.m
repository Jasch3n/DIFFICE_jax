function defaults = config_defaults()
%CONFIG_DEFAULTS Return default ISSM adjoint shelf inversion settings.
%
% These defaults are intentionally shelf-independent. Dataset paths,
% shelf-specific bounds, clips, and output directories live in configs/*.yaml.

defaults = struct();
defaults.name = '';

defaults.runtime = struct();
defaults.runtime.issm_dir = '/Users/jiapchen/Software/ISSM';
defaults.runtime.np = 2;

defaults.data = struct();
defaults.data.bedmachine_file = '';
defaults.data.measures_file = '';
defaults.data.bedmachine_bounds = [];
defaults.data.bedmachine_clip = struct();

defaults.paths = struct();
defaults.paths.shelf_dir = '';
defaults.paths.geometry_dir = '';
defaults.paths.results_dir = '';

defaults.outline = struct();
defaults.outline.minimum_contour_length = 10000;
defaults.outline.minimum_contour_points = 5;
defaults.outline.minimum_hole_area = 1e6;
defaults.outline.closed_contour_tolerance = 750;
defaults.outline.minimum_grounded_hole_fraction = 0.5;
defaults.outline.outline_cleanup_radius = 1000;
defaults.outline.front_probe_distances = [500 1000 2000 4000 8000];
defaults.outline.grounding_line_tolerance = 5000;

defaults.mesh = struct();
defaults.mesh.initial_hmax = 10000;
defaults.mesh.hmax = 10000;
defaults.mesh.hmin = 1000;
defaults.mesh.gradation = 1.5;
defaults.mesh.adaptation_error = [0.20 0.20];
defaults.mesh.maxnbv = 1000000;
defaults.mesh.data_padding = 20000;

defaults.physics = struct();
defaults.physics.initial_temperature = 263.15;
defaults.physics.rheology_min_temperature = 273;
defaults.physics.rheology_max_temperature = 240;
defaults.physics.minimum_ice_thickness = 20;
defaults.physics.grounded_friction_coefficient = 30;

defaults.inversion = struct();
defaults.inversion.min_speed_for_cost = 1;
defaults.inversion.velocity_abs_weight = 1000;
defaults.inversion.regularization_weight = 1e-17;
defaults.inversion.initial_shelf_b_scale = 1.10;
defaults.inversion.maxsteps = 40;
defaults.inversion.maxiter = 40;
defaults.inversion.solver_residue_threshold = NaN;

defaults.smoke = struct();
defaults.smoke.regularization_weight = struct('use_inversion_weight', true);
defaults.smoke.maxsteps = 10;
defaults.smoke.maxiter = 10;
end
