function [md, inversion_result] = run_rheology_single_inversion( ...
    config, regularization_weight, maxsteps, maxiter)
%RUN_RHEOLOGY_SINGLE_INVERSION Run one rheology-B adjoint inversion.
%
% Syntax:
%   [md, result] = run_rheology_single_inversion(config, 1e-17, 10, 10);
%
% This function runs one MaterialsRheologyBbar inversion with one
% regularization weight and returns the solved model plus objective and
% velocity diagnostics.
%
% Assumptions:
%   The parameterized model already exists at config.parameterized_path. The
%   active velocity-cost mask follows rheology_b_inversion_setup.

if nargin < 2 || isempty(regularization_weight)
    regularization_weight = config.regularization_weight;
end
if nargin < 3 || isempty(maxsteps)
    maxsteps = config.smoke_invert_maxsteps;
end
if nargin < 4 || isempty(maxiter)
    maxiter = maxsteps;
end
if ~isscalar(regularization_weight) || ~isfinite(regularization_weight) || ...
        regularization_weight <= 0
    error('regularization_weight must be a positive finite scalar.');
end

helpers('bootstrap_issm_path', config.issm_dir);
md = loadmodel(config.parameterized_path);
setup = rheology_b_inversion_setup(config, md, 'diagnostics');

fprintf('Single rheology-B inversion on %s\n', config.shelf_name);
fprintf('  regularization weight: %.6g\n', regularization_weight);
fprintf('  floating-shelf control vertices: %d of %d\n', ...
    setup.control_vertices, md.mesh.numberofvertices);
fprintf('  active velocity-cost vertices: %d\n', ...
    setup.active_velocity_vertices);

base_B = md.materials.rheology_B;
control_mask = rheologyControlMask(md, setup.shelf_vertices);
if ~any(control_mask)
    error('No rheology_B entries are controlled by the floating-shelf mask.');
end
base_B(control_mask) = base_B(control_mask) * config.initial_shelf_b_scale;

md = setupInversionParameters(config, md, base_B, setup.shelf_vertices, ...
    setup.valid_velocity, regularization_weight, maxsteps, maxiter);
md = solve(md, 'Stressbalance');
md = carryForwardInvertedB(md);

diagnostics = helpers('summarize_velocity_misfit', md, setup.valid_velocity);
[Jo, alphaR, R, total_J] = inversionObjectiveTerms(md, ...
    regularization_weight);
[initial_total_J, final_total_J] = objectiveHistoryEndpoints(md);

inversion_result = struct();
inversion_result.shelf_name = config.shelf_name;
inversion_result.regularization_weight = regularization_weight;
inversion_result.Jo = Jo;
inversion_result.alphaR = alphaR;
inversion_result.R = R;
inversion_result.total_J = total_J;
inversion_result.initial_total_J = initial_total_J;
inversion_result.final_total_J = final_total_J;
inversion_result.loss_decrease = initial_total_J - final_total_J;
inversion_result.loss_decrease_fraction = ...
    inversion_result.loss_decrease ./ initial_total_J;
inversion_result.velocity_diagnostics = diagnostics;
inversion_result.active_velocity_vertices = setup.active_velocity_vertices;

fprintf('Single inversion result for %s:\n', config.shelf_name);
fprintf('  initial J %.6g, final J %.6g, decrease %.6g\n', ...
    initial_total_J, final_total_J, inversion_result.loss_decrease);
fprintf('  speed RMSE %.6g m/yr over %d active vertices\n', ...
    diagnostics.speed_rmse, diagnostics.active_vertices);
end

function md = setupInversionParameters(config, md, base_B, shelf_vertices, ...
    valid_velocity, alpha, maxsteps, maxiter)
md.materials.rheology_B = base_B;
md.inversion = m1qn3inversion(md.inversion);
md.inversion.iscontrol = 1;
md.inversion.incomplete_adjoint = 1;
md.inversion.control_parameters = {'MaterialsRheologyBbar'};
md.inversion.control_scaling_factors = 1e8;
md.inversion.maxsteps = maxsteps;
md.inversion.maxiter = maxiter;
md.inversion.dxmin = 0.1;
md.inversion.dfmin_frac = 0;
md.inversion.gttol = 1e-6;
md.inversion.cost_functions = [101 502];
md.inversion.cost_functions_coefficients = ...
    zeros(md.mesh.numberofvertices, numel(md.inversion.cost_functions));
md.inversion.cost_functions_coefficients(valid_velocity, 1) = ...
    config.velocity_abs_weight;
md.inversion.cost_functions_coefficients(shelf_vertices, 2) = alpha;

min_B = md.materials.rheology_B;
max_B = md.materials.rheology_B;
min_B(shelf_vertices) = cuffey(config.rheology_min_temperature);
max_B(shelf_vertices) = cuffey(config.rheology_max_temperature);
md.inversion.min_parameters = min_B;
md.inversion.max_parameters = max_B;

md.cluster = generic('name', oshostname, 'np', config.np);
md.verbose = verbose('solution', false, 'control', true);
md.settings.solver_residue_threshold = config.solver_residue_threshold;
end

function md = carryForwardInvertedB(md)
if isfield(md.results.StressbalanceSolution, 'MaterialsRheologyBbar')
    md.materials.rheology_B = ...
        md.results.StressbalanceSolution.MaterialsRheologyBbar;
end
end

function [Jo, alphaR, R, total_J] = inversionObjectiveTerms(md, alpha)
J = md.results.StressbalanceSolution.J;
if isempty(J)
    Jo = NaN;
    alphaR = NaN;
    R = NaN;
    total_J = NaN;
    return;
end

final_J = J(end, :);
Jo = final_J(1);
if numel(final_J) >= 3
    alphaR = final_J(end - 1);
    total_J = final_J(end);
elseif numel(final_J) == 2
    alphaR = final_J(2);
    total_J = sum(final_J);
else
    alphaR = NaN;
    total_J = final_J(1);
end
R = alphaR ./ alpha;
end

function [initial_total_J, final_total_J] = objectiveHistoryEndpoints(md)
J = md.results.StressbalanceSolution.J;
if isempty(J)
    initial_total_J = NaN;
    final_total_J = NaN;
    return;
end
if size(J, 2) >= 3
    total_history = J(:, end);
else
    total_history = sum(J, 2);
end
initial_total_J = total_history(1);
final_total_J = total_history(end);
end

function control_mask = rheologyControlMask(md, shelf_vertices)
if numel(md.materials.rheology_B) == md.mesh.numberofvertices
    control_mask = shelf_vertices;
elseif numel(md.materials.rheology_B) == md.mesh.numberofelements
    control_mask = all(shelf_vertices(md.mesh.elements), 2);
else
    error(['Unexpected rheology_B size: %d entries for %d vertices ', ...
        'and %d elements.'], numel(md.materials.rheology_B), ...
        md.mesh.numberofvertices, md.mesh.numberofelements);
end
end
