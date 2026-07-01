function [best_md, lcurve] = invert_rheology_b_lcurve_core(md, ...
    shelf_vertices, valid_velocity, options)
%INVERT_RHEOLOGY_B_LCURVE_CORE Core rheology-B L-curve inversion algorithm.
%
% Syntax:
%   [best_md, lcurve] = invert_rheology_b_lcurve_core(md, shelf_vertices, valid_velocity, options);
%
% Required inputs:
%   md - parameterized ISSM model with observations and stress-balance BCs.
%   shelf_vertices - logical vertex mask where rheology_B can be controlled.
%   valid_velocity - logical vertex mask where velocity cost function 101 is
%       active.
%   options - struct with regularization_weights, initial_shelf_b_scale,
%       velocity_abs_weight, np, solver_residue_threshold, maxsteps, maxiter,
%       rheology_min_temperature, and rheology_max_temperature.
%
% Outputs:
%   best_md - model selected at the L-curve corner, with inverted B carried
%       into md.materials.rheology_B.
%   lcurve  - struct array with objective terms, RMSE values, failure state,
%       and selected flag for each alpha.
%
% Saved artifacts:
%   None. This file intentionally contains only the inversion logic so readers
%   can focus on m1qn3 controls, L-curve scoring, and model selection.
%
% Assumptions:
%   The model is already in EPSG:3031 with BedMachine mask codes converted to
%   ISSM levelsets. Cost functions are [101 502]. ISSM solve, m1qn3inversion,
%   generic, verbose, cuffey, and oshostname are available.

% 1. Choose controllable rheology-B entries.
base_B = md.materials.rheology_B;
control_mask = rheologyControlMask(md, shelf_vertices);
if ~any(control_mask)
    error('No rheology_B entries are controlled by the floating-shelf mask.');
end

% 2. Scale the initial shelf B field before each alpha trial.
base_B(control_mask) = base_B(control_mask) * options.initial_shelf_b_scale;

regularization_weights = options.regularization_weights(:);
nweights = numel(regularization_weights);
lcurve = repmat(struct( ...
    'alpha', NaN, ...
    'Jo', NaN, ...
    'alphaR', NaN, ...
        'R', NaN, ...
        'total_J', NaN, ...
        'initial_total_J', NaN, ...
        'final_total_J', NaN, ...
        'loss_decrease', NaN, ...
        'loss_decrease_fraction', NaN, ...
        'vector_rmse', NaN, ...
    'speed_rmse', NaN, ...
    'vx_rmse', NaN, ...
    'vy_rmse', NaN, ...
    'active_vertices', 0, ...
    'failed', false, ...
    'message', '', ...
    'selected', false), nweights, 1);
models = cell(nweights, 1);

for k = 1:nweights
    alpha = regularization_weights(k);
    fprintf('  L-curve alpha %d/%d: %.6g\n', k, nweights, alpha);
    % 3. Build ISSM inversion parameters for one alpha.
    trial_md = setupInversionParameters(md, base_B, shelf_vertices, ...
        valid_velocity, alpha, options);

    lcurve(k).alpha = alpha;
    try
        % 4. Solve stress balance with m1qn3.
        trial_md = solve(trial_md, 'Stressbalance');
        trial_md = carryForwardInvertedB(trial_md);
        % 5. Extract objective terms and velocity diagnostics.
        diagnostics = helpers('summarize_velocity_misfit', ...
            trial_md, valid_velocity);
        [Jo, alphaR, R, total_J] = inversionObjectiveTerms(trial_md, alpha);

        lcurve(k).Jo = Jo;
        lcurve(k).alphaR = alphaR;
        lcurve(k).R = R;
        lcurve(k).total_J = total_J;
        [initial_total_J, final_total_J] = objectiveHistoryEndpoints( ...
            trial_md);
        lcurve(k).initial_total_J = initial_total_J;
        lcurve(k).final_total_J = final_total_J;
        lcurve(k).loss_decrease = initial_total_J - final_total_J;
        lcurve(k).loss_decrease_fraction = ...
            lcurve(k).loss_decrease ./ initial_total_J;
        lcurve(k).vector_rmse = diagnostics.vector_rmse;
        lcurve(k).speed_rmse = diagnostics.speed_rmse;
        lcurve(k).vx_rmse = diagnostics.vx_rmse;
        lcurve(k).vy_rmse = diagnostics.vy_rmse;
        lcurve(k).active_vertices = diagnostics.active_vertices;
        models{k} = trial_md;
        fprintf(['    Jo %.6g, R %.6g, total J %.6g, ', ...
            'speed RMSE %.6g m/yr\n'], Jo, R, total_J, ...
            diagnostics.speed_rmse);
    catch solve_error
        lcurve(k).failed = true;
        lcurve(k).message = solve_error.message;
        fprintf('    failed: %s\n', solve_error.message);
    end
end

finite = find(~[lcurve.failed]' & isfinite([lcurve.Jo]') & ...
    isfinite([lcurve.R]') & [lcurve.Jo]' > 0 & [lcurve.R]' > 0);
if isempty(finite)
    error('All L-curve rheology_B inversions failed.');
end
% 6. Select the L-curve corner, falling back to speed RMSE for short grids.
if numel(finite) >= 3
    selected_finite_index = chooseLcurveCorner( ...
        log10([lcurve(finite).Jo]'), log10([lcurve(finite).R]'));
    selected_index = finite(selected_finite_index);
else
    [~, local_index] = min([lcurve(finite).speed_rmse]);
    selected_index = finite(local_index);
end

lcurve(selected_index).selected = true;
best_md = models{selected_index};
fprintf(['Selected L-curve alpha %.6g: Jo %.6g, R %.6g, ', ...
    'speed RMSE %.6g m/yr.\n'], lcurve(selected_index).alpha, ...
    lcurve(selected_index).Jo, lcurve(selected_index).R, ...
    lcurve(selected_index).speed_rmse);
end

function md = setupInversionParameters(md, base_B, shelf_vertices, ...
    valid_velocity, alpha, options)
% This block intentionally mirrors the "Inversion parameters" section in
% aashray_amery.m, with the generic shelf masks controlling where each cost is
% active.
md.materials.rheology_B = base_B;
md.inversion = m1qn3inversion(md.inversion);
md.inversion.iscontrol = 1;
md.inversion.incomplete_adjoint = 1;
md.inversion.control_parameters = {'MaterialsRheologyBbar'};
md.inversion.control_scaling_factors = 1e8;
md.inversion.maxsteps = options.maxsteps;
md.inversion.maxiter = options.maxiter;
md.inversion.dxmin = 0.1;
md.inversion.dfmin_frac = 0;
md.inversion.gttol = 1e-6;
md.inversion.cost_functions = [101 502];
md.inversion.cost_functions_coefficients = ...
    zeros(md.mesh.numberofvertices, numel(md.inversion.cost_functions));
md.inversion.cost_functions_coefficients(valid_velocity, 1) = ...
    options.velocity_abs_weight;
md.inversion.cost_functions_coefficients(shelf_vertices, 2) = alpha;

min_B = md.materials.rheology_B;
max_B = md.materials.rheology_B;
min_B(shelf_vertices) = cuffey(options.rheology_min_temperature);
max_B(shelf_vertices) = cuffey(options.rheology_max_temperature);
md.inversion.min_parameters = min_B;
md.inversion.max_parameters = max_B;

md.cluster = generic('name', oshostname, 'np', options.np);
md.verbose = verbose('solution', false, 'control', true);
md.settings.solver_residue_threshold = options.solver_residue_threshold;
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

function selected_index = chooseLcurveCorner(log_Jo, log_R)
x = normalizeVector(log_Jo);
y = normalizeVector(log_R);
curvature = -Inf(size(x));

for k = 2:(numel(x) - 1)
    a = [x(k - 1), y(k - 1)];
    b = [x(k), y(k)];
    c = [x(k + 1), y(k + 1)];
    ab = norm(b - a);
    bc = norm(c - b);
    ac = norm(c - a);
    area2 = abs(det([b - a; c - a]));
    if ab > 0 && bc > 0 && ac > 0
        curvature(k) = 2 * area2 / (ab * bc * ac);
    end
end

[~, selected_index] = max(curvature);
if ~isfinite(curvature(selected_index))
    selected_index = ceil(numel(x) / 2);
end
end

function values = normalizeVector(values)
span = max(values) - min(values);
if span == 0 || ~isfinite(span)
    values = zeros(size(values));
else
    values = (values - min(values)) ./ span;
end
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
