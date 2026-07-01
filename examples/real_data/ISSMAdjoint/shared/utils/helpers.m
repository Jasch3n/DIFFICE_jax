function varargout = helpers(action, varargin)
%HELPERS Shared low-level utilities for the ISSM adjoint shelf pipeline.
%
% Syntax:
%   helpers('bootstrap_issm_path', issm_dir);
%   [x, y, data] = helpers('read_grid_subset', ncfile, variable, bounds);
%
% This file intentionally collects small utilities so the shared directory is
% reserved for functions that express the main shelf workflow and inversion
% logic.

switch action
    case 'bootstrap_issm_path'
        bootstrap_issm_path(varargin{:});
        varargout = {};
    case 'ensure_directory'
        ensure_directory(varargin{:});
        varargout = {};
    case 'read_grid_subset'
        [varargout{1:nargout}] = read_grid_subset(varargin{:});
    case 'mesh_bounds'
        varargout{1} = mesh_bounds(varargin{:});
    case 'contour_bounds'
        varargout{1} = contour_bounds(varargin{:});
    case 'standardize_contours'
        varargout{1} = standardize_contours(varargin{:});
    case 'apply_bedmachine_mask'
        varargout{1} = apply_bedmachine_mask(varargin{:});
    case 'fill_invalid'
        varargout{1} = fill_invalid(varargin{:});
    case 'replace_nan_with_median'
        varargout{1} = replace_nan_with_median(varargin{:});
    case 'summarize_velocity_misfit'
        varargout{1} = summarize_velocity_misfit(varargin{:});
    case 'plot_velocity_diagnostics'
        plot_velocity_diagnostics(varargin{:});
        varargout = {};
    case 'plot_lcurve'
        plot_lcurve(varargin{:});
        varargout = {};
    otherwise
        error('Unknown helpers action: %s', action);
end
end

function bootstrap_issm_path(issm_dir)
if nargin < 1 || isempty(issm_dir)
    error('bootstrap_issm_path requires issm_dir.');
end
if (exist('model', 'file') == 2 || exist('model', 'class') == 8) && ...
        exist('expread', 'file') == 2 && exist('expwrite', 'file') == 2
    return;
end
addpath(fullfile(issm_dir, 'bin'));
addpath(genpath(fullfile(issm_dir, 'src', 'm')));
end

function ensure_directory(directory_path)
if ~exist(directory_path, 'dir')
    mkdir(directory_path);
end
end

function [x_out, y_out, data_out] = read_grid_subset(ncfile, variable, bounds)
x = double(ncread(ncfile, 'x'));
y = double(ncread(ncfile, 'y'));
ix = find(x >= bounds(1) & x <= bounds(2));
iy = find(y >= bounds(3) & y <= bounds(4));
if isempty(ix) || isempty(iy)
    error('%s has no overlap with requested bounds.', variable);
end
start = [ix(1), iy(1)];
count = [numel(ix), numel(iy)];
data = double(ncread(ncfile, variable, start, count));
data = applyFillValue(ncfile, variable, data);
x_out = x(ix);
y_out = y(iy);
data_out = data';
[x_out, x_order] = sort(x_out);
data_out = data_out(:, x_order);
[y_out, y_order] = sort(y_out);
data_out = data_out(y_order, :);
end

function data = applyFillValue(ncfile, variable, data)
info = ncinfo(ncfile, variable);
for k = 1:numel(info.Attributes)
    if strcmp(info.Attributes(k).Name, '_FillValue')
        fill_value = double(info.Attributes(k).Value);
        data(data == fill_value) = NaN;
        return;
    end
end
end

function bounds = mesh_bounds(md, padding)
bounds = [
    min(md.mesh.x) - padding, max(md.mesh.x) + padding, ...
    min(md.mesh.y) - padding, max(md.mesh.y) + padding
];
end

function bounds = contour_bounds(contours, padding)
all_x = [];
all_y = [];
for k = 1:numel(contours)
    all_x = [all_x; contours(k).x(:)]; %#ok<AGROW>
    all_y = [all_y; contours(k).y(:)]; %#ok<AGROW>
end
bounds = [
    min(all_x) - padding, max(all_x) + padding, ...
    min(all_y) - padding, max(all_y) + padding
];
end

function contours = standardize_contours(contours)
template = struct('name', {}, 'nods', {}, 'density', {}, ...
    'x', {}, 'y', {}, 'closed', {});
standardized = template;
for k = 1:numel(contours)
    contour = contours(k);
    if ~isfield(contour, 'name') || isempty(contour.name)
        contour.name = sprintf('Contour_%02d', k);
    end
    if ~isfield(contour, 'density') || isempty(contour.density)
        contour.density = 1;
    end
    contour.x = contour.x(:);
    contour.y = contour.y(:);
    contour.nods = numel(contour.x);
    contour.closed = contour.x(1) == contour.x(end) && ...
        contour.y(1) == contour.y(end);
    standardized(k).name = contour.name; %#ok<AGROW>
    standardized(k).nods = contour.nods;
    standardized(k).density = contour.density;
    standardized(k).x = contour.x;
    standardized(k).y = contour.y;
    standardized(k).closed = contour.closed;
end
contours = standardized;
end

function md = apply_bedmachine_mask(md, bedmachine_mask)
local_mask = round(bedmachine_mask);
floating_shelf = local_mask == 3;
no_ice = local_mask == 0 | local_mask == 1;
md.mask.ocean_levelset = ones(md.mesh.numberofvertices, 1);
md.mask.ocean_levelset(floating_shelf) = -1;
md.mask.ice_levelset = -ones(md.mesh.numberofvertices, 1);
md.mask.ice_levelset(no_ice) = 1;
if ~any(md.mask.ocean_levelset < 0)
    error('BedMachine mask did not identify any floating shelf vertices.');
end
end

function values = fill_invalid(values, fill_value)
values(~isfinite(values)) = fill_value;
end

function values = replace_nan_with_median(values)
finite_values = values(isfinite(values));
if isempty(finite_values)
    values(:) = 0;
    return;
end
values(~isfinite(values)) = median(finite_values);
end

function diagnostics = summarize_velocity_misfit(md, valid_velocity)
sol = md.results.StressbalanceSolution;
active = valid_velocity & isfinite(sol.Vx) & isfinite(sol.Vy) & ...
    isfinite(md.inversion.vx_obs) & isfinite(md.inversion.vy_obs);
if ~any(active)
    error('No active finite vertices available for velocity RMSE.');
end
vx_misfit = sol.Vx - md.inversion.vx_obs;
vy_misfit = sol.Vy - md.inversion.vy_obs;
observed_speed = sqrt(md.inversion.vx_obs.^2 + md.inversion.vy_obs.^2);
modeled_speed = sqrt(sol.Vx.^2 + sol.Vy.^2);
speed_misfit = modeled_speed - observed_speed;
diagnostics = struct();
diagnostics.active_vertices = nnz(active);
diagnostics.vector_rmse = sqrt(mean(vx_misfit(active).^2 + ...
    vy_misfit(active).^2));
diagnostics.speed_rmse = sqrt(mean(speed_misfit(active).^2));
diagnostics.vx_rmse = sqrt(mean(vx_misfit(active).^2));
diagnostics.vy_rmse = sqrt(mean(vy_misfit(active).^2));
diagnostics.mean_speed_misfit = mean(speed_misfit(active));
diagnostics.mean_observed_speed = mean(observed_speed(active));
diagnostics.mean_modeled_speed = mean(modeled_speed(active));
diagnostics.max_abs_speed_misfit = max(abs(speed_misfit(active)));
end

function plot_velocity_diagnostics(config, md, valid_velocity, diagnostics)
sol = md.results.StressbalanceSolution;
metric_active = valid_velocity & isfinite(sol.Vx) & isfinite(sol.Vy) & ...
    isfinite(md.inversion.vx_obs) & isfinite(md.inversion.vy_obs);
observed_speed = sqrt(md.inversion.vx_obs.^2 + md.inversion.vy_obs.^2);
modeled_speed = sqrt(sol.Vx.^2 + sol.Vy.^2);
speed_misfit = modeled_speed - observed_speed;
vx_misfit = sol.Vx - md.inversion.vx_obs;
vy_misfit = sol.Vy - md.inversion.vy_obs;
relative_abs_error = abs(speed_misfit) ./ max(observed_speed, 1);
plot_active = isfinite(sol.Vx) & isfinite(sol.Vy) & ...
    isfinite(md.inversion.vx_obs) & isfinite(md.inversion.vy_obs);
observed_speed_plot = maskInactive(observed_speed, plot_active);
modeled_speed_plot = maskInactive(modeled_speed, plot_active);
relative_abs_error_plot = maskInactive(relative_abs_error, plot_active);
relative_abs_error_stats = relativeErrorStats(relative_abs_error, metric_active);
speed_misfit_plot = maskInactive(speed_misfit, plot_active);
vx_misfit_plot = maskInactive(vx_misfit, plot_active);
vy_misfit_plot = maskInactive(vy_misfit, plot_active);
speed_max = max([observed_speed(plot_active); modeled_speed(plot_active)]);
if ~isfinite(speed_max) || speed_max <= 0
    speed_max = 1;
end
residual_max = max(abs([speed_misfit(metric_active); ...
    vx_misfit(metric_active); vy_misfit(metric_active)]));
if ~isfinite(residual_max) || residual_max <= 0
    residual_max = 1;
end
layout_direction = diagnosticPlotLayout(config.shelf_name);
plotStackedMeshPanels(md, ...
    {observed_speed_plot, modeled_speed_plot, relative_abs_error_plot}, ...
    {'Observed speed', sprintf('Modeled speed (RMSE %.3g)', ...
        diagnostics.vector_rmse), 'Relative abs. error'}, ...
    {[0 speed_max], [0 speed_max], [0 1]}, ...
    {'m/yr', 'm/yr', 'relative'}, config.speed_plot_path, ...
    {'', '', relative_abs_error_stats}, layout_direction);
plotStackedMeshPanels(md, ...
    {speed_misfit_plot, vx_misfit_plot, vy_misfit_plot}, ...
    {sprintf('Speed misfit (%.3g)', diagnostics.speed_rmse), ...
     sprintf('Vx misfit (%.3g)', diagnostics.vx_rmse), ...
     sprintf('Vy misfit (%.3g)', diagnostics.vy_rmse)}, ...
    {[-residual_max residual_max], [-residual_max residual_max], ...
     [-residual_max residual_max]}, ...
    {'m/yr', 'm/yr', 'm/yr'}, config.misfit_plot_path, {}, ...
    layout_direction);
end

function values = maskInactive(values, active)
values(~active) = NaN;
end

function layout_direction = diagnosticPlotLayout(shelf_name)
switch lower(shelf_name)
    case {'larsenc', 'rnflch', 'ross'}
        layout_direction = 'horizontal';
    otherwise
        layout_direction = 'vertical';
end
end

function plotStackedMeshPanels(md, data_list, title_list, caxis_list, ...
    colorbar_labels, output_file, annotations, layout_direction)
if nargin < 7
    annotations = {};
end
if nargin < 8 || isempty(layout_direction)
    layout_direction = 'vertical';
end
panel_count = numel(data_list);
[figure_width, figure_height] = figureSizeForMesh(md, panel_count, ...
    layout_direction);
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 figure_width figure_height]);
if strcmp(layout_direction, 'horizontal')
    layout = tiledlayout(fig, 1, panel_count, 'TileSpacing', 'compact', ...
        'Padding', 'compact');
else
    layout = tiledlayout(fig, panel_count, 1, 'TileSpacing', 'compact', ...
        'Padding', 'compact');
end

for k = 1:panel_count
    ax = nexttile(layout);
    if iscell(colorbar_labels)
        colorbar_label = colorbar_labels{k};
    else
        colorbar_label = colorbar_labels;
    end
    annotation = '';
    if numel(annotations) >= k && ~isempty(annotations{k})
        annotation = annotations{k};
    end
    plot_fem_field(md, data_list{k}, 'Parent', ax, ...
        'Title', title_list{k}, 'Caxis', caxis_list{k}, ...
        'ColorbarLabel', colorbar_label, 'Annotation', annotation);
end

exportgraphics(fig, output_file, 'Resolution', 200);
close(fig);
end

function [figure_width, figure_height] = figureSizeForMesh(md, panel_count, ...
    layout_direction)
x_range = max(md.mesh.x) - min(md.mesh.x);
y_range = max(md.mesh.y) - min(md.mesh.y);
aspect = x_range / max(y_range, eps);
single_width = round(min(1600, max(850, 900 * aspect)));
single_height = round(min(1300, max(650, single_width / max(aspect, eps))));
if strcmp(layout_direction, 'horizontal')
    figure_width = min(4800, panel_count * single_width + 180);
    figure_height = single_height + 120;
else
    figure_width = single_width;
    figure_height = min(4200, panel_count * single_height + 120);
end
end

function stats_text = relativeErrorStats(relative_abs_error, active)
values = relative_abs_error(active & isfinite(relative_abs_error));
if isempty(values)
    stats_text = sprintf('RAE stats\nmin    n/a\nmax    n/a\nmean   n/a\nmedian n/a');
    return;
end
stats_text = sprintf(['RAE stats\n', ...
    'min    %7.3f\n', ...
    'max    %7.3f\n', ...
    'mean   %7.3f\n', ...
    'median %7.3f'], ...
    min(values), max(values), mean(values), median(values));
end

function plot_lcurve(config, lcurve)
FONT_SIZE = 28;
valid = ~[lcurve.failed]' & isfinite([lcurve.Jo]') & ...
    isfinite([lcurve.R]') & [lcurve.Jo]' > 0 & [lcurve.R]' > 0;
if ~any(valid)
    return;
end
figure('Visible', 'off');
loglog([lcurve(valid).Jo], [lcurve(valid).R], '-o', ...
    'Color', [0.1 0.35 0.75], 'MarkerFaceColor', [0.1 0.35 0.75]);
hold on;
selected = [lcurve.selected]';
if any(selected)
    loglog([lcurve(selected).Jo], [lcurve(selected).R], 'rp', ...
        'MarkerSize', 14, 'MarkerFaceColor', 'r');
end
set(gca, 'FontSize', FONT_SIZE);
xlabel('absolute velocity misfit J_o', 'FontSize', FONT_SIZE);
ylabel('rheology_B regularization R', 'FontSize', FONT_SIZE);
title('Rheology B L-curve', 'FontWeight', 'bold', 'FontSize', FONT_SIZE);
grid on;
axis square;
exportgraphics(gcf, config.lcurve_plot_path, 'Resolution', 200);
close(gcf);
end
