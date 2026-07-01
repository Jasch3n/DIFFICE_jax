function outputs = build_bedmachine_outline(config)
%BUILD_BEDMACHINE_OUTLINE Build the BAMG outline directly from BM4 mask.
%
% Syntax:
%   config = shelf_config('Ross');
%   outputs = build_bedmachine_outline(config);
%
% Required input:
%   config - struct from shelf_config with bedmachine_file, bedmachine_bounds,
%       geometry_dir, mesh_domain_file, and gl_preview_file.
%
% Outputs:
%   outputs - struct containing outline_file, preview_file, the selected
%       floating-domain contour, grounded-island holes, and GL contours.
%
% Saved artifacts:
%   Geometry/<Shelf>_Outline.exp and Geometry/<Shelf>_GL_preview.png.
%
% Assumptions:
%   The working outline is built directly in EPSG:3031 from BedMachine v4
%   mask transitions. BedMachine mask codes are 0 ocean, 1 ice-free land,
%   2 grounded ice, 3 floating ice, and 4 Lake Vostok. Cyan preview segments
%   are floating-to-ocean only; red segments are floating-to-solid, where
%   solid means mask 1, 2, or 4. BM2 outlines are not used for transforms.
%
% Examples:
%   config = shelf_config('Ross');
%   build_bedmachine_outline(config);

required = {'shelf_name', 'mesh_domain_file', 'gl_preview_file', ...
    'geometry_dir', 'bedmachine_file', 'bedmachine_bounds', 'issm_dir'};
assertConfigFields(config, required);
assert(isfile(config.bedmachine_file), 'Missing BedMachine file: %s', ...
    config.bedmachine_file);
helpers('ensure_directory', config.geometry_dir);
helpers('bootstrap_issm_path', config.issm_dir);

fprintf('Reading BedMachine v4 mask around %s shelf bounds...\n', ...
    config.shelf_name);
[bed_x, bed_y, bedmachine_mask] = helpers('read_grid_subset', ...
    config.bedmachine_file, 'mask', config.bedmachine_bounds);

floating_mask = round(bedmachine_mask) == 3;
floating_mask = applyShelfClip(floating_mask, bed_x, bed_y, config);
floating_mask = cleanFloatingMask(floating_mask, bed_x, bed_y, config);
floating_field = double(floating_mask);
floating_contours = contourMatrixToStruct( ...
    contourc(bed_x, bed_y, floating_field, [0.5 0.5]), ...
    sprintf('%s_Floating', config.shelf_name));
[domain_epsg, hole_epsg] = selectFloatingDomainAndHoles( ...
    floating_contours, bed_x, bed_y, bedmachine_mask, config);

if isempty(domain_epsg)
    error('No BedMachine-v4 floating ice contour found for %s.', ...
        config.shelf_name);
end

solid_field = double(isSolidMask(bedmachine_mask) & ...
    clipMask(bed_x, bed_y, config));
solid_contours = contourMatrixToStruct( ...
    contourc(bed_x, bed_y, solid_field, [0.5 0.5]), ...
    sprintf('%s_Solid', config.shelf_name));
hole_epsg = selectSolidIslandHoles(solid_contours, domain_epsg, config);

gl_field = NaN(size(bedmachine_mask));
clip_mask = clipMask(bed_x, bed_y, config);
gl_field(round(bedmachine_mask) == 3 & clip_mask) = 0;
gl_field(isSolidMask(bedmachine_mask)) = 1;
gl_raw = contourMatrixToStruct( ...
    contourc(bed_x, bed_y, gl_field, [0.5 0.5]), ...
    sprintf('%s_GL', config.shelf_name));
gl_epsg = selectContoursInsideDomain(gl_raw, domain_epsg, ...
    config.minimum_contour_length, config.minimum_contour_points, ...
    sprintf('%s_GL', config.shelf_name));

domain_epsg = helpers('standardize_contours', domain_epsg);
hole_epsg = helpers('standardize_contours', hole_epsg);
gl_epsg = helpers('standardize_contours', gl_epsg);
outline_epsg = [domain_epsg hole_epsg];
expwrite(outline_epsg, config.mesh_domain_file);

fprintf('Wrote direct BedMachine-v4 %s floating-domain outline:\n', ...
    config.shelf_name);
fprintf('  %s\n', config.mesh_domain_file);
fprintf('Wrote %d BedMachine-v4 solid-island hole contour(s).\n', ...
    numel(hole_epsg));
fprintf('Found %d BedMachine-v4 floating/solid boundary contour(s).\n', ...
    numel(gl_epsg));

try
    plotPreview(bed_x, bed_y, bedmachine_mask, domain_epsg, gl_epsg, ...
        hole_epsg, config.gl_preview_file, config.shelf_name);
    fprintf('  %s\n', config.gl_preview_file);
catch preview_error
    warning('Could not create preview plot: %s', preview_error.message);
end

outputs = struct();
outputs.outline_file = config.mesh_domain_file;
outputs.preview_file = config.gl_preview_file;
outputs.domain_epsg3031 = domain_epsg;
outputs.gl_epsg3031 = gl_epsg;
outputs.holes_epsg3031 = hole_epsg;
outputs.outline_epsg3031 = outline_epsg;
end

function assertConfigFields(config, fields)
for k = 1:numel(fields)
    if ~isfield(config, fields{k})
        error('config.%s is required.', fields{k});
    end
end
end

function contours = contourMatrixToStruct(c, name_prefix)
contours = struct('name', {}, 'nods', {}, 'density', {}, ...
    'x', {}, 'y', {}, 'closed', {});
cursor = 1;
contour_id = 0;
while cursor < size(c, 2)
    npoints = c(2, cursor);
    first = cursor + 1;
    last = cursor + npoints;
    contour_id = contour_id + 1;
    contours(contour_id).name = sprintf('%s_raw_%02d', ...
        name_prefix, contour_id); %#ok<AGROW>
    contours(contour_id).nods = npoints;
    contours(contour_id).density = 1;
    contours(contour_id).x = c(1, first:last)';
    contours(contour_id).y = c(2, first:last)';
    contours(contour_id).closed = isClosedContour( ...
        contours(contour_id).x, contours(contour_id).y, 0);
    cursor = last + 1;
end
end

function [domain, holes] = selectFloatingDomainAndHoles( ...
    contours, bed_x, bed_y, bedmachine_mask, config)
domain = struct('name', {}, 'nods', {}, 'density', {}, ...
    'x', {}, 'y', {}, 'closed', {});
holes = domain;
areas = [];
valid = false(numel(contours), 1);
for k = 1:numel(contours)
    [x, y] = closeContour(contours(k).x(:), contours(k).y(:));
    if numel(x) < config.minimum_contour_points
        continue;
    end
    area = polyarea(x, y);
    if area <= 0
        continue;
    end
    contours(k).x = x;
    contours(k).y = y;
    contours(k).nods = numel(x);
    contours(k).closed = true;
    areas(k) = area; %#ok<AGROW>
    valid(k) = true;
end
if ~any(valid)
    return;
end

valid_indices = find(valid);
[~, largest_local] = max(areas(valid_indices));
domain_index = valid_indices(largest_local);
domain = contours(domain_index);
domain.name = sprintf('%s_Floating_Domain', config.shelf_name);

[grid_x, grid_y] = meshgrid(bed_x, bed_y);
for k = valid_indices(:)'
    if k == domain_index
        continue;
    end
    x = contours(k).x(:);
    y = contours(k).y(:);
    if ~all(inpolygon(x, y, domain.x, domain.y)) || ...
            polyarea(x, y) < config.minimum_hole_area
        continue;
    end
    grid_inside = inpolygon(grid_x, grid_y, x, y);
    interior_mask = round(bedmachine_mask(grid_inside));
    interior_mask = interior_mask(isfinite(interior_mask));
    if isempty(interior_mask)
        continue;
    end
    solid_fraction = mean(interior_mask == 1 | ...
        interior_mask == 2 | interior_mask == 4);
    if solid_fraction >= config.minimum_grounded_hole_fraction
        contours(k).name = sprintf('%s_Hole_%02d', ...
            config.shelf_name, numel(holes) + 1);
        holes(end + 1) = contours(k); %#ok<AGROW>
    end
end
end

function selected = selectContoursInsideDomain(contours, domain, ...
    minimum_length, minimum_points, name_prefix)
selected = struct('name', {}, 'nods', {}, 'density', {}, ...
    'x', {}, 'y', {}, 'closed', {});
keep_lengths = [];
for k = 1:numel(contours)
    x = contours(k).x(:);
    y = contours(k).y(:);
    if numel(x) < minimum_points
        continue;
    end
    contour_length = polylineLength(x, y);
    if contour_length < minimum_length
        continue;
    end
    inside = inpolygon(x, y, domain.x, domain.y);
    if any(inside)
        selected(end + 1) = contours(k); %#ok<AGROW>
        keep_lengths(end + 1) = contour_length; %#ok<AGROW>
    end
end
[~, order] = sort(keep_lengths, 'descend');
selected = selected(order);
for k = 1:numel(selected)
    selected(k).name = sprintf('%s_%02d', name_prefix, k);
    selected(k).density = 1;
end
end

function holes = selectSolidIslandHoles(contours, domain, config)
holes = struct('name', {}, 'nods', {}, 'density', {}, ...
    'x', {}, 'y', {}, 'closed', {});
areas = [];
for k = 1:numel(contours)
    [x, y] = closeContour(contours(k).x(:), contours(k).y(:));
    if numel(x) < config.minimum_contour_points
        continue;
    end
    area = polyarea(x, y);
    if area < config.minimum_hole_area
        continue;
    end
    inside = inpolygon(x, y, domain.x, domain.y);
    if mean(inside) < 0.95
        continue;
    end
    contours(k).x = x;
    contours(k).y = y;
    contours(k).nods = numel(x);
    contours(k).closed = true;
    contours(k).name = sprintf('%s_Hole_%02d', ...
        config.shelf_name, numel(holes) + 1);
    holes(end + 1) = contours(k); %#ok<AGROW>
    areas(end + 1) = area; %#ok<AGROW>
end
[~, order] = sort(areas, 'descend');
holes = holes(order);
for k = 1:numel(holes)
    holes(k).name = sprintf('%s_Hole_%02d', config.shelf_name, k);
end
end

function closed = isClosedContour(x, y, tolerance)
closed = hypot(x(1) - x(end), y(1) - y(end)) <= tolerance;
end

function [x, y] = closeContour(x, y)
if x(1) ~= x(end) || y(1) ~= y(end)
    x(end + 1) = x(1);
    y(end + 1) = y(1);
end
end

function length_m = polylineLength(x, y)
length_m = sum(sqrt(diff(x).^2 + diff(y).^2));
end

function solid = isSolidMask(mask)
mask = round(mask);
solid = mask == 1 | mask == 2 | mask == 4;
end

function field = applyShelfClip(field, x, y, config)
inside = clipMask(x, y, config);
field(~inside) = 0;
end

function mask = cleanFloatingMask(mask, x, y, config)
if ~isfield(config, 'outline_cleanup_radius') || ...
        isempty(config.outline_cleanup_radius) || ...
        config.outline_cleanup_radius <= 0
    return;
end
dx = median(abs(diff(x)));
dy = median(abs(diff(y)));
grid_spacing = min(dx, dy);
radius_cells = max(1, round(config.outline_cleanup_radius / grid_spacing));
se = strel('disk', radius_cells, 0);
mask = imopen(mask, se);
mask = imclose(mask, se);
end

function inside = clipMask(x, y, config)
[grid_x, grid_y] = meshgrid(x, y); %#ok<ASGLU>
inside = true(size(grid_x));
if isfield(config, 'bedmachine_clip') && ~isempty(config.bedmachine_clip)
    clip = config.bedmachine_clip;
    if isfield(clip, 'xmin') && ~isempty(clip.xmin)
        inside = inside & grid_x >= clip.xmin;
    end
    if isfield(clip, 'xmax') && ~isempty(clip.xmax)
        inside = inside & grid_x <= clip.xmax;
    end
    if isfield(clip, 'ymin') && ~isempty(clip.ymin)
        inside = inside & grid_y >= clip.ymin;
    end
    if isfield(clip, 'ymax') && ~isempty(clip.ymax)
        inside = inside & grid_y <= clip.ymax;
    end
end
end

function plotPreview(x, y, bedmachine_mask, domain_epsg, gl_epsg, ...
    hole_epsg, output_file, shelf_name)
figure('Visible', 'off');
imagesc(x / 1e3, y / 1e3, bedmachine_mask);
set(gca, 'YDir', 'normal');
axis equal tight;
hold on;
plotClassifiedDomainBoundary(x, y, bedmachine_mask, domain_epsg);
for k = 1:numel(gl_epsg)
    plot(gl_epsg(k).x / 1e3, gl_epsg(k).y / 1e3, ...
        'r-', 'LineWidth', 1.5);
end
for k = 1:numel(hole_epsg)
    plot(hole_epsg(k).x / 1e3, hole_epsg(k).y / 1e3, ...
        'm-', 'LineWidth', 1.2);
end
title(sprintf('BedMachine v4 %s floating shelf outline and solid boundary', ...
    shelf_name));
xlabel('x_{ps} [km]');
ylabel('y_{ps} [km]');
colorbar;
saveas(gcf, output_file);
close(gcf);
end

function plotClassifiedDomainBoundary(grid_x, grid_y, mask_grid, domain)
for k = 1:(numel(domain.x) - 1)
    x1 = domain.x(k);
    y1 = domain.y(k);
    x2 = domain.x(k + 1);
    y2 = domain.y(k + 1);
    segment_length = hypot(x2 - x1, y2 - y1);
    if segment_length == 0
        continue;
    end

    midpoint_x = 0.5 * (x1 + x2);
    midpoint_y = 0.5 * (y1 + y2);
    normal_x = -(y2 - y1) / segment_length;
    normal_y = (x2 - x1) / segment_length;
    probe_distance = 750;
    mask_a = interp2(grid_x, grid_y, mask_grid, ...
        midpoint_x + probe_distance * normal_x, ...
        midpoint_y + probe_distance * normal_y, 'nearest', NaN);
    mask_b = interp2(grid_x, grid_y, mask_grid, ...
        midpoint_x - probe_distance * normal_x, ...
        midpoint_y - probe_distance * normal_y, 'nearest', NaN);

    adjacent = round([mask_a mask_b]);
    if any(adjacent == 0)
        segment_color = 'c';
    elseif any(isSolidMask(adjacent))
        segment_color = 'r';
    else
        segment_color = [0.2 0.2 0.2];
    end
    plot([x1 x2] / 1e3, [y1 y2] / 1e3, '-', ...
        'Color', segment_color, 'LineWidth', 1.5);
end
end
