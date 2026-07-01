function outputs = create_roi_outline_from_mat(config)
%CREATE_ROI_OUTLINE_FROM_MAT Create a shelf ROI ARGUS outline from x_gd/y_gd.
%
% Syntax:
%   config = shelf_config('Ross');
%   outputs = create_roi_outline_from_mat(config);
%
% Required input:
%   config - struct from shelf_config with input_mat_file, roi_outline_file,
%       shelf_name, and geometry_dir fields.
%
% Outputs:
%   outputs - struct with outline_file, contours, and component_count fields.
%
% Saved artifacts:
%   Geometry/BM2_<Shelf>_Outline.exp is written in config.geometry_dir.
%
% Assumptions:
%   The source .mat file contains top-level x_gd and y_gd arrays. This BM2
%   outline is retained as a Geometry reference artifact only; the working
%   BAMG outline is built directly from BM4 in EPSG:3031.
%
% Examples:
%   config = shelf_config('Ross');
%   create_roi_outline_from_mat(config);

required = {'input_mat_file', 'roi_outline_file', 'shelf_name', 'geometry_dir'};
assertConfigFields(config, required);
assert(isfile(config.input_mat_file), 'Missing input .mat file: %s', ...
    config.input_mat_file);
helpers('ensure_directory', config.geometry_dir);

data = load(config.input_mat_file, 'x_gd', 'y_gd');
if ~isfield(data, 'x_gd') || ~isfield(data, 'y_gd')
    error('Input file must contain top-level x_gd and y_gd variables: %s', ...
        config.input_mat_file);
end

points = [data.x_gd(:), data.y_gd(:)];
points = points(all(isfinite(points), 2), :);
if isempty(points)
    error('No finite x_gd/y_gd points found in %s.', config.input_mat_file);
end

contours = roiContoursFromPoints(points, config.shelf_name);
writeExpContours(contours, config.roi_outline_file);

fprintf('Wrote %d ROI contour(s) from %s:\n', numel(contours), ...
    config.input_mat_file);
fprintf('  %s\n', config.roi_outline_file);

outputs = struct();
outputs.outline_file = config.roi_outline_file;
outputs.contours = contours;
outputs.component_count = numel(contours);
end

function assertConfigFields(config, fields)
for k = 1:numel(fields)
    if ~isfield(config, fields{k})
        error('config.%s is required.', fields{k});
    end
end
end

function contours = roiContoursFromPoints(points, shelf_name)
spacing = gridSpacing(points);
min_x = min(points(:, 1));
min_y = min(points(:, 2));
cols = round((points(:, 1) - min_x) / spacing) + 2;
rows = round((points(:, 2) - min_y) / spacing) + 2;

mask = false(max(rows) + 1, max(cols) + 1);
mask(sub2ind(size(mask), rows, cols)) = true;
boundaries = bwboundaries(mask, 8, 'noholes');
if isempty(boundaries)
    error('bwboundaries did not find any ROI components.');
end

areas = zeros(numel(boundaries), 1);
raw_contours = cell(numel(boundaries), 1);
for k = 1:numel(boundaries)
    boundary = boundaries{k};
    contour = [
        min_x + (boundary(:, 2) - 2) * spacing, ...
        min_y + (boundary(:, 1) - 2) * spacing
    ];
    [~, keep] = unique(contour, 'rows', 'stable');
    contour = contour(sort(keep), :);
    contour(end + 1, :) = contour(1, :);
    raw_contours{k} = contour;
    areas(k) = polyarea(contour(:, 1), contour(:, 2));
end

[~, order] = sort(areas, 'descend');
contours = struct('name', {}, 'nods', {}, 'density', {}, ...
    'x', {}, 'y', {}, 'closed', {});
for output_id = 1:numel(order)
    contour = raw_contours{order(output_id)};
    contour = orientForBamg(contour, output_id > 1);
    contours(output_id).name = sprintf('%s_Domain_%d', shelf_name, output_id); %#ok<AGROW>
    contours(output_id).nods = size(contour, 1);
    contours(output_id).density = 1;
    contours(output_id).x = contour(:, 1);
    contours(output_id).y = contour(:, 2);
    contours(output_id).closed = true;
end
end

function writeExpContours(contours, output_file)
fid = fopen(output_file, 'w');
if fid < 0
    error('Could not open output file for writing: %s', output_file);
end
cleanup = onCleanup(@() fclose(fid));

for component = 1:numel(contours)
    fprintf(fid, '## Name:%s\n', contours(component).name);
    fprintf(fid, '## Icon:0\n');
    fprintf(fid, '# Points Count  Value\n');
    fprintf(fid, '%d 1\n', contours(component).nods);
    fprintf(fid, '# X pos Y pos\n');
    for k = 1:contours(component).nods
        fprintf(fid, '%.12g %.12g\n', ...
            contours(component).x(k), contours(component).y(k));
    end
end
end

function contour = orientForBamg(contour, is_hole)
open_contour = contour(1:end-1, :);
next_contour = contour(2:end, :);
orientation = sum((next_contour(:, 1) - open_contour(:, 1)) .* ...
                  (next_contour(:, 2) + open_contour(:, 2)));

if (~is_hole && orientation > 0) || (is_hole && orientation < 0)
    contour = flipud(contour);
end
end

function spacing = gridSpacing(points)
dx = diff(unique(points(:, 1)));
dy = diff(unique(points(:, 2)));
spacing = min([dx(dx > 0); dy(dy > 0)]);
if isempty(spacing) || ~isfinite(spacing)
    error('Could not infer regular grid spacing from x_gd/y_gd points.');
end
end
