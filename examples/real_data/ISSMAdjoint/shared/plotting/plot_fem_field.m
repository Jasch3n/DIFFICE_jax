function h = plot_fem_field(md, values, varargin)
%PLOT_FEM_FIELD Plot scalar data on an ISSM triangular FEM mesh.
%
% Syntax:
%   plot_fem_field(md, values);
%   plot_fem_field(md, values, 'Parent', ax, 'Caxis', [0 1]);
%   plot_fem_field(md, rae, 'Annotation', stats_text);
%
% Required input:
%   md - ISSM model with md.mesh.x, md.mesh.y, and md.mesh.elements.
%   values - vertex-sized scalar field. Elements touching NaN/Inf vertices
%       are omitted so masked shelf fronts and grounded holes are not drawn
%       as partial plotting triangles.
%
% Outputs:
%   h - handle to the trisurf object.
%
% Assumptions:
%   Coordinates are EPSG:3031 meters and are displayed in kilometers as
%   x_ps and y_ps. This function uses the ISSM mesh connectivity directly,
%   rather than re-triangulating with delaunay, so interior holes are
%   preserved. ISSM itself is not required by this plotting helper after md
%   has been loaded.
%
% Examples:
%   md = loadmodel(config.smoke_inversion_path);
%   plot_fem_field(md, md.results.StressbalanceSolution.Vel);

FONT_SIZE = 28;
opts = parseOptions(FONT_SIZE, varargin{:});
ax = opts.Parent;
if isempty(ax)
    ax = gca;
end

panel_data = double(values(:));
faces = normalizeFaces(md.mesh.elements);
if numel(panel_data) ~= md.mesh.numberofvertices
    error('plot_fem_field expects one scalar value per mesh vertex.');
end

keep_faces = all(isfinite(panel_data(faces)), 2);
faces = faces(keep_faces, :);

x_km = double(md.mesh.x(:)) / 1e3;
y_km = double(md.mesh.y(:)) / 1e3;
h = trisurf(faces, x_km, y_km, zeros(size(x_km)), panel_data, ...
    'Parent', ax, 'EdgeColor', 'none', 'FaceColor', 'interp');
view(ax, 2);
shading(ax, 'interp');
axis(ax, 'equal');
axis(ax, 'tight');
box(ax, 'on');
grid(ax, 'off');
set(ax, 'FontSize', opts.FontSize);
xlabel(ax, 'x_{ps} [km]', 'FontSize', opts.FontSize);
ylabel(ax, 'y_{ps} [km]', 'FontSize', opts.FontSize);
colormap(ax, opts.Colormap);

if ~isempty(opts.Caxis)
    clim(ax, opts.Caxis);
end
if ~isempty(opts.Title)
    title(ax, opts.Title, 'FontWeight', 'bold', 'Interpreter', 'none', ...
        'FontSize', opts.FontSize);
end
cb = colorbar(ax);
cb.FontSize = opts.FontSize;
if ~isempty(opts.ColorbarLabel)
    cb.Label.String = opts.ColorbarLabel;
    cb.Label.FontSize = opts.FontSize;
end
if ~isempty(opts.Annotation)
    addTransparentTextBox(ax, opts.Annotation, opts.FontSize);
end
end

function addTransparentTextBox(ax, annotation_text, font_size)
text_handle = text(ax, 0.02, 0.98, annotation_text, ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'HorizontalAlignment', 'left', 'FontName', 'Monospaced', ...
    'FontSize', font_size, 'Interpreter', 'none', ...
    'Color', [0 0 0], 'Margin', 6);
drawnow;
extent = text_handle.Extent;
pad_x = 0.006;
pad_y = 0.010;
box_x = [
    extent(1) - pad_x
    extent(1) + extent(3) + pad_x
    extent(1) + extent(3) + pad_x
    extent(1) - pad_x
];
box_y = [
    extent(2) - pad_y
    extent(2) - pad_y
    extent(2) + extent(4) + pad_y
    extent(2) + extent(4) + pad_y
];
x_limits = xlim(ax);
y_limits = ylim(ax);
box_x_data = x_limits(1) + box_x * diff(x_limits);
box_y_data = y_limits(1) + box_y * diff(y_limits);
patch(ax, 'XData', box_x_data, 'YData', box_y_data, ...
    'ZData', ones(size(box_x_data)), 'FaceColor', [1 1 1], ...
    'FaceAlpha', 0.4, 'EdgeColor', [0.25 0.25 0.25], ...
    'LineWidth', 0.5);
uistack(text_handle, 'top');
end

function opts = parseOptions(FONT_SIZE, varargin)
opts = struct();
opts.Parent = [];
opts.Caxis = [];
opts.Title = '';
opts.ColorbarLabel = '';
opts.Annotation = '';
opts.Colormap = parula;
opts.FontSize = FONT_SIZE;

if mod(numel(varargin), 2) ~= 0
    error('plot_fem_field options must be name/value pairs.');
end
for k = 1:2:numel(varargin)
    name = lower(varargin{k});
    value = varargin{k + 1};
    switch name
        case 'parent'
            opts.Parent = value;
        case 'caxis'
            opts.Caxis = value;
        case 'title'
            opts.Title = value;
        case 'colorbarlabel'
            opts.ColorbarLabel = value;
        case 'annotation'
            opts.Annotation = value;
        case 'colormap'
            opts.Colormap = value;
        case 'fontsize'
            opts.FontSize = value;
        otherwise
            error('Unknown plot_fem_field option: %s', varargin{k});
    end
end
end

function faces = normalizeFaces(elements)
faces = double(elements);
if size(faces, 2) ~= 3 && size(faces, 1) == 3
    faces = faces';
end
if size(faces, 2) ~= 3
    error('plot_fem_field expects triangular mesh elements.');
end
end
