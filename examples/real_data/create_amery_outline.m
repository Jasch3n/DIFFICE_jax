% Create an ARGUS .exp file from x_gd and y_gd column vectors.
%
% Expected input in workspace: x_gd, y_gd
% Output in workspace: component_id

points = [x_gd(:), y_gd(:)];

component_id = labelComponents(points);
contours = orderContours(points, component_id);

fid = fopen('DomainOutline.exp', 'w');

for component = 1:numel(contours)
    contour = contours{component};

    fprintf(fid, '## Name:Domain_%d\n', component);
    fprintf(fid, '## Icon:0\n');
    fprintf(fid, '# Points Count  Value\n');
    fprintf(fid, '%d 1\n', size(contour, 1));
    fprintf(fid, '# X pos Y pos\n');

    for k = 1:size(contour, 1)
        fprintf(fid, '%.12g %.12g\n', contour(k, 1), contour(k, 2));
    end
end

fclose(fid);

% -------------------------------------------------------------------------
function component_id = labelComponents(points)
    %LABELCOMPONENTS Assign one connected-component id to every raw point.
    %
    % Components are found with a breadth-first nearest-neighbor search. The
    % visited array is the cache of points already assigned to an earlier loop,
    % so every input point receives exactly one component label.
    npoints = size(points, 1);
    component_id = zeros(npoints, 1);
    visited = false(npoints, 1);
    component_count = 0;

    % On a regular grid, sqrt(2) times the grid spacing includes horizontal,
    % vertical, and diagonal neighbors without jumping across one-cell gaps.
    search_radius = sqrt(2) * gridSpacing(points);

    while any(~visited)
        component_count = component_count + 1;
        seed = find(~visited, 1);

        % Preallocate a BFS queue large enough for the whole component.
        frontier = zeros(npoints, 1);
        frontier_start = 1;
        frontier_end = 1;
        frontier(frontier_end) = seed;
        visited(seed) = true;
        component_id(seed) = component_count;

        while frontier_start <= frontier_end
            current = frontier(frontier_start);
            frontier_start = frontier_start + 1;

            % Search the full point set for unvisited points adjacent to the
            % current point on the native grid.
            distance = vecnorm(points - points(current, :), 2, 2);
            neighbors = find(~visited & distance <= search_radius);

            visited(neighbors) = true;
            component_id(neighbors) = component_count;

            next_end = frontier_end + numel(neighbors);
            frontier(frontier_end + 1:next_end) = neighbors;
            frontier_end = next_end;
        end
    end
end

% -------------------------------------------------------------------------
function contours = orderContours(points, component_id)
    %ORDERCONTOURS Build one closed ARGUS contour for each connected component.
    %
    % The raw grounding-line points can be thick, so each component is first
    % converted to the outer boundary of its grid mask. The boundary trace is
    % already ordered, so it should not be reordered with a nearest-neighbor
    % walk before passing it to BAMG.
    component_count = max(component_id);
    contours = cell(component_count, 1);
    spacing = gridSpacing(points);

    for component = 1:component_count
        component_points = points(component_id == component, :);
        contour = boundaryLoop(component_points, spacing);
        contours{component} = orientForBamg(contour, component > 1);
    end
end

% -------------------------------------------------------------------------
function contour = boundaryLoop(points, spacing)
    %BOUNDARYLOOP Trace the outer boundary of one thick connected component.
    %
    % The component points are snapped to a local grid mask. bwboundaries then
    % returns the outer pixel boundary in traversal order, which avoids the
    % self-intersections that can happen when a thick band is ordered by nearest
    % neighbor alone.
    min_x = min(points(:, 1));
    min_y = min(points(:, 2));

    % Use a one-cell padding so every component edge has background around it.
    cols = round((points(:, 1) - min_x) / spacing) + 2;
    rows = round((points(:, 2) - min_y) / spacing) + 2;

    % Rasterize the component onto its native grid.
    mask = false(max(rows) + 1, max(cols) + 1);
    point_index = sub2ind(size(mask), rows, cols);
    mask(point_index) = true;

    % Trace only the exterior boundary; interior gaps in a thick GL should not
    % become separate holes in this component.
    boundaries = bwboundaries(mask, 8, 'noholes');
    boundary = boundaries{1};

    contour = [
        min_x + (boundary(:, 2) - 2) * spacing, ...
        min_y + (boundary(:, 1) - 2) * spacing
    ];

    % bwboundaries can repeat vertices at diagonal contacts. BAMG rejects
    % duplicate geometry points, so keep the first visit to each coordinate.
    [~, keep] = unique(contour, 'rows', 'stable');
    contour = contour(sort(keep), :);

    contour(end + 1, :) = contour(1, :);
end

% -------------------------------------------------------------------------
function contour = orientForBamg(contour, is_hole)
    %ORIENTFORBAMG Match the contour orientation convention used by BAMG.
    %
    % BAMG expects the principal domain and holes to have opposite orientation.
    % The same signed-edge test appears in ISSM's bamg.m before it flips
    % incorrectly oriented profiles.
    open_contour = contour(1:end-1, :);
    next_contour = contour(2:end, :);
    orientation = sum((next_contour(:, 1) - open_contour(:, 1)) .* ...
                      (next_contour(:, 2) + open_contour(:, 2)));

    if (~is_hole && orientation > 0) || (is_hole && orientation < 0)
        contour = flipud(contour);
    end
end

% -------------------------------------------------------------------------
function spacing = gridSpacing(points)
    %GRIDSPACING Return the native spacing of the x/y coordinate grid.
    %
    % The Amery outline points are sampled on a regular grid, so the smallest
    % positive x or y increment is the spacing used for connectivity and masks.
    dx = diff(unique(points(:, 1)));
    dy = diff(unique(points(:, 2)));
    spacing = min([dx(dx > 0); dy(dy > 0)]);
end
