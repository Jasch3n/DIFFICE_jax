function [md, iceedge_nodes, grounding_line_nodes, island_boundary_nodes] = ...
    set_shelf_boundary_conditions(config, md, bedmachine_data)
%SET_SHELF_BOUNDARY_CONDITIONS Apply shelf stress-balance boundary conditions.
%
% Syntax:
%   [md, iceedge_nodes, grounding_line_nodes, island_nodes] = set_shelf_boundary_conditions(config, md, bedmachine_data);
%
% Required inputs:
%   config - struct from shelf_config with front_probe_distances,
%       mesh_domain_file and grounding_line_tolerance.
%   md - ISSM model with BedMachine mask and MEaSURES observations already set.
%   bedmachine_data - struct with x, y, mask_grid, and interpolated mask.
%
% Outputs:
%   md - model with ice-ocean front left as Neumann and solid boundaries
%       constrained to observed velocity.
%   iceedge_nodes, grounding_line_nodes, island_boundary_nodes - logical
%       vertex masks for boundary diagnostics. The names mirror
%       iceedge_idx/grounding_idx in aashray_amery.m, but are logical masks.
%
% Saved artifacts:
%   None.
%
% Assumptions:
%   Mesh coordinates, BedMachine grids, and outline profiles are EPSG:3031.
%   BedMachine mask codes are 0 ocean, 1 ice-free land, 2 grounded ice,
%   3 floating ice, and 4 Lake Vostok. Only mask 0 is an ice-ocean front;
%   masks 1, 2, and 4 are solid boundary constraints.
%
% Examples:
%   cd examples/real_data/ISSMAdjoint/Amery
%   steps = [1 2 3 4];
%   Amery_Inversion

boundary_nodes = find(md.mesh.vertexonboundary);
iceedge_nodes = false(md.mesh.numberofvertices, 1);
grounding_line_nodes = false(md.mesh.numberofvertices, 1);
island_boundary_nodes = false(md.mesh.numberofvertices, 1);

bedmachine_mask = bedmachine_data.mask;
if any(~isfinite(bedmachine_mask(boundary_nodes)))
    warning('BedMachine mask is missing at %d boundary vertices.', ...
        nnz(~isfinite(bedmachine_mask(boundary_nodes))));
end

local_mask = round(bedmachine_mask);
ocean_boundary = false(numel(boundary_nodes), 1);
grounded_boundary = false(numel(boundary_nodes), 1);
classified_boundary = false(numel(boundary_nodes), 1);
direct_ocean = local_mask(boundary_nodes) == 0;
ocean_boundary(direct_ocean) = true;
classified_boundary(direct_ocean) = true;
probe_angles = linspace(0, 2 * pi, 9);
probe_angles(end) = [];
for k = 1:numel(config.front_probe_distances)
    distance_ocean = false(numel(boundary_nodes), 1);
    distance_grounded = false(numel(boundary_nodes), 1);
    for angle_id = 1:numel(probe_angles)
        needs_classification = ~classified_boundary;
        if ~any(needs_classification)
            break;
        end
        sample_x = md.mesh.x(boundary_nodes) + ...
            config.front_probe_distances(k) * cos(probe_angles(angle_id));
        sample_y = md.mesh.y(boundary_nodes) + ...
            config.front_probe_distances(k) * sin(probe_angles(angle_id));
        sample_mask = InterpFromGridToMesh( ...
            bedmachine_data.x, bedmachine_data.y, bedmachine_data.mask_grid, ...
            sample_x, sample_y, NaN);
        sample_mask = round(sample_mask);
        distance_ocean = distance_ocean | ...
            (needs_classification & sample_mask == 0);
        distance_grounded = distance_grounded | ...
            (needs_classification & isSolidMask(sample_mask));
    end
    sampled_ocean = ~classified_boundary & distance_ocean;
    sampled_grounded = ~classified_boundary & ~distance_ocean & ...
        distance_grounded;
    ocean_boundary(sampled_ocean) = true;
    grounded_boundary(sampled_grounded) = true;
    classified_boundary(sampled_ocean | sampled_grounded) = true;
end
iceedge_nodes(boundary_nodes(ocean_boundary)) = true;
grounding_line_nodes(boundary_nodes(grounded_boundary)) = true;

if isfield(config, 'mesh_domain_file') && isfile(config.mesh_domain_file)
    island_contours = readIslandContoursFromOutline(config.mesh_domain_file);
    island_boundary_nodes(boundary_nodes) = nodesNearContours( ...
        md.mesh.x(boundary_nodes), md.mesh.y(boundary_nodes), ...
        island_contours, config.grounding_line_tolerance);
end
grounding_line_nodes(island_boundary_nodes) = false;
iceedge_nodes(grounding_line_nodes | island_boundary_nodes) = false;

md.stressbalance.spcvx = NaN * ones(md.mesh.numberofvertices, 1);
md.stressbalance.spcvy = NaN * ones(md.mesh.numberofvertices, 1);
md.stressbalance.spcvz = NaN * ones(md.mesh.numberofvertices, 1);
md.stressbalance.referential = NaN * ones(md.mesh.numberofvertices, 6);
md.stressbalance.loadingforce = zeros(md.mesh.numberofvertices, 3);

ice_present = md.mask.ice_levelset <= 0;
iceedge_nodes = iceedge_nodes & ice_present;
md.mask.ice_levelset(ice_present) = -1;
md.mask.ice_levelset(iceedge_nodes) = 0;

solid_boundary_nodes = ~iceedge_nodes(boundary_nodes) | ...
    island_boundary_nodes(boundary_nodes);
dirichlet_nodes = boundary_nodes(solid_boundary_nodes & ...
    md.mask.ice_levelset(boundary_nodes) < 0);
md.stressbalance.spcvx(dirichlet_nodes) = md.inversion.vx_obs(dirichlet_nodes);
md.stressbalance.spcvy(dirichlet_nodes) = md.inversion.vy_obs(dirichlet_nodes);
md.stressbalance.spcvz(dirichlet_nodes) = 0;

md.smb = initialize(md.smb, md);
md.basalforcings = initialize(md.basalforcings, md);
if isnan(md.balancethickness.thickening_rate)
    md.balancethickness.thickening_rate = ...
        zeros(md.mesh.numberofvertices, 1);
    disp('      no balancethickness.thickening_rate specified: values set as zero');
end
md.masstransport.spcthickness = NaN * ones(md.mesh.numberofvertices, 1);
md.balancethickness.spcthickness = NaN * ones(md.mesh.numberofvertices, 1);
md.damage.spcdamage = NaN * ones(md.mesh.numberofvertices, 1);
end

function island_contours = readIslandContoursFromOutline(outline_file)
contours = expread(outline_file);
if numel(contours) <= 1
    island_contours = struct('x', {}, 'y', {}, 'density', {}, 'name', {});
    return;
end

island_contours = contours(2:end);
end

function solid = isSolidMask(mask)
mask = round(mask);
solid = mask == 1 | mask == 2 | mask == 4;
end

function near = nodesNearContours(x, y, contours, tolerance)
near = false(size(x));
tolerance_squared = tolerance^2;

for contour_id = 1:numel(contours)
    contour_x = contours(contour_id).x(:);
    contour_y = contours(contour_id).y(:);
    for segment_id = 1:(numel(contour_x) - 1)
        x1 = contour_x(segment_id);
        y1 = contour_y(segment_id);
        x2 = contour_x(segment_id + 1);
        y2 = contour_y(segment_id + 1);

        dx = x2 - x1;
        dy = y2 - y1;
        segment_length_squared = dx^2 + dy^2;
        if segment_length_squared == 0
            distance_squared = (x - x1).^2 + (y - y1).^2;
        else
            t = ((x - x1) * dx + (y - y1) * dy) ./ ...
                segment_length_squared;
            t = min(1, max(0, t));
            projected_x = x1 + t * dx;
            projected_y = y1 + t * dy;
            distance_squared = (x - projected_x).^2 + ...
                (y - projected_y).^2;
        end

        near = near | distance_squared <= tolerance_squared;
        if all(near)
            return;
        end
    end
end
end
