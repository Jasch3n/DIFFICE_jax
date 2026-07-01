function [md, bedmachine_data, valid_velocity] = parameterize_from_bedmachine_measures(config, md)
%PARAMETERIZE_FROM_BEDMACHINE_MEASURES Set shelf geometry, mask, and fields.
%
% Syntax:
%   config = shelf_config('configs/amery.yaml');
%   md = loadmodel(config.mesh_path);
%   [md, bedmachine_data, valid_velocity] = parameterize_from_bedmachine_measures(config, md);
%
% Required inputs:
%   config - struct from shelf_config with data-source paths, mesh outline,
%       and physical defaults.
%   md     - ISSM model with mesh coordinates in EPSG:3031.
%
% Outputs:
%   md              - parameterized ISSM model ready for stress balance.
%   bedmachine_data - struct with subset grids and interpolated mask.
%   valid_velocity  - logical vector where MEaSURES vx/vy were finite. This
%       is the inverse of original_nan_mask in aashray_amery.m terminology.
%
% Saved artifacts:
%   Results/<Shelf>_Parameterization.mat when config.parameterized_path exists.
%
% Assumptions:
%   The mesh is already in EPSG:3031. BedMachine mask codes are 0 ocean,
%   1 ice-free land, 2 grounded ice, 3 floating ice, and 4 Lake Vostok. ISSM
%   classes and interpolation functions are available.
%
% Examples:
%   cd examples/real_data/ISSMAdjoint/Amery
%   steps = [1 2 3 4];
%   Amery_Inversion

required = {'shelf_name', 'mesh_domain_file', 'model_dir', ...
    'parameterized_path', 'bedmachine_file', 'measures_file'};
assertConfigFields(config, required);
helpers('bootstrap_issm_path', config.issm_dir);
helpers('ensure_directory', config.model_dir);

fprintf('Step 2: Parameterization from BedMachine and MEaSURES for %s\n', ...
    config.shelf_name);

md.miscellaneous.name = config.shelf_name;
md.mesh.epsg = 3031;
md = setmask(md, 'all', '');
md = setflowequation(md, 'SSA', 'all');

bounds = helpers('mesh_bounds', md, config.data_padding);
% Read gridded data. The original Amery script used hd, ud, and vd for
% thickness and velocity components; the same names are kept below once the
% fields have been interpolated onto mesh nodes.
[bed_x, bed_y, surface_grid] = helpers('read_grid_subset', ...
    config.bedmachine_file, 'surface', bounds);
[~, ~, thickness_grid] = helpers('read_grid_subset', ...
    config.bedmachine_file, 'thickness', bounds);
[~, ~, bed_grid] = helpers('read_grid_subset', config.bedmachine_file, 'bed', bounds);
[~, ~, bedmachine_mask_grid] = helpers('read_grid_subset', ...
    config.bedmachine_file, 'mask', bounds);
[vel_x, vel_y, vx_grid] = helpers('read_grid_subset', config.measures_file, 'VX', bounds);
[~, ~, vy_grid] = helpers('read_grid_subset', config.measures_file, 'VY', bounds);

nodes_trans = [md.mesh.x(:), md.mesh.y(:)];
md.geometry.surface = InterpFromGridToMesh(bed_x, bed_y, surface_grid, ...
    md.mesh.x, md.mesh.y, NaN);
hd_on_nodes = InterpFromGridToMesh(bed_x, bed_y, thickness_grid, ...
    md.mesh.x, md.mesh.y, NaN);
md.geometry.bed = InterpFromGridToMesh(bed_x, bed_y, bed_grid, ...
    md.mesh.x, md.mesh.y, NaN);
bedmachine_mask = InterpFromGridToMesh(bed_x, bed_y, bedmachine_mask_grid, ...
    md.mesh.x, md.mesh.y, NaN);
if any(~isfinite(bedmachine_mask))
    error('BedMachine mask interpolation failed at %d mesh vertices.', ...
        nnz(~isfinite(bedmachine_mask)));
end

md.geometry.thickness = max(hd_on_nodes, config.minimum_ice_thickness);
md.geometry.base = md.geometry.surface - md.geometry.thickness;
pos = find(~isfinite(md.geometry.bed));
md.geometry.bed(pos) = md.geometry.base(pos) - 1000;
md.geometry.hydrostatic_ratio = ones(md.mesh.numberofvertices, 1);
md = helpers('apply_bedmachine_mask', md, bedmachine_mask);

md.inversion = m1qn3inversion();
ud_on_nodes = InterpFromGridToMesh(vel_x, vel_y, vx_grid, ...
    md.mesh.x, md.mesh.y, NaN);
vd_on_nodes = InterpFromGridToMesh(vel_x, vel_y, vy_grid, ...
    md.mesh.x, md.mesh.y, NaN);
original_nan_mask = ~(isfinite(ud_on_nodes) & isfinite(vd_on_nodes));
valid_velocity = ~original_nan_mask;
ud_on_nodes_filled = helpers('fill_invalid', ud_on_nodes, 0);
vd_on_nodes_filled = helpers('fill_invalid', vd_on_nodes, 0);
md.inversion.vx_obs = ud_on_nodes_filled;
md.inversion.vy_obs = vd_on_nodes_filled;
md.inversion.vel_obs = sqrt(md.inversion.vx_obs.^2 + ...
    md.inversion.vy_obs.^2);

md.initialization.vx = md.inversion.vx_obs;
md.initialization.vy = md.inversion.vy_obs;
md.initialization.vz = zeros(md.mesh.numberofvertices, 1);
md.initialization.vel = sqrt(md.initialization.vx.^2 + ...
    md.initialization.vy.^2);
md.initialization.pressure = md.materials.rho_ice * md.constants.g * ...
    md.geometry.thickness;
md.initialization.temperature = config.initial_temperature * ...
    ones(md.mesh.numberofvertices, 1);

md.materials.rheology_n = 3 * ones(md.mesh.numberofelements, 1);
md.materials.rheology_B = cuffey(md.initialization.temperature);

md.friction.coefficient = config.grounded_friction_coefficient * ...
    ones(md.mesh.numberofvertices, 1);
md.friction.coefficient(md.mask.ocean_levelset < 0) = 0;
md.friction.p = ones(md.mesh.numberofelements, 1);
md.friction.q = ones(md.mesh.numberofelements, 1);

md.smb.mass_balance = zeros(md.mesh.numberofvertices, 1);
md.basalforcings.floatingice_melting_rate = ...
    zeros(md.mesh.numberofvertices, 1);
md.basalforcings.groundedice_melting_rate = ...
    zeros(md.mesh.numberofvertices, 1);
md.basalforcings.geothermalflux = zeros(md.mesh.numberofvertices, 1);
md.masstransport.spcthickness = NaN * ones(md.mesh.numberofvertices, 1);

bedmachine_data = struct();
bedmachine_data.x = bed_x;
bedmachine_data.y = bed_y;
bedmachine_data.mask_grid = bedmachine_mask_grid;
bedmachine_data.mask = bedmachine_mask;
bedmachine_data.nodes_trans = nodes_trans;
bedmachine_data.hd_on_nodes = hd_on_nodes;
bedmachine_data.ud_on_nodes = ud_on_nodes_filled;
bedmachine_data.vd_on_nodes = vd_on_nodes_filled;
bedmachine_data.original_nan_mask = original_nan_mask;

[md, iceedge_nodes, grounding_line_nodes, island_boundary_nodes] = ...
    set_shelf_boundary_conditions(config, md, bedmachine_data);
grounding_idx = find(grounding_line_nodes | island_boundary_nodes);
iceedge_idx = find(iceedge_nodes);
bedmachine_data.grounding_idx = grounding_idx;
bedmachine_data.iceedge_idx = iceedge_idx;

missing_geometry = ~isfinite(md.geometry.surface) | ...
    ~isfinite(md.geometry.thickness) | ~isfinite(md.geometry.base);
if any(missing_geometry)
    error('BedMachine interpolation failed at %d mesh vertices.', ...
        nnz(missing_geometry));
end
if any(~valid_velocity)
    warning('MEaSURES interpolation has %d missing velocity vertices.', ...
        nnz(~valid_velocity));
end
fprintf('BedMachine floating vertices used by ISSM: %d of %d\n', ...
    nnz(md.mask.ocean_levelset < 0), md.mesh.numberofvertices);
fprintf('BedMachine ice-edge/front boundary vertices: %d\n', ...
    nnz(iceedge_nodes));
fprintf('BedMachine-v4 GL boundary vertices: %d\n', ...
    nnz(grounding_line_nodes));
fprintf('BedMachine-v4 grounded-island boundary vertices: %d\n', ...
    nnz(island_boundary_nodes));

save(config.parameterized_path, 'md', '-v7.3');
end

function assertConfigFields(config, fields)
for k = 1:numel(fields)
    if ~isfield(config, fields{k})
        error('config.%s is required.', fields{k});
    end
end
end
