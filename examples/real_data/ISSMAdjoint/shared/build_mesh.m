function md = build_mesh(config)
%BUILD_MESH Create and adapt a BAMG mesh for one shelf.
%
% Syntax:
%   config = shelf_config('Amery');
%   md = build_mesh(config);
%
% Required input:
%   config - struct from shelf_config with mesh_domain_file, model_dir,
%       mesh_path, BedMachine/MEaSURES paths, and BAMG mesh parameters.
%
% Output:
%   md - ISSM model with EPSG:3031 mesh coordinates.
%
% Saved artifacts:
%   Results/<Shelf>_Mesh.mat and Geometry/<Shelf>_mesh.png.
%
% Assumptions:
%   Geometry/<Shelf>_Outline.exp is a direct BedMachine v4 EPSG:3031 outline.
%   BedMachine mask codes are not interpreted here, except thickness is used
%   as a mesh adaptation field. ISSM must be available through config.issm_dir.
%
% Examples:
%   cd examples/real_data/ISSMAdjoint/LarsenC
%   steps = [1 2];
%   LarsenC_Inversion

required = {'mesh_domain_file', 'geometry_dir', 'model_dir', 'mesh_path', ...
    'issm_dir', 'bedmachine_file', 'measures_file'};
assertConfigFields(config, required);
if ~isfile(config.mesh_domain_file)
    build_bedmachine_outline(config);
end
assert(isfile(config.bedmachine_file), 'Missing BedMachine file: %s', ...
    config.bedmachine_file);
assert(isfile(config.measures_file), 'Missing MEaSURES file: %s', ...
    config.measures_file);
helpers('ensure_directory', config.model_dir);
helpers('bootstrap_issm_path', config.issm_dir);

fprintf('Step 1: Mesh creation and velocity-based adaptation for %s\n', ...
    config.shelf_name);

md = bamg(model, 'domain', config.mesh_domain_file, ...
    'hmax', config.mesh_initial_hmax, 'maxnbv', config.mesh_maxnbv);

data_md = md;
nodes_trans = [data_md.mesh.x(:), data_md.mesh.y(:)]; %#ok<NASGU>
bounds = helpers('mesh_bounds', data_md, config.data_padding);
[vx_grid_x, vx_grid_y, vx_grid] = helpers('read_grid_subset', ...
    config.measures_file, 'VX', bounds);
[~, ~, vy_grid] = helpers('read_grid_subset', config.measures_file, 'VY', bounds);
[thickness_grid_x, thickness_grid_y, thickness_grid] = helpers('read_grid_subset', ...
    config.bedmachine_file, 'thickness', bounds);

ud_on_nodes = InterpFromGridToMesh(vx_grid_x, vx_grid_y, vx_grid, ...
    data_md.mesh.x, data_md.mesh.y, NaN);
vd_on_nodes = InterpFromGridToMesh(vx_grid_x, vx_grid_y, vy_grid, ...
    data_md.mesh.x, data_md.mesh.y, NaN);
vel_obs = sqrt(ud_on_nodes.^2 + vd_on_nodes.^2);
vel_for_mesh = helpers('replace_nan_with_median', vel_obs);
hd_on_nodes = InterpFromGridToMesh( ...
    thickness_grid_x, thickness_grid_y, thickness_grid, ...
    data_md.mesh.x, data_md.mesh.y, NaN);
hd_for_mesh = helpers('replace_nan_with_median', hd_on_nodes);
adaptation_fields = [vel_for_mesh hd_for_mesh];

md = bamg(md, 'hmax', config.mesh_hmax, 'hmin', config.mesh_hmin, ...
    'gradation', config.mesh_gradation, 'field', adaptation_fields, ...
    'err', config.mesh_adaptation_error, 'Metrictype', 2, ...
    'maxnbv', config.mesh_maxnbv);
md.mesh.epsg = 3031;

plotmodel(md, 'figure', 1, 'data', 'mesh', ...
    'title', sprintf('%s mesh', config.shelf_name));
if isfield(config, 'mesh_preview_file') && ~isempty(config.mesh_preview_file)
    saveas(gcf, config.mesh_preview_file);
end
save(config.mesh_path, 'md', '-v7.3');
end

function assertConfigFields(config, fields)
for k = 1:numel(fields)
    if ~isfield(config, fields{k})
        error('config.%s is required.', fields{k});
    end
end
end
