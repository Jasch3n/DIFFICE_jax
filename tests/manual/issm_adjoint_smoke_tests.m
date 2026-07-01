% ISSM adjoint shelf pipeline smoke-test commands.
%
% Run these manually from the repository root or copy individual commands into
% MATLAB. Mesh and solve steps require a working ISSM installation and the
% configured BedMachine/MEaSURES NetCDF files.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
shared_dir = fullfile(repo_root, 'examples', 'real_data', ...
    'ISSMAdjoint', 'shared');
addpath(genpath(shared_dir));

% Outline-only smoke tests. These generate Geometry/<Shelf>_Outline.exp and
% Geometry/<Shelf>_GL_preview.png from BedMachine.
shelves = {'Amery', 'LarsenC', 'LarsenD', 'RnFlch', 'Ross'};
config_files = {'amery.yaml', 'larsenc.yaml', 'larsend.yaml', ...
    'rnflch.yaml', 'ross.yaml'};
for k = 1:numel(shelves)
    config = shelf_config(fullfile(repo_root, 'examples', 'real_data', ...
        'ISSMAdjoint', 'configs', config_files{k}));
    build_bedmachine_outline(config);
end

% Mesh-only examples. Run one shelf at a time because BAMG and data subsets can
% take time and memory.
% cd examples/real_data/ISSMAdjoint/Amery
% steps = [1];
% Amery_Inversion
%
% cd examples/real_data/ISSMAdjoint/LarsenC
% steps = [1];
% LarsenC_Inversion

% Parameterization-only example after a mesh exists.
% cd examples/real_data/ISSMAdjoint/Amery
% steps = [2];
% Amery_Inversion

% Full inversion example. This writes Results/<Shelf>_* outputs and should be
% run only after the outline preview and parameterization diagnostics look
% reasonable.
% cd examples/real_data/ISSMAdjoint/Amery
% steps = [1 2 3 4];
% Amery_Inversion
