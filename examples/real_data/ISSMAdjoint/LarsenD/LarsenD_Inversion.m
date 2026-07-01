% Larsen D ISSM adjoint inversion entrypoint.
%
% Usage:
%   cd examples/real_data/ISSMAdjoint/LarsenD
%   steps = [1 2];
%   LarsenD_Inversion

if ~exist('steps', 'var')
    steps = [1 2 3 4];
end
clearvars -except steps;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(fullfile(fileparts(script_dir), 'shared'));

config = shelf_config('LarsenD');
run_shelf_inversion_steps(config, steps);
