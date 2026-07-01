% Ross ISSM adjoint inversion entrypoint.
%
% Usage:
%   cd examples/real_data/ISSMAdjoint/Ross
%   steps = [1 2];
%   Ross_Inversion

if ~exist('steps', 'var')
    steps = [1 2 3 4];
end
clearvars -except steps;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
adjoint_dir = fileparts(script_dir);
addpath(genpath(fullfile(adjoint_dir, 'shared')));

config = shelf_config(fullfile(adjoint_dir, 'configs', 'ross.yaml'));
run_shelf_inversion_steps(config, steps);
