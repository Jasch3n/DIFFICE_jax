% Ronne-Filchner ISSM adjoint inversion entrypoint.
%
% Usage:
%   cd examples/real_data/ISSMAdjoint/RnFlch
%   steps = [1 2];
%   RnFlch_Inversion

if ~exist('steps', 'var')
    steps = [1 2 3 4];
end
clearvars -except steps;

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(fullfile(fileparts(script_dir), 'shared'));

config = shelf_config('RnFlch');
run_shelf_inversion_steps(config, steps);
