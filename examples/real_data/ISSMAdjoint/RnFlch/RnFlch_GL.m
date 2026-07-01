function outputs = RnFlch_GL(output_dir)
%RNFLCH_GL Compatibility wrapper for Ronne-Filchner outline generation.
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
adjoint_dir = fileparts(script_dir);
addpath(genpath(fullfile(adjoint_dir, 'shared')));
config = shelf_config(fullfile(adjoint_dir, 'configs', 'rnflch.yaml'));
if nargin >= 1 && ~isempty(output_dir)
    config.geometry_dir = output_dir;
    config.mesh_domain_file = fullfile(output_dir, 'RnFlch_Outline.exp');
    config.gl_preview_file = fullfile(output_dir, 'RnFlch_GL_preview.png');
end
outputs = build_bedmachine_outline(config);
end
