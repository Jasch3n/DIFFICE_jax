function outputs = LarsenC_GL(output_dir)
%LARSENC_GL Compatibility wrapper for Larsen C BedMachine outline generation.
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
addpath(fullfile(fileparts(script_dir), 'shared'));
config = shelf_config('LarsenC');
if nargin >= 1 && ~isempty(output_dir)
    config.geometry_dir = output_dir;
    config.mesh_domain_file = fullfile(output_dir, 'LarsenC_Outline.exp');
    config.gl_preview_file = fullfile(output_dir, 'LarsenC_GL_preview.png');
end
outputs = build_bedmachine_outline(config);
end
