function outputs = Amery_GL(output_dir)
%AMERY_GL Compatibility wrapper for Amery BedMachine outline generation.
%
% Syntax:
%   outputs = Amery_GL();
%   outputs = Amery_GL('Geometry');
%
% Required inputs:
%   output_dir - optional Geometry directory override.
%
% Outputs and saved artifacts:
%   Delegates to build_bedmachine_outline and writes Amery_Outline.exp plus
%   Amery_GL_preview.png.
%
% Assumptions:
%   The Amery ROI outline is shelf-local and translated to EPSG:3031 by the
%   configured override [-2.55e6, -2.15e6]. BedMachine mask codes follow v4.
%   ISSM expread/expwrite are available through config.issm_dir.

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(fullfile(fileparts(script_dir), 'shared'));
config = shelf_config('Amery');
if nargin >= 1 && ~isempty(output_dir)
    config.geometry_dir = output_dir;
    config.mesh_domain_file = fullfile(output_dir, 'Amery_Outline.exp');
    config.gl_preview_file = fullfile(output_dir, 'Amery_GL_preview.png');
end
outputs = build_bedmachine_outline(config);
end
