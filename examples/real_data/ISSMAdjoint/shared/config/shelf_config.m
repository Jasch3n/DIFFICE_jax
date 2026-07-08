function config = shelf_config(config_source)
%SHELF_CONFIG Load an ISSM adjoint shelf inversion experiment config.
%
% Syntax:
%   config = shelf_config('configs/amery.yaml');
%   config = shelf_config('/absolute/path/to/ross.yaml');
%
% Compatibility:
%   config = shelf_config('Amery') maps to configs/amery.yaml.
%
% Output:
%   config - flat struct consumed by the existing ISSMAdjoint workflow
%       modules. The YAML file is the public experiment interface; this
%       loader handles defaults, relative paths, derived artifacts, compact
%       values such as logspace, and validation.
%
% Assumptions:
%   The YAML reader supports the schema used by configs/*.yaml: nested maps,
%   scalar strings/numbers/booleans, [] arrays, {}, .nan, and logspace maps.

if nargin < 1 || isempty(config_source)
    error('shelf_config requires a config path or shelf name.');
end

shared_dir = fileparts(mfilename('fullpath'));
config_path = resolveConfigPath(config_source, shared_dir);
raw_config = readShelfYaml(config_path);
defaults = config_defaults();
spec = mergeStructs(defaults, raw_config);
config = config_paths(spec, config_path, shared_dir);
validateConfig(config);
end

function config_path = resolveConfigPath(config_source, shared_dir)
config_source = char(config_source);
if isfile(config_source)
    config_path = absolutePath(config_source);
    return;
end

[~, ~, ext] = fileparts(config_source);
if any(strcmpi(ext, {'.yaml', '.yml'}))
    candidate = absolutePath(config_source);
    if isfile(candidate)
        config_path = candidate;
        return;
    end
    error('Config file does not exist: %s', config_source);
end

canonical_name = canonicalShelfName(config_source);
config_name = lower(canonical_name);
config_path = fullfile(fileparts(shared_dir), 'configs', ...
    sprintf('%s.yaml', config_name));
if ~isfile(config_path)
    error(['Unsupported shelf "%s". Add examples/real_data/ISSMAdjoint/', ...
        'configs/%s.yaml or pass a config path.'], config_source, ...
        config_name);
end
warning(['shelf_config(''%s'') is kept for compatibility. Prefer ', ...
    'shelf_config(''%s'').'], config_source, config_path);
end

function path = absolutePath(path)
path = char(path);
if isAbsolutePath(path)
    return;
end
path = char(java.io.File(fullfile(pwd, path)).getCanonicalPath());
end

function tf = isAbsolutePath(path)
if ispc
    tf = ~isempty(regexp(path, '^[A-Za-z]:[\\/]', 'once')) || ...
        startsWith(path, '\\');
else
    tf = startsWith(path, filesep);
end
end

function canonical_name = canonicalShelfName(shelf_name)
normalized = lower(regexprep(strtrim(char(shelf_name)), '[^a-zA-Z0-9]', ''));
switch normalized
    case 'amery'
        canonical_name = 'Amery';
    case {'larsenc', 'larsen'}
        canonical_name = 'LarsenC';
    case 'larsend'
        canonical_name = 'LarsenD';
    case {'rnflch', 'ronnefilchner', 'ronnefilchnerice'}
        canonical_name = 'RnFlch';
    case 'ross'
        canonical_name = 'Ross';
    otherwise
        canonical_name = shelf_name;
end
end

function raw = readShelfYaml(config_path)
lines = readlines(config_path);
parsed_lines = struct('text', {}, 'number', {});
for k = 1:numel(lines)
    text = stripYamlComment(char(lines(k)));
    if isempty(strtrim(text))
        continue;
    end
    parsed_lines(end + 1).text = text; %#ok<AGROW>
    parsed_lines(end).number = k;
end
if isempty(parsed_lines)
    raw = struct();
    return;
end
[raw, next_index] = parseYamlBlock(parsed_lines, 1, leadingSpaces( ...
    parsed_lines(1).text));
if next_index <= numel(parsed_lines)
    error('Unexpected YAML content at line %d.', ...
        parsed_lines(next_index).number);
end
end

function text = stripYamlComment(text)
comment_index = find(text == '#', 1);
if ~isempty(comment_index)
    text = extractBefore(text, comment_index);
end
text = char(text);
end

function [block, index] = parseYamlBlock(lines, index, expected_indent)
block = struct();
while index <= numel(lines)
    line = lines(index).text;
    indent = leadingSpaces(line);
    if indent < expected_indent
        return;
    end
    if indent > expected_indent
        error('Unexpected YAML indentation at line %d.', lines(index).number);
    end

    trimmed = strtrim(line);
    colon_index = find(trimmed == ':', 1);
    if isempty(colon_index)
        error('Unsupported YAML syntax at line %d: %s', ...
            lines(index).number, trimmed);
    end
    key = strtrim(trimmed(1:(colon_index - 1)));
    if isempty(regexp(key, '^[A-Za-z_][A-Za-z0-9_]*$', 'once'))
        error('Unsupported YAML key at line %d: %s', ...
            lines(index).number, key);
    end
    value_text = strtrim(trimmed((colon_index + 1):end));

    if isempty(value_text)
        [value, index] = parseYamlBlock(lines, index + 1, ...
            expected_indent + 2);
    else
        value = parseYamlValue(value_text, lines(index).number);
        index = index + 1;
    end
    block.(key) = value;
end
end

function count = leadingSpaces(text)
count = numel(text) - numel(regexprep(text, '^\s*', ''));
end

function value = parseYamlValue(value_text, line_number)
if strcmp(value_text, '{}')
    value = struct();
    return;
end
if strcmp(value_text, '[]')
    value = [];
    return;
end
if startsWith(value_text, '[') && endsWith(value_text, ']')
    value = parseInlineArray(value_text, line_number);
    return;
end

lower_value = lower(value_text);
switch lower_value
    case 'true'
        value = true;
        return;
    case 'false'
        value = false;
        return;
    case {'.nan', 'nan'}
        value = NaN;
        return;
    case {'null', '~'}
        value = [];
        return;
end

if (startsWith(value_text, '''') && endsWith(value_text, '''')) || ...
        (startsWith(value_text, '"') && endsWith(value_text, '"'))
    value = extractBetween(value_text, 2, strlength(value_text) - 1);
    value = char(value);
    return;
end

numeric_value = str2double(value_text);
if ~isnan(numeric_value)
    value = numeric_value;
else
    if contains(value_text, ':')
        error(['Unsupported inline YAML map at line %d. Use an indented ', ...
            'map instead.'], line_number);
    end
    value = char(value_text);
end
end

function values = parseInlineArray(value_text, line_number)
inner = strtrim(extractBetween(value_text, 2, strlength(value_text) - 1));
inner = char(inner);
if isempty(inner)
    values = [];
    return;
end
parts = strtrim(strsplit(inner, ','));
numeric_values = NaN(1, numel(parts));
all_numeric = true;
for k = 1:numel(parts)
    numeric_values(k) = str2double(parts{k});
    if isnan(numeric_values(k)) && ~any(strcmpi(parts{k}, {'nan', '.nan'}))
        all_numeric = false;
        break;
    end
end
if all_numeric
    values = numeric_values;
    return;
end

values = cell(1, numel(parts));
for k = 1:numel(parts)
    part = parts{k};
    if isempty(part)
        error('Empty YAML array entry at line %d.', line_number);
    end
    values{k} = parseYamlValue(part, line_number);
end
end

function merged = mergeStructs(defaults, overrides)
merged = defaults;
if isempty(overrides)
    return;
end
fields = fieldnames(overrides);
for k = 1:numel(fields)
    field = fields{k};
    override_value = overrides.(field);
    if isfield(merged, field) && isstruct(merged.(field)) && ...
            isstruct(override_value)
        merged.(field) = mergeStructs(merged.(field), override_value);
    else
        merged.(field) = override_value;
    end
end
end

function validateConfig(config)
required = {'shelf_name', 'config_path', 'base_dir', 'shelf_dir', ...
    'geometry_dir', 'results_dir', 'bedmachine_file', 'measures_file', ...
    'bedmachine_bounds', 'mesh_path', 'parameterized_path'};
for k = 1:numel(required)
    if ~isfield(config, required{k}) || isempty(config.(required{k}))
        error('Config field "%s" is required.', required{k});
    end
end
if numel(config.bedmachine_bounds) ~= 4
    error('data.bedmachine_bounds must have four entries: [xmin xmax ymin ymax].');
end
end
