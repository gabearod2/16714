function run_all_scripts(rootDir)
% RUN_ALL_SCRIPTS  Run all .m scripts in a folder, quietly.
% - Lets scripts use CLEAR normally (memory resets within each script)
% - Blocks CLC so command window output isn't cleared
% - Hides figures and closes them after each script
% - Suppresses script prints (only shows per-script status)
% - Stops on first error and rethrows it
%
% Usage:
%   run_all_scripts
%   run_all_scripts('path/to/folder')

    if nargin < 1 || isempty(rootDir), rootDir = pwd; end
    if ~isfolder(rootDir), error('Not a folder: %s', rootDir); end
    rootDir = char(rootDir);

    % Hide figures globally; restore on exit
    oldFigVis = get(0,'DefaultFigureVisible');
    set(0,'DefaultFigureVisible','off');
    cleanupFig = onCleanup(@() set(0,'DefaultFigureVisible',oldFigVis)); 

    % Suppress warnings (still captured by evalc)
    oldWarn = warning; warning('off','all');
    cleanupWarn = onCleanup(@() warning(oldWarn));

    % Gather top-level .m
    S = dir(fullfile(rootDir,'*.m'));
    fnames = {S.name};

    % Exclude this file if colocated
    thisfile = [mfilename '.m'];
    fnames(strcmp(fnames,thisfile)) = [];

    % Keep only scripts (first non-comment line not function/classdef)
    scripts = {};
    for i = 1:numel(fnames)
        fp = fullfile(rootDir, fnames{i});
        if is_script_file(fp), scripts{end+1} = fp; end 
    end
    scripts = sort(scripts);

    fprintf('Found %d script(s) in %s\n', numel(scripts), rootDir);

    % Run each script; stop on first error
    for i = 1:numel(scripts)
        f = scripts{i};
        fprintf('[%2d/%2d] Running %s ... ', i, numel(scripts), f);
        try
            % Capture and suppress script output; block clc; allow clear
            evalc(sprintf('safe_run(''%s'')', f));
            close all force;
            fprintf('OK\n');
        catch ME
            fprintf('FAILED\n');
            fprintf('%s\n', getReport(ME,'basic','hyperlinks','off'));
            rethrow(ME);
        end
    end

    fprintf('All scripts completed successfully.\n');
end

% ---------------- SUBFUNCTIONS (no nested funcs above!) ------------------

function safe_run(fname)
% Run a script while blocking clc/home (but allowing clear/clearvars).
% Shadow with function-handle variables in this workspace so the script
% calls these instead of the built-ins.

    clc = @()[];           %#ok<NASGU>  % block command-window clear
    home = @()[];          %#ok<NASGU>  % some scripts use 'home' synonym
    commandwindow = @()[]; %#ok<NASGU>  % just in case scripts call it

    run(fname);  % executes in THIS workspace, so the shadows apply
end


function clc(varargin) 
% No-op to prevent scripts from clearing the Command Window.
% Accepts any args so 'clc', 'home' or variants won't error.
end

function tf = is_script_file(filepath)
% True if first non-comment line is not 'function' or 'classdef'.
    txt = fileread(filepath);
    lines = regexp(txt,'\r\n|\n','split');
    firstCode = '';
    for k = 1:numel(lines)
        L = strtrim(lines{k});
        if isempty(L) || startsWith(L,'%'), continue; end
        firstCode = L; break;
    end
    tf = ~isempty(firstCode) && ~startsWith(firstCode,{'function','classdef'});
end
