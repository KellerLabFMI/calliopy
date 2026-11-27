function make_reference(sample_dir, out_dir)
% MAKE_REFERENCE  Save MATLAB outputs of calliope readers for calliopy's tests.
%
%   make_reference(sample_dir, out_dir)
%
%   Run once in MATLAB with calliope r2003 on the path (svn checkout -r 2003).
%   sample_dir holds one small file per format    -> CALLIOPY_TESTDATA/samples
%   out_dir receives the MATLAB outputs (v7.3)    -> CALLIOPY_TESTDATA/reference
%
%   Each ref_<group>.m in this folder handles one group of readers and skips
%   formats that have no sample file. Re-running overwrites earlier outputs.

here = fileparts(mfilename('fullpath'));
addpath(here);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

info.matlab_version = version;
info.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
info.readini_path = which('readini');
info.sample_dir = sample_dir;
save(fullfile(out_dir, 'ref_info.mat'), '-struct', 'info', '-v7.3');

groups = {'ref_types', 'ref_aux', 'ref_imaging', 'ref_eeg', 'ref_containers', 'ref_widefield', ...
          'ref_core', 'ref_io_save', 'ref_analysis_core', 'ref_wf_analysis', ...
          'ref_twophoton', 'ref_rois', 'ref_projmeta'};
for k = 1:numel(groups)
    if exist(groups{k}, 'file') ~= 2
        fprintf('%s: not written yet, skipped\n', groups{k});
        continue
    end
    try
        feval(groups{k}, sample_dir, out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', groups{k}, err.message);
    end
end
end
