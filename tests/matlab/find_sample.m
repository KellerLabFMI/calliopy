function f = find_sample(sample_dir, ext)
% FIND_SAMPLE  Full path of the first file in sample_dir ending in ext, or '' if none.
d = dir(fullfile(sample_dir, ['*' ext]));
d = d(~[d.isdir]);
if isempty(d)
    fprintf('no %s sample, skipped\n', ext);
    f = '';
else
    f = fullfile(d(1).folder, d(1).name);
end
end
