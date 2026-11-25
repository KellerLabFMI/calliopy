function ref_core(sample_dir, out_dir) %#ok<INUSL>
% REF_CORE  MATLAB outputs of the deterministic functions behind calliopy.analysis,
% calliopy.preprocessing, calliopy.viewstack, calliopy.plotting.cmap_centered and the
% line-shift helpers of calliopy.data, for calliopy's tests.
%
%   ref_core(sample_dir, out_dir)
%
%   Every block builds a small synthetic input, calls the ORIGINAL calliope function and
%   saves the inputs followed by the outputs with save_ref as ref_core_<name>.mat (out1,
%   out2, ... in the order given in the block; input_file is 'synthetic').
%   tests/test_core_ref.py loads these files. sample_dir is not used.
%
%   NOT covered: nbstrp, nbstrp_get_indices and plotBSTRP draw random numbers that Python
%   cannot reproduce (tested for their properties instead); Violin/violinplot are
%   re-implemented, not ported; view_stack is interactive; getExpLog, read_info_from_ExpLog,
%   find_adata_file, get_adata_filename and load_exp need the ExpLog database and lab folders.

rng(1);
blocks = {@ref_ampd, @ref_frame_times, @ref_rot_ind, @ref_line_shift, @ref_cmap};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
end
end

function ref_ampd(out_dir)
% out1 = x, out2..out4 = pks of 'original', 'extended', 'adaptive' (window 120),
% out5 = l_scale of 'extended' (debug), out6 = adaptive_scale (debug)
t = linspace(0, 12*pi, 600);
x = sin(t) + 0.3*sin(3.1*t) + 0.2*randn(size(t)) + linspace(0, 2, numel(t));
p1 = ampd(x, 'mode', 'original');
p2 = ampd(x, 'mode', 'extended');
p3 = ampd(x, 'mode', 'adaptive', 'window', 120);
[~, ~, ~, ls] = ampd(x, 'mode', 'extended', 'debug', true);
[~, ~, as] = ampd(x, 'mode', 'adaptive', 'window', 120, 'debug', true);
save_ref(out_dir, 'core_ampd', 'synthetic', x, double(p1), double(p2), double(p3), double(ls), double(as));
end

function ref_frame_times(out_dir)
% out1 = trig_ch (regular galvo), out2 = get_frame_times(out1),
% out3 = trig_ch at minimum before the first frame, out4 = get_frame_times(out3)
ramp = [linspace(-1, 1, 110) ones(1, 40)];
sig = [zeros(1, 60) repmat(ramp, 1, 7) -ones(1, 50)];
sig = sig + 0.01*randn(size(sig));
sig2 = [-ones(1, 60) repmat(ramp, 1, 5) -ones(1, 50)];
save_ref(out_dir, 'core_frame_times', 'synthetic', sig, get_frame_times(sig), sig2, get_frame_times(sig2));
end

function ref_rot_ind(out_dir)
% out1 = dims, then for (cf, tx, ty) in out2 (one row each) the indices in out3, out4, ...
dims = [5 4 6];
cases = [3 0 0; 3 2 0; 3 0 -1; 4 1 2; 1 -3 0; 6 2 2];
outs = cell(1, size(cases, 1));
for k = 1:size(cases, 1)
    outs{k} = get_rot_ind(dims, cases(k, 1), cases(k, 2), cases(k, 3));
end
save_ref(out_dir, 'core_rot_ind', 'synthetic', dims, cases, outs{:});
end

function ref_line_shift(out_dir)
% out1 = data (3 frames, odd rows shifted by 3 columns), out2 = estimate_line_shift(mean),
% out3 = correct_line_shift(out1), out4 = its shift output
img = imgaussfilt(randn(48, 96), [5 2]);
img(1:2:end, :) = circshift(img(1:2:end, :), [0 3]);
data = repmat(img, 1, 1, 3) + 0.01*randn(48, 96, 3);
[corrected, shift] = correct_line_shift(data);
save_ref(out_dir, 'core_line_shift', 'synthetic', data, estimate_line_shift(mean(data, 3)), corrected, shift);
end

function ref_cmap(out_dir)
% for each (name, m, center, clim) row: out1 = clims (one row per case), out2 = centers,
% out3 = m values, out4.. = cmap_centered(name, m, center) with the axes CLim set first
names = {'RdBu', 'RdBu', 'PuOr', 'Spectral', 'RdBu', 'BrBg'};
clims = [-1 1; -1 3; 0.5 2; -5 0.2; 2 4; -2 -1];
centers = [0 0 0 0 3 0];
ms = [11 100 64 33 50 7];
fig = figure('Visible', 'off');
ax = axes(fig);
outs = cell(1, numel(names));
for k = 1:numel(names)
    set(ax, 'CLim', clims(k, :));
    axes(ax); %#ok<LAXES>
    outs{k} = cmap_centered(names{k}, ms(k), centers(k));
end
close(fig);
save_ref(out_dir, 'core_cmap', 'synthetic', clims, centers, ms, outs{:});
end
