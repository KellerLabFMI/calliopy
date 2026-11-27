function ref_analysis_core(sample_dir, out_dir) %#ok<INUSL>
% REF_ANALYSIS_CORE  MATLAB outputs of the deterministic analysis-core functions
% (21_Analysis signal helpers, 06_Library utilities, 10_Behavior, ROIs2image, specTraces)
% for calliopy's tests.
%
%   ref_analysis_core(sample_dir, out_dir)
%
%   Every block builds a small synthetic input, calls the ORIGINAL calliope function and
%   saves the inputs followed by the outputs with save_ref as ref_analysis_core_<name>.mat
%   (out1, out2, ... in the order given in the block; input_file is 'synthetic').
%   tests/test_analysis_core_ref.py loads these files. sample_dir is not used.
%
%   NOT covered: nbstrp_old and plotBSTRP_multi (random numbers), plotSEM (graphics only),
%   preprocess_owl (needs ExpLog and an .avi; its core is tested synthetically), act2spks
%   and rastermap_sort (external Python packages).

rng(1);
blocks = {@ref_psmooth, @ref_smooth2, @ref_ftfil, @ref_ngsc, @ref_psd, @ref_envelope, ...
    @ref_vis_speed, @ref_library, @ref_behavior, @ref_find_saccades, @ref_rois2image, ...
    @ref_spec_traces};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
end
end

function ref_psmooth(out_dir)
% out1 = x (row), out2 = psmooth(x, 'win_size', 100), out3 = psmooth(x) (default window),
% out4 = x with NaNs, out5 = psmooth(out4, 'win_size', 60)
x = cumsum(randn(1, 1500)) + 5*sin((1:1500)/40);
xn = x(1:400); xn([50 51 300]) = NaN;
save_ref(out_dir, 'analysis_core_psmooth', 'synthetic', x, psmooth(x, 'win_size', 100), ...
    psmooth(x), xn, psmooth(xn, 'win_size', 60));
end

function ref_smooth2(out_dir)
% out1 = A (30 x 4), out2 = smooth2(A, 5), out3 = smooth2(A, 4, 2), out4 = v (row),
% out5 = smooth2(v, 7)
A = randn(30, 4); v = randn(1, 50);
save_ref(out_dir, 'analysis_core_smooth2', 'synthetic', A, smooth2(A, 5), smooth2(A, 4, 2), ...
    v, smooth2(v, 7));
end

function ref_ftfil(out_dir)
% out1 = x (3 x 500), out2 = ftfil(x, 100, 1, 20), out3 = ftfil(x', 100, 0.5, 30)
x = randn(3, 500);
save_ref(out_dir, 'analysis_core_ftfil', 'synthetic', x, ftfil(x, 100, 1, 20), ftfil(x', 100, 0.5, 30));
end

function ref_ngsc(out_dir)
% out1 = data (5 x 300), out2 = nGSC(data)
data = randn(5, 300) .* [1; 2; 0.5; 3; 1];
data(2, :) = data(2, :) + data(1, :);
save_ref(out_dir, 'analysis_core_ngsc', 'synthetic', data, nGSC(data));
end

function ref_psd(out_dir)
% out1 = x (row, 255), out2 = lin [5 20], out3 = dB [5 20], out4 = M (4 x 256),
% out5 = AvgPsdInRange(M, 50, [1 10], false, 2), out6 = AvgPsdInRange(M', 50, [1 10], true)
x = randn(1, 255); M = randn(4, 256);
save_ref(out_dir, 'analysis_core_psd', 'synthetic', x, AvgPsdInRange(x, 100, [5 20]), ...
    AvgPsdInRange(x, 100, [5 20], true), M, AvgPsdInRange(M, 50, [1 10], false, 2), ...
    AvgPsdInRange(M', 50, [1 10], true));
end

function ref_envelope(out_dir)
% out1 = s (column), out2 = extract_pulse_envelope(s, 'dt', 4)
amps = [1 2 3 0 0 0 7 8 5 4];
s = zeros(7 + 20*numel(amps) + 10, 1);
for k = 1:numel(amps)
    s(7 + (k-1)*20 + (1:6)) = amps(k);
end
s = s + 0.001*randn(size(s));
s(100) = 5;
save_ref(out_dir, 'analysis_core_envelope', 'synthetic', s, extract_pulse_envelope(s, 'dt', 4));
end

function ref_vis_speed(out_dir)
% out1 = pd (row), out2 = check_vis_speed(pd)
pd = repmat([zeros(1, 80) ones(1, 120)], 1, 8) + 0.05*randn(1, 1600);
save_ref(out_dir, 'analysis_core_vis_speed', 'synthetic', pd, check_vis_speed(pd));
end

function ref_library(out_dir)
% out1 = b, out2 = expmat(b, 3), out3 = X (with NaN), out4 = nan_sem(X), out5 = nan_sem(X, 2)
b = [1 2 3; 4 5 6];
X = randn(5, 4); X(2, 3) = NaN; X(4, :) = NaN; X(5, 1:3) = NaN;
save_ref(out_dir, 'analysis_core_library', 'synthetic', b, expmat(b, 3), X, nan_sem(X), nan_sem(X, 2));
end

function ref_behavior(out_dir)
% out1 = velM, out2 = velR, out3 = VRangle, out4 = LeinweberPerformance(velM, velR, VRangle),
% out5 = LeinweberPerformance(velM, velR, [0 VRangle]),
% out6 = frame_times, out7 = iframe_times, out8..out12 = map_eye_data_to_frame_times outputs
velM = abs(randn(1, 200)); velR = 0.3*randn(1, 200); VRangle = 5*rand(1, 200);
velM(10) = NaN;
ft = sort(rand(1, 50))*10; it = (0:0.33:10.5);
pd = randn(1, numel(it)); pp = randn(2, numel(it)); bl = randn(1, numel(it));
fm = randn(numel(it), 1); wp = randn(2, numel(it));
[o1, o2, o3, o4, o5] = map_eye_data_to_frame_times(pd, pp, bl, fm, wp, ft, it);
save_ref(out_dir, 'analysis_core_behavior', 'synthetic', velM, velR, VRangle, ...
    LeinweberPerformance(velM, velR, VRangle), LeinweberPerformance(velM, velR, [0 VRangle]), ...
    ft, it, o1, o2, o3, o4, o5, pd, pp, bl, fm, wp);
end

function ref_find_saccades(out_dir)
% out1 = idata (uint8, 100 x 120 x 60, bright pupil that jumps at frame 30 and a corner
% that brightens at frame 41 and darkens at frame 51), out2..out6 = find_saccades outputs
[cc, rr] = meshgrid(1:120, 1:100);
idata = 50*ones(100, 120, 60, 'uint8');
for k = 1:60
    cr = 50 + 8*(k >= 30) + round(randn);
    fr = idata(:, :, k);
    fr((rr - cr).^2 + (cc - 60).^2 <= 15^2) = 200;
    idata(:, :, k) = fr;
end
idata(1:50, 1:50, 41:end) = idata(1:50, 1:50, 41:end) + 10;
idata(1:50, 1:50, 51:end) = idata(1:50, 1:50, 51:end) - 10;
[s, d, p, b, f] = find_saccades(idata);
save_ref(out_dir, 'analysis_core_find_saccades', 'synthetic', idata, s, d, p, b, f);
end

function ref_rois2image(out_dir)
% out1 = dim, out2 = ROI indices as a 3-element cell, out3 = ROIs2image(..., 'randcolor', 0),
% out4 = same with 'type','perim', out5 = with 'sel', [2 3] (1-based; length(sel) = 2, so
% only ROIs 1-2 are drawn), out6 = template, out7 = template overlay with 'sel', [2 1],
% out8 = 'sel', [2 1; 3 2; 1 1] (rows [ROI group], three groups)
dim = [20 24];
ROIs = struct('indices', {});
m = false(dim); m(2:5, 3:6) = true; ROIs(1).indices = find(m);
m = false(dim); m(10:14, 10:12) = true; m(12, 13) = true; ROIs(2).indices = find(m);
m = false(dim); m(16:19, 1:4) = true; ROIs(3).indices = find(m);
template = reshape(1:prod(dim), dim) + 5*randn(dim);
save_ref(out_dir, 'analysis_core_rois2image', 'synthetic', dim, {ROIs.indices}, ...
    ROIs2image(ROIs, dim, 'randcolor', 0), ROIs2image(ROIs, dim, 'randcolor', 0, 'type', 'perim'), ...
    ROIs2image(ROIs, dim, 'randcolor', 0, 'sel', [2 3]), template, ...
    ROIs2image(ROIs, dim, 'randcolor', 0, 'template', template, 'sel', [2 1]), ...
    ROIs2image(ROIs, dim, 'randcolor', 0, 'sel', [2 1; 3 2; 1 1]));
end

function ref_spec_traces(out_dir)
% out1 = data (3 x 400), out2 = normalised stacked data from specTraces(data, 'plot', 0)
data = cumsum(randn(3, 400), 2);
[~, out] = specTraces(data, 'plot', 0);
save_ref(out_dir, 'analysis_core_spec_traces', 'synthetic', data, out);
end
