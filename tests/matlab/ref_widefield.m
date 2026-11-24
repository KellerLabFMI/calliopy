function ref_widefield(sample_dir, out_dir) %#ok<INUSL>
% REF_WIDEFIELD  MATLAB outputs of the widefield functions, for calliopy's tests.
%
%   ref_widefield(sample_dir, out_dir)
%
%   Run in MATLAB with calliope on the path (Image Processing Toolbox needed). Every block
%   builds a small input, calls the ORIGINAL MATLAB function and saves the inputs followed by
%   the outputs with save_ref as ref_wf_<name>.mat (out1, out2, ... in the order given in the
%   block; input_file is 'synthetic'). The inputs are saved too, because random numbers differ between MATLAB and
%   Python. tests/test_widefield_ref.py loads these files. No lab data is needed, so
%   sample_dir is not used.
%
%   NOT covered here, because they need ExpLog, adata files or are local functions:
%     calc_widefield_act, wid_adata_check, register_wid (and its local compute_tform),
%     propagate_tform_and_ROIs_from_expid, wid_show_stack_drift, online_retinotopy,
%     UV_img_to_linear / UV_linear_to_img / WF_grid2vid / quiver_video (need the grid template),
%     the ROItemplate_generate_* functions (compared with the shipped .mat templates instead).
%   Python's image_similarity_snippets and registration intentionally differ from MATLAB
%   (see their docstrings) and have no reference here.
%
%   A block that fails is reported and skipped; the others still run.

rng(1);
blocks = {@ref_builtins, @ref_transforms, @ref_frames, @ref_filters, @ref_flow, @ref_activity};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
end
end

%% MATLAB built-ins that calliopy._matlab reproduces -------------------------------------------
function ref_builtins(out_dir)
% round: out1 = x, out2 = round(x)
x = [0.5 1.5 2.5 -0.5 -2.5 2.4999 0.49999999999999994 -1.5 3 NaN Inf];
save_ref(out_dir, 'wf_round', 'synthetic', x, round(x));

% prctile (the copy on the calliope path): out1 = x, out2 = p, out3 = prctile(x(:), p),
% out4 = prctile(x, [10 90], 2)
x = randn(7, 5); x(3, 2) = NaN;
p = [0 5 25 50 95 100];
save_ref(out_dir, 'wf_prctile', which('prctile'), x, p, prctile(x(:), p), prctile(x, [10 90], 2));

% corr2: out1 = a, out2 = b, out3 = corr2(a, b)
a = magic(6); b = a' + (1:6)';
save_ref(out_dir, 'wf_corr2', 'synthetic', a, b, corr2(a, b));

% conv2: out1 = a, out2 = k2, out3 = k3, out4..7 = full/same/valid with k2, same with k3
a = magic(5); k2 = [1 2; 3 4]; k3 = reshape(1:9, 3, 3);
save_ref(out_dir, 'wf_conv2', 'synthetic', a, k2, k3, conv2(a, k2, 'full'), conv2(a, k2, 'same'), ...
    conv2(a, k2, 'valid'), conv2(a, k3, 'same'));

% filter2: out1 = a, out2 = k2, out3 = k3, out4 = filter2(k2, a), out5 = filter2(k3, a, 'valid'),
% out6 = filter2(ones(4)/16, a)
a = magic(6);
save_ref(out_dir, 'wf_filter2', 'synthetic', a, k2, k3, filter2(k2, a), filter2(k3, a, 'valid'), ...
    filter2(ones(4)/16, a));

% imgaussfilt: out1 = a, out2 = imgaussfilt(a, 1.3), out3 = imgaussfilt(a, [0.8 2.2]),
% out4 = imgaussfilt(uint16(a*100), 1)
a = magic(12);
save_ref(out_dir, 'wf_imgaussfilt', 'synthetic', a, imgaussfilt(a, 1.3), imgaussfilt(a, [0.8 2.2]), ...
    imgaussfilt(uint16(a*100), 1));

% normxcorr2: out1 = a, out2 = t, out3 = normxcorr2(t, a)
a = magic(9) + (1:9); t = a(3:5, 4:7);
save_ref(out_dir, 'wf_normxcorr2', 'synthetic', a, t, normxcorr2(t, a));

% linspace: out1 = linspace(0, 1, 10), out2 = linspace(-3.3, 7.9, 13)
save_ref(out_dir, 'wf_linspace', 'synthetic', linspace(0, 1, 10), linspace(-3.3, 7.9, 13));
end

%% transforms (python: calliopy.widefield similarity_matrix, fit_similarity, warp_image) ----------
function ref_transforms(out_dir)
% simtform2d: out1 = [scale angle tx ty], out2 = tform.A
tf = simtform2d(1.3, -25, [40 -7.5]);
save_ref(out_dir, 'wf_simtform2d', 'synthetic', [1.3 -25 40 -7.5], tf.A);

% fitgeotform2d similarity: out1 = moving (2 points), out2 = fixed, out3 = A,
% out4 = moving (5 points), out5 = fixed, out6 = A
m2 = [50 250; 450 250]; f2 = [750 1200; 750 200];
tf2 = fitgeotform2d(m2, f2, 'similarity');
m5 = [10 20; 50 25; 30 80; 90 60; 70 10];
f5 = m5*[0.9 0.2; -0.2 0.9] + [3 9] + [0.1 -0.2; 0.3 0; -0.1 0.2; 0 0.1; 0.2 -0.3];
tf5 = fitgeotform2d(m5, f5, 'similarity');
save_ref(out_dir, 'wf_fitgeotform2d', 'synthetic', m2, f2, tf2.A, m5, f5, tf5.A);

% transformPointsForward: out1 = A, out2 = points, out3 = transformed points
tf = simtform2d(1.5, 90, [10 20]); pts = [1 0; 0 2; 3 4];
save_ref(out_dir, 'wf_transform_points', 'synthetic', tf.A, pts, transformPointsForward(tf, pts));

% imwarp with 'OutputView': out1 = img, out2 = A1, out3 = warp to [7 9] (integer shift),
% out4 = A2, out5 = linear warp to [8 10], out6 = nearest warp to [8 10],
% out7 = linear warp of uint16(img) to [7 9]
img = reshape(1:63, 7, 9);
tf1 = simtform2d(1, 0, [2 1]); tf2 = simtform2d(1.2, 10, [1.5 0.5]);
save_ref(out_dir, 'wf_imwarp', 'synthetic', img, tf1.A, imwarp(img, tf1, 'OutputView', imref2d([7 9])), ...
    tf2.A, imwarp(img, tf2, 'OutputView', imref2d([8 10])), ...
    imwarp(img, tf2, 'nearest', 'OutputView', imref2d([8 10])), ...
    imwarp(uint16(img), tf2, 'OutputView', imref2d([7 9])));

% imwarp of a label image with constant regions: out1 = lab, out2 = A, out3 = warped
lab = 182305689*ones(30, 40); lab(:, 21:end) = 182305701;
tf = simtform2d(1.07, 17, [2.3 -1.1]);
save_ref(out_dir, 'wf_imwarp_labels', 'synthetic', lab, tf.A, imwarp(lab, tf, 'OutputView', imref2d([30 40])));
end

%% frame shifts (05_LoadData: fftreg, shift_data, register_frames) --------------------------------
function ref_frames(out_dir)
% fftreg: out1 = img, out2 = applied [dr dc] per row, out3 = returned [row_shift col_shift] per row
img = randn(10, 12);
d = [2 -3; -4 5; 4 7; 5 6; 6 1];
res = zeros(size(d));
for k = 1:size(d, 1)
    [res(k, 1), res(k, 2)] = fftreg(fft2(circshift(img, d(k, :))), fft2(img));
end
save_ref(out_dir, 'wf_fftreg', 'synthetic', img, d, res);

% shift_data: out1 = stack, out2 = shift_data(stack, [1 0], [0 -1]),
% out3 = shift_data(uint16(stack), 0.5, -0.5)
a = reshape(1:12, 4, 3)';
s = cat(3, a, a + 100);
save_ref(out_dir, 'wf_shift_data', 'synthetic', s, shift_data(s, [1 0], [0 -1]), shift_data(uint16(s), 0.5, -0.5));

% register_frames: out1 = stack (double), out2 = applied offsets [row col] per frame,
% out3/4 = [shift_x shift_y] with a double template, out5/6 = with uint16 stack and template
% (as register_wid calls it), out7/8 = default template (mean over frames)
scene = imgaussfilt(randn(200, 220), 2)*1000 + 5000;
offsets = repmat([0 0; 3 -2; -5 4; 7 7; -1 -6; 2 0], 12, 1);
st = zeros(120, 130, size(offsets, 1));
for k = 1:size(offsets, 1)
    st(:, :, k) = scene(41-offsets(k, 1):160-offsets(k, 1), 46-offsets(k, 2):175-offsets(k, 2));
end
[sx1, sy1] = register_frames(st, st(:, :, 1));
[sx2, sy2] = register_frames(uint16(st), uint16(st(:, :, 1)));
[sx3, sy3] = register_frames(st);
save_ref(out_dir, 'wf_register_frames', 'synthetic', st, offsets, sx1, sy1, sx2, sy2, sx3, sy3);
end

%% FFT filters --------------------------------------------------------------------------------
function ref_filters(out_dir)
% zeroShiftBandPass: out1 = sig, out2 = [Fs lb ub], out3 = filtered; three more cases follow
% as out4..6, out7..9, out10..12 (lb beyond Nyquist; even length near Nyquist; ub beyond Nyquist)
s1 = randn(3, 201); s2 = randn(2, 64); s3 = randn(1, 64); s4 = randn(2, 65);
save_ref(out_dir, 'wf_zeroShiftBandPass', 'synthetic', ...
    s1, [100 5 20], zeroShiftBandPass(s1, 100, 5, 20), ...
    s2, [64 40 50], zeroShiftBandPass(s2, 64, 40, 50), ...
    s3, [64 31.5 50], zeroShiftBandPass(s3, 64, 31.5, 50), ...
    s4, [64 3 100], zeroShiftBandPass(s4, 64, 3, 100));

% zeroShiftNotch: out1 = sig, out2 = [Fs nf bw], out3 = filtered; two more cases as out4..6
% (upper edge beyond Nyquist) and out7..9 (odd length)
s1 = randn(2, 1000); s2 = randn(2, 100); s3 = randn(2, 999);
save_ref(out_dir, 'wf_zeroShiftNotch', 'synthetic', ...
    s1, [1000 50 2], zeroShiftNotch(s1, 1000, 50, 2), ...
    s2, [100 49 4], zeroShiftNotch(s2, 100, 49, 4), ...
    s3, [1000 50 2], zeroShiftNotch(s3, 1000, 50, 2));
end

%% velocity field -------------------------------------------------------------------------------
function ref_flow(out_dir)
% calculate_velocity_field_Horn_Schunck: out1 = mat_vid, out2 = [Alpha MaxIter Sigma],
% out3 = U, out4 = V
mat_vid = rand(16, 18, 5); mat_vid(3, 4, 2) = NaN;
[U, V] = calculate_velocity_field_Horn_Schunck(mat_vid, 'Alpha', 0.01, 'MaxIter', 50, 'Sigma', 1);
save_ref(out_dir, 'wf_velocity_field', 'synthetic', mat_vid, [0.01 50 1], U, V);
end

%% activity helpers -----------------------------------------------------------------------------
function ref_activity(out_dir)
% get_frame_times_widefield: out1 = trig_ch, out2 = shutter_ch, out3 = sample rate,
% out4 = frame timestamps as datevec rows [y m d H M S] (the stack crosses midnight),
% out5 = frame_times, out6 = frame_times_timestamp
fs = 10000; n = 3*fs; t = 0:n-1;
trig = 5*double(mod(t, 100) < 50);
shutter = zeros(1, n); shutter(2000:28000) = 5;
nfr = 261;   % frames inside the shutter period
stamps = datetime(2024, 3, 5, 23, 59, 58.5) + seconds((0:nfr-1)*0.01);
meta_info = struct('timestamp', num2cell(stamps));
[frame_times, frame_times_timestamp] = get_frame_times_widefield(trig, shutter, meta_info, fs);
save_ref(out_dir, 'wf_get_frame_times_widefield', 'synthetic', trig, shutter, fs, datevec(stamps), ...
    frame_times, frame_times_timestamp);

% median_divide_video: out1 = data (uint16), out2 = data_divided
data = uint16(randi(4000, 6, 7, 9));
save_ref(out_dir, 'wf_median_divide_video', 'synthetic', data, median_divide_video(data));

% angles_from_top_left: out1 = [row col] indices, out2 = matrixSize, out3 = MPOINT, out4 = thetaDeg
idxs = [3 4; 2 3; 3 2; 4 3; 2 4; 4 2];
save_ref(out_dir, 'wf_angles_from_top_left', 'synthetic', idxs, [5 5], [3 3], angles_from_top_left(idxs, [5 5], [3 3]));
end
