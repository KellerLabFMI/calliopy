function ref_twophoton(sample_dir, out_dir) %#ok<INUSL>
% REF_TWOPHOTON  MATLAB outputs of the deterministic 2P / EEG / OII functions for calliopy's
% tests (calliopy.twophoton, calliopy.eeg, calliopy.oii).
%
%   ref_twophoton(sample_dir, out_dir)
%
%   Every block builds a small synthetic input, calls the ORIGINAL calliope r2003 function
%   (or the MATLAB built-in the port reproduces) and saves inputs followed by outputs with
%   save_ref as ref_twophoton_<name>.mat (out1, out2, ... in the order given in the block;
%   input_file is 'synthetic'). tests/test_twophoton_ref.py loads these files.
%   sample_dir is not used.
%
%   NOT covered (need ExpLog, adata files, dialogs or a viewer): calc_act, register_bin,
%   register_exp_downsample/avgplanes/fixoutliers, correct_dxdy, tpi_adata_check,
%   SpineQuest callbacks (local functions), analyzeOII_Fourier, AnalyzeWaveStimulusIOSseparate,
%   EEG_check_*, online_response, compareStacks.
%   Needs the Image Processing Toolbox (impyramid, fspecial, imfilter, mat2gray), the Curve
%   Fitting Toolbox (smooth) and R2019a+ (matchpairs); blocks that fail are reported and skipped.

rng(1);
blocks = {@ref_roi_activity, @ref_calc_act_map, @ref_fix_registration, ...
    @ref_register_multilayer, @ref_get_eeg_times, @ref_erp, @ref_resample, @ref_welch, ...
    @ref_iframe_times, @ref_fourier_maps, @ref_builtins, @ref_np_activity};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
end
close all
end

function ref_roi_activity(out_dir)
% out1 = data (int16 6x5x7), out2 = indices ROI 1, out3 = indices ROI 2,
% out4 = [ROIs.activity] (7 x 2)
data = int16(randi(1000, 6, 5, 7));
ROIs(1).indices = [1; 2; 12];
ROIs(2).indices = [30; 7];
out = ROI_activity(data, ROIs);
save_ref(out_dir, 'twophoton_roi_activity', 'synthetic', data, ROIs(1).indices, ...
    ROIs(2).indices, [out.activity]);
end

function ref_np_activity(out_dir)
% out1 = data, out2 = np.indices, out3 = np.indices_middle, out4 = activity,
% out5 = activity_middle
data = int16(randi(1000, 6, 5, 7));
np.indices = (1:10)'; np.indices_middle = [3; 4; 25];
out = np_activity(data, np);
save_ref(out_dir, 'twophoton_np_activity', 'synthetic', data, np.indices, ...
    np.indices_middle, out.activity, out.activity_middle);
end

function ref_calc_act_map(out_dir)
% out1 = data (int16 6x5x37), out2 = calc_act_map(data), out3 = calc_act_map(data, 4)
data = int16(randi(500, 6, 5, 37));
save_ref(out_dir, 'twophoton_calc_act_map', 'synthetic', data, calc_act_map(data), ...
    calc_act_map(data, 4));
end

function ref_fix_registration(out_dir)
% out1 = data (100x100x4 double), out2 = template, out3 = shift_x in, out4 = shift_y in,
% out5 = shift_x out, out6 = shift_y out
base = conv2(randn(130), ones(5)/25, 'same') * 100 + 1000;
t = base(16:115, 16:115);
data = cat(3, t, base(13:112, 18:117), base(13:112, 18:117), base(15:114, 15:114));
sx = [0; 150; 150; 1]; sy = [0; 0; 0; 1];
[ox, oy] = fix_registration(data, t, sx, sy);
save_ref(out_dir, 'twophoton_fix_registration', 'synthetic', data, t, sx, sy, ox, oy);
end

function ref_register_multilayer(out_dir)
% out1 = data (int16 60x64x(8*30)), out2 = 30, out3 = mean_data, out4 = dx_corr,
% out5 = dy_corr
base = conv2(randn(80), ones(5)/25, 'same') * 300 + 2000;
data = zeros(60, 64, 240, 'int16');
for z = 1:8
    for f = 1:30
        data(:, :, (z-1)*30+f) = int16(base(10+mod(z,3):69+mod(z,3), 8:71) + randn(60, 64));
    end
end
[m, dx, dy] = register_multilayer(data, 30);
save_ref(out_dir, 'twophoton_register_multilayer', 'synthetic', data, 30, m, dx, dy);
end

function ref_get_eeg_times(out_dir)
% out1 = aux sync, out2 = eeg sync, out3 = eeg_times
aux = zeros(1, 160000); eeg = zeros(1, 15500);
for k = 0:14
    aux(5001+10000*k+(0:99)) = 5;
    eeg(301+1000*k+3+(0:9)) = 5;
end
save_ref(out_dir, 'twophoton_get_eeg_times', 'synthetic', aux, eeg, get_eeg_times(aux, eeg));
end

function ref_erp(out_dir)
% out1 = eegData (3 x 4000), out2 = eegTimes, out3 = triggerTimes, out4 = erp,
% out5 = time, out6 = epochsIds
eegData = randn(3, 4000);
eegTimes = 10 * (1:4000) + 3;
trig = [5000 12345 30010];
[erp, time, ids] = create_erp_auxaligned(eegData, eegTimes, trig, 1000, 0.2, 0.5);
save_ref(out_dir, 'twophoton_erp', 'synthetic', eegData, eegTimes, trig, erp, time, ids);
end

function ref_resample(out_dir)
% out1 = x (3 x 300), out2 = myResample(x, 1, 2, 2), out3 = v (column, 100),
% out4 = myResample(v, 2, 3)
x = randn(3, 300); v = randn(100, 1);
save_ref(out_dir, 'twophoton_resample', 'synthetic', x, myResample(x, 1, 2, 2), v, ...
    myResample(v, 2, 3));
end

function ref_welch(out_dir)
% out1 = x, out2 = Pxx, out3 = f  (window 128, noverlap 64, nfft 128, fs 1000),
% out4 = Pxx with nfft 101
x = randn(1, 2000);
[p, f] = myWelch(x, 128, 64, 128, 1000);
p2 = myWelch(x, 128, 64, 101, 1000);
save_ref(out_dir, 'twophoton_welch', 'synthetic', x, p, f, p2);
end

function ref_iframe_times(out_dir)
% out1 = imeta (4x5), out2 = shutter, out3 = nbr_iframes, out4 = iframe_times,
% out5 = shutter_open
sh = zeros(1, 200); sh(21:80) = 5; sh(121:190) = 5;
im = zeros(4, 5); im(2, :) = [100 110 120 500 530];
[it, so] = get_iframe_times(im, sh, [3 2]);
save_ref(out_dir, 'twophoton_iframe_times', 'synthetic', im, sh, [3 2], it, so);
end

function ref_fourier_maps(out_dir)
% out1..out4 = the four complex spectra (12 x 14), out5 = meandat_mask (logical),
% out6 = Power, out7 = PhaseHor, out8 = PhaseVer, out9 = PhaseMapHor, out10 = Phase{1}
F = cell(1, 4);
for k = 1:4
    F{k} = (1 + rand(12, 14)) .* exp(1i * (2*pi*rand(12, 14) - pi));
end
mask = true(12, 14); mask(2, 3) = false;
[Power, Phase, ~, PhaseMapHor, ~, PhaseHor, PhaseVer] = showFourierMaps(F, 1, ones(12, 14), mask);
save_ref(out_dir, 'twophoton_fourier_maps', 'synthetic', F{1}, F{2}, F{3}, F{4}, mask, ...
    Power, PhaseHor, PhaseVer, PhaseMapHor, Phase{1});
end

function ref_builtins(out_dir)
% out1 = A (double 13x10), out2 = impyramid(A, 'reduce'), out3 = U (uint16 13x10),
% out4 = impyramid(U, 'reduce'), out5 = fspecial('gaussian', 15, 2),
% out6 = imfilter(U, out5), out7 = mat2gray(A), out8 = y (column 60),
% out9 = smooth(y, 81), out10 = smooth(y, 6), out11 = D (2x2x9), out12 = movmean(D, 4, 3),
% out13 = C (5x6 cost), out14 = matchpairs(C, 0.6)
A = randn(13, 10); U = uint16(randi(60000, 13, 10));
h = fspecial('gaussian', 15, 2);
y = cumsum(randn(60, 1)); D = randn(2, 2, 9); C = rand(5, 6);
save_ref(out_dir, 'twophoton_builtins', 'synthetic', A, impyramid(A, 'reduce'), U, ...
    impyramid(U, 'reduce'), h, imfilter(U, h), mat2gray(A), y, smooth(y, 81), smooth(y, 6), ...
    D, movmean(D, 4, 3), C, matchpairs(C, 0.6));
end
