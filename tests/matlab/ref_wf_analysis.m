function ref_wf_analysis(sample_dir, out_dir) %#ok<INUSL>
% REF_WF_ANALYSIS  MATLAB outputs of the 21_Analysis WF_* functions, for calliopy's tests.
%
%   ref_wf_analysis(sample_dir, out_dir)
%
%   Run in MATLAB with calliope r2003 on the path (Signal Processing and Statistics
%   toolboxes needed). Every block builds a small synthetic proj_meta, calls the ORIGINAL
%   MATLAB function and saves the inputs followed by the outputs with save_ref as
%   ref_wfa_<name>.mat (out1, out2, ... in the order given in the block; input_file is
%   'synthetic'). The inputs are saved too, because random numbers differ between MATLAB
%   and Python. tests/test_wf_analysis_ref.py loads these files. No lab data is needed, so
%   sample_dir is not used.
%
%   NOT covered here: WF_bandpass and WF_velocityField (they call set_lab_paths and, for
%   the velocity field, the lab's grid template; Python's band-pass filter is a documented
%   approximation of MATLAB's bandpass), WF_corrmatBoxplot, WF_seedcorr and
%   WF_plotROICountsByLocation (no outputs, figures only), WF_plotCorrChangeInteractive,
%   WF_maskROIs (interactive), WF_imageROIvelocityField (figure only, lab template).
%
%   A block that fails is reported and skipped; the others still run.

rng(1);
blocks = {@ref_avgpsd, @ref_corrmat, @ref_corrmat_matrix, @ref_findlow, @ref_lesion, ...
          @ref_trim, @ref_commodore, @ref_onsets, @ref_roi_filter};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
    close all
end
end

%% synthetic proj_meta (same ROI layout as tests/test_wf_analysis.py make_roiinfo) --------
function ROIinfo = synth_roiinfo()
acr = {'VISp', 'RSP', 'MOp', 'SSp'};
ROIinfo = struct('ML_index', {}, 'AP_index', {}, 'Allen_acronym', {}, 'name', {}, 'label', {});
k = 0;
for ap = 0:39
    for ml = -20:20
        if k == 818, break; end
        k = k + 1;
        ROIinfo(k).ML_index = ml;
        ROIinfo(k).AP_index = ap;
        ROIinfo(k).Allen_acronym = acr{min(floor(ap / 6), 3) + 1};
        if ml < 0
            ROIinfo(k).name = 'left hemisphere';
        else
            ROIinfo(k).name = 'right hemisphere';
        end
        ROIinfo(k).label = '';
    end
end
end

function pm = synth_pm(nsites, ntp, nbr)
ROIinfo = synth_roiinfo();
region = cellfun(@(a) find(strcmp(a, {'VISp', 'RSP', 'MOp', 'SSp'})), {ROIinfo.Allen_acronym});
T = sum(nbr);
for s = 1:nsites
    pm(s).ExpGroup = [s 1]; %#ok<AGROW>
    pm(s).projID = 'PRJ';
    pm(s).Acode = 'A1';
    pm(s).animal = sprintf('m%d', s);
    pm(s).uuid = 'abc';
    for t = 1:ntp
        common = randn(4, T);
        act = 5 + randn(818, T) + 0.7 * common(region, :);
        run = zeros(1, T);
        run(1:3:end) = 1;
        pm(s).rd(1, t).act = act;
        pm(s).rd(1, t).nbr_frames = nbr;
        pm(s).rd(1, t).comments = {'FB one', 'PB two'};
        pm(s).rd(1, t).session = {'FB one', 'PB two'};
        pm(s).rd(1, t).Running = run;
        pm(s).rd(1, t).Running_smoothed = run;
        pm(s).rd(1, t).ROIinfo = ROIinfo;
    end
end
end

%% blocks ---------------------------------------------------------------------------------
function ref_avgpsd(out_dir)
% out1 = x (3 x 200), out2 = x2 (3 x 201), out3 = AvgPsdInRange(x, 50, [2 7], false, 2),
% out4 = AvgPsdInRange(x2, 50, [2 7], false, 2), out5 = AvgPsdInRange(x, 50, [2 7], true, 2)
x = randn(3, 200); x2 = randn(3, 201);
save_ref(out_dir, 'wfa_avgpsd', 'synthetic', x, x2, AvgPsdInRange(x, 50, [2 7], false, 2), ...
    AvgPsdInRange(x2, 50, [2 7], false, 2), AvgPsdInRange(x, 50, [2 7], true, 2));
end

function ref_corrmat(out_dir)
% out1 = proj_meta, out2 = WF_corrmat default, out3 = 'comment','PB','tp_list',1,
% out4 = 'medianDiv_on',true, out5 = 'runningState','running',
% out6 = normalized_data of WF_corrmat_plot(out2(:,:,:,1), proj_meta)
pm = synth_pm(2, 2, [60 40]);
c2 = WF_corrmat(pm, 'plot_on', false);
c3 = WF_corrmat(pm, 'plot_on', false, 'comment', 'PB', 'tp_list', 1);
c4 = WF_corrmat(pm, 'plot_on', false, 'medianDiv_on', true);
c5 = WF_corrmat(pm, 'plot_on', false, 'runningState', 'running');
[~, nd] = WF_corrmat_plot(c2(:, :, :, 1), pm);
save_ref(out_dir, 'wfa_corrmat', 'synthetic', pm, c2, c3, c4, c5, nd);
end

function ref_corrmat_matrix(out_dir)
% out1 = proj_meta, out2 = WF_corrmat_matrix default (organisation 1),
% out3 = 'hemisphere_organization', 2, out4 = 'separate_hemispheres', false
pm = synth_pm(2, 1, [80]);
m1 = WF_corrmat_matrix(pm, 'plot_on', false);
m2 = WF_corrmat_matrix(pm, 'plot_on', false, 'hemisphere_organization', 2);
m3 = WF_corrmat_matrix(pm, 'plot_on', false, 'separate_hemispheres', false);
save_ref(out_dir, 'wfa_corrmat_matrix', 'synthetic', pm, m1, m2, m3);
end

function ref_findlow(out_dir)
% out1 = corrmat_all, out2 = proj_meta, out3 = idx_counts_sort_all (cell 1x1xnSites,
% 1-based ROIs) of WF_findLowCorrcoef(corrmat_all, proj_meta, 'CC_range', [0 0.05])
pm = synth_pm(2, 1, [60]);
c = WF_corrmat(pm, 'plot_on', false);
idx = WF_findLowCorrcoef(c, pm, 'CC_range', [0 0.05]);
save_ref(out_dir, 'wfa_findlow', 'synthetic', c, pm, idx);
end

function ref_lesion(out_dir)
% out1 = proj_meta, out2 = lesioning_scores of WF_LesioningROIs(pm, 'ShowAtlasBorders', false)
pm = synth_pm(2, 2, [50]);
scores = WF_LesioningROIs(pm, 'ShowAtlasBorders', false);
save_ref(out_dir, 'wfa_lesion', 'synthetic', pm, scores);
end

function ref_trim(out_dir)
% out1 = proj_meta, out2 = WF_trimStackFrames(proj_meta, 5)
pm = synth_pm(1, 2, [30 8]);
save_ref(out_dir, 'wfa_trim', 'synthetic', pm, WF_trimStackFrames(pm, 5));
end

function ref_commodore(out_dir)
% out1 = proj_meta, out2 = WF_commodore (cell), FrameRate 10, AnalysisWindow 10,
% out3 = the same with 'AverageOverPsd', false
pm = synth_pm(2, 1, [1000 900]);
a = WF_commodore(pm, 'FrameRate', 10, 'AnalysisWindow', 10, 'ShowFigure', false, ...
    'ShowAtlasBorders', false);
b = WF_commodore(pm, 'FrameRate', 10, 'AnalysisWindow', 10, 'ShowFigure', false, ...
    'ShowAtlasBorders', false, 'AverageOverPsd', false);
save_ref(out_dir, 'wfa_commodore', 'synthetic', pm, a, b);
end

function ref_onsets(out_dir)
% out1 = proj_meta (with StimulusID), out2 = WF_onsets(pm, 'gratings', 'NoPlot', true,
% 'Window', [-10 60], 'BaselineWindow', [-30 -20])
pm = synth_pm(2, 1, [600 600]);
for s = 1:2
    stim = zeros(1, 1200);
    for f = [300 450 620 900]
        stim(f:f + 19) = 1;
        pm(s).rd(1, 1).act(1:100, f:f + 49) = pm(s).rd(1, 1).act(1:100, f:f + 49) + 2;
    end
    pm(s).rd(1, 1).StimulusID = stim;
end
snps = WF_onsets(pm, 'gratings', 'NoPlot', true, 'Window', [-10 60], ...
    'BaselineWindow', [-30 -20]);
save_ref(out_dir, 'wfa_onsets', 'synthetic', pm, snps);
end

function ref_roi_filter(out_dir)
% out1 = proj_meta, out2 = corrmat_all_filtered of WF_corrmat_ROI_filter(pm,
% 'exclude_ROIs', {'RSP'}, 'plot_on', false)
pm = synth_pm(1, 1, [60]);
f = WF_corrmat_ROI_filter(pm, 'exclude_ROIs', {'RSP'}, 'plot_on', false);
save_ref(out_dir, 'wfa_roi_filter', 'synthetic', pm, f);
end
