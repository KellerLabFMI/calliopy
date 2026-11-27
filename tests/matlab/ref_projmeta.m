function ref_projmeta(sample_dir, out_dir) %#ok<INUSL>
% REF_PROJMETA  MATLAB outputs of the deterministic Meta/ExpLog functions (24_ProjMeta,
% 20_MetaData, 02_ExpLog, 06_Library/get_stackid, 21_Analysis/get_vel_ind_from_adata,
% get_lick_times_thrsh, get_smoothed_ROIs) for calliopy's tests.
%
%   ref_projmeta(sample_dir, out_dir)
%
%   Every block builds a small synthetic input, calls the ORIGINAL calliope function and
%   saves the inputs followed by the outputs with save_ref as ref_projmeta_<name>.mat
%   (out1, out2, ... in the order given in the block; input_file is 'synthetic').
%   tests/test_projmeta_ref.py loads these files. sample_dir is not used; no database
%   connection is needed (the ExpLog is synthetic and passed in).
%
%   NOT covered: generate_proj_meta, hEEG_proj_meta, generateMetaInfo (need lab paths,
%   project defaults, Excel files and interactive input), TP_checkPmSites and the cell
%   browser (graphics), pm_info (prints only).

blocks = {@ref_stackid, @ref_remove_html, @ref_pulsed, @ref_lick, @ref_extend_trig, ...
    @ref_vel, @ref_vr_vel, @ref_act2mat, @ref_pm_expids, @ref_qexp, @ref_smoothed_rois, ...
    @ref_preprocess_wf, @ref_aux_method};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
end
end

function ref_stackid(out_dir)
% out1 = get_stackid({'S1-T1234.lvd','nothing','s1-p77_ch525.bin','S1-T0123.lvd'})
save_ref(out_dir, 'projmeta_stackid', 'synthetic', ...
    get_stackid({'S1-T1234.lvd', 'nothing', 's1-p77_ch525.bin', 'S1-T0123.lvd'}));
end

function ref_remove_html(out_dir)
% out1 = removeHTML('a <b>bold</b> <i x=''1''>it</i>')
save_ref(out_dir, 'projmeta_remove_html', 'synthetic', removeHTML('a <b>bold</b> <i x=''1''>it</i>'));
end

function ref_pulsed(out_dir)
% out1 = x (row), out2 = pulsed_to_continuous(x, 1000, 2)
x = zeros(1, 10000);
for s = [1000 1050 1100 6000]
    x(s+1:s+11) = 5;
end
x(7000:7003) = 3;
save_ref(out_dir, 'projmeta_pulsed', 'synthetic', x, pulsed_to_continuous(x, 1000, 2));
end

function ref_lick(out_dir)
% out1 = x, out2 = get_lick_times_thrsh(x)
x = zeros(1, 500); x([10:12 100 300:305]) = 4; x(400) = 0.5;
save_ref(out_dir, 'projmeta_lick', 'synthetic', x, get_lick_times_thrsh(x));
end

function ref_extend_trig(out_dir)
% out1 = x, out2 = assign_processed_aux(x, {'extend_trig_to_200ms'}, 'Reward'){1,2}
x = zeros(1, 2000); x(101:105) = 1; x(900:1000) = 2; x(1950:1960) = 1;
p = assign_processed_aux(x, {'extend_trig_to_200ms'}, 'Reward');
save_ref(out_dir, 'projmeta_extend_trig', 'synthetic', x, p{1, 2});
end

function ref_vel(out_dir)
% out1 = x (encoder saw tooth, row), out2..out5 = [vel, vel_ind, vel_smoothed, vel_raw]
x = mod((1:3000) * 0.013, 10) - 5;
[vel, vel_ind, vel_smoothed, vel_raw] = get_vel_ind_from_adata(x);
save_ref(out_dir, 'projmeta_vel', 'synthetic', x, vel, double(vel_ind), vel_smoothed, vel_raw);
end

function ref_vr_vel(out_dir)
% out1 = x, out2 = assign_processed_aux(x, {'VR_pos_to_vel_smoothed'}, 'VRx'){1,2}
x = mod((1:3000) * 0.004, 5);
p = assign_processed_aux(x, {'VR_pos_to_vel_smoothed'}, 'VRx');
save_ref(out_dir, 'projmeta_vr_vel', 'synthetic', x, p{1, 2});
end

function ref_act2mat(out_dir)
% out1 = act layer 1 (2 x 50), out2 = act layer 2 (3 x 50), out3..out5 = act2mat outputs
% [actmat, indROIinlayer, indlayer], out6..out7 = trig2snps(pm, 1, 1, [10 30 48 26], [-3 3], {'speed'})
a1 = reshape(1:100, 2, 50); a2 = reshape(sin(1:150), 3, 50);
pm = struct;
pm(1).rd(1, 1).act = a1; pm(1).rd(2, 1).act = a2;
pm(1).rd(1, 1).nbr_frames = [25 25];
pm(1).rd(1, 1).frame_times = 1:50;
pm(1).rd(1, 1).speed = 2 * (1:50);
[actmat, i1, i2] = act2mat(pm, 1, 1);
[snps, arr] = trig2snps(pm, 1, 1, [10 30 48 26], [-3 3], {'speed'});
save_ref(out_dir, 'projmeta_act2mat', 'synthetic', a1, a2, actmat, i1, i2, snps, arr{1});
end

function ref_pm_expids(out_dir)
% out1 = pm_expids(pm), out2 = pm_expids(pm, [], [], 1), out3 = pm_expids(pm, [], 1)
pm = struct;
pm(1).ExpGroup = [10 11]; pm(1).rd(1, 2).act = [];
pm(2).ExpGroup = 20; pm(2).rd(1, 1).act = [];
save_ref(out_dir, 'projmeta_pm_expids', 'synthetic', pm_expids(pm), pm_expids(pm, [], [], 1), ...
    pm_expids(pm, [], 1));
end

function EL = synthetic_explog()
EL.stackid = {100; 101; 102; 103; 104; 105};
EL.expid = {100; 100; 102; 103; 104; 105};
EL.comment = {'day1 <b>gratings</b>'; 'day1 dark'; ['day2 ' repmat('x', 1, 80)]; 'ori omission'; 'fb'; 'pb'};
EL.stackdate = {'05-Jan-2024'; '05-Jan-2024'; '07-Jan-2024'; '10-Jan-2024'; '11-Jan-2024'; '12-Jan-2024'};
EL.datadeleted = {0; 0; 0; 0; 0; 1};
EL.siteid = {100; 100; 100; 103; 104; 103};
EL.analysiscode = {'1'; '1'; '1'; '2'; '1'; '1'};
EL.animalid = {'AA_240101_001'; 'AA_240101_001'; 'AA_240101_001'; 'AA_240101_002'; 'AA_240101_003'; ...
    'AA_240101_002'};
EL.project = {'PRJ'; 'PRJ'; 'PRJ'; 'PRJ'; 'OTH'; 'PRJ'};
EL.location = {'L2/3'; 'L2/3'; 'L2/3'; 'L5'; 'L2/3'; 'L5'};
EL.pi = {'hans'; 'hans'; 'hans'; 'hans'; 'greta'; 'hans'};
EL.strain = {'C57'; 'C57'; 'C57'; 'C57'; 'C57'; 'C57'};
end

function ref_qexp(out_dir)
% out1 = ExpLog (struct of cell columns), out2..out7 = qexp results:
% ('pi','hans','expid'), ('regex','day1|day2','stackid'), ('acode','2','stackid'),
% ('pi','hans','last',2,'stackid'), ('animalid','/002/','stackid'), ('isexp',0,'stackid');
% out8 = eid('hans', EL), out9 = eid(103, EL), out10 = eid('101', EL)
EL = synthetic_explog();
save_ref(out_dir, 'projmeta_qexp', 'synthetic', EL, ...
    qexp('pi', 'hans', 'ExpLog', EL, 'expid'), qexp('regex', 'day1|day2', 'ExpLog', EL, 'stackid'), ...
    qexp('acode', '2', 'ExpLog', EL, 'stackid'), qexp('pi', 'hans', 'last', 2, 'ExpLog', EL, 'stackid'), ...
    qexp('animalid', '/002/', 'ExpLog', EL, 'stackid'), qexp('isexp', 0, 'ExpLog', EL, 'stackid'), ...
    eid('hans', EL), eid(103, EL), eid('101', EL));
end

function ref_smoothed_rois(out_dir)
% out1 = activity of ROI 1..3 as columns (500 x 3), out2..out3 = [act, ROIinfo.F0 as a row]
t = (1:500)';
A = [5 + sin(t/10) + t/200, 3 + cos(t/7), 4 + 0.5 * sin(t/3) .* (t > 250)];
ROIs = struct('indices', {[1 2], [3 4], [5 6]}, 'activity', {A(:, 1), A(:, 2), A(:, 3)});
[act, info] = get_smoothed_ROIs(ROIs, 0);
save_ref(out_dir, 'projmeta_smoothed_rois', 'synthetic', A, act, [info.F0]);
end

function ref_preprocess_wf(out_dir)
% out1 = activity (14000 x 2), out2 = preprocess_widefield_activity_for_proj_meta(ROIs, [7000 7000], '')
t = (1:14000)';
A = [10 + sin(t/50) + t/5000, 8 + cos(t/30)];
ROIs = struct('activity', {A(:, 1), A(:, 2)});
save_ref(out_dir, 'projmeta_preprocess_wf', 'synthetic', A, ...
    preprocess_widefield_activity_for_proj_meta(ROIs, [7000 7000], ''));
end

function ref_aux_method(out_dir)
% out1 = channel/project pairs (n x 2 cell), out2 = methods joined with ',' (n x 1 cell)
pairs = {'Running', 'X'; 'VRx', 'X'; 'Lick', 'X'; 'Reward', 'X'; 'nose_x', 'X'; 'nose_y_vel', 'X'; ...
    'TonotopyID', 'X'; 'opto', 'APC'; 'STIM', 'PLS'; 'Stim', 'CAT'; 'stim', 'YBA'; ...
    'StimulationPower', 'YBA'; 'VRmotorsensoryGain', 'SNM'; 'Stim', 'ECT'; 'screenPD', 'X'};
out = cell(size(pairs, 1), 1);
for k = 1:size(pairs, 1)
    out{k} = strjoin(assign_aux_method(pairs{k, 1}, pairs{k, 2}), ',');
end
save_ref(out_dir, 'projmeta_aux_method', 'synthetic', pairs, out);
end
