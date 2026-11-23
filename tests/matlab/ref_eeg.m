function ref_eeg(sample_dir, out_dir)
% REF_EEG  Save MATLAB r2003 outputs of the EEG and human-recording readers.
%
%   ref_eeg(sample_dir, out_dir)        (called by make_reference)
%
%   tests/test_io_eeg.py compares calliopy.io._eeg with the files written here:
%       ref_load_EEG.mat           out1 EEGdata, out2 EEGsync, out3 SampsPerStack
%       ref_load_hEEG.mat          out1 hEEGdata (struct)
%       ref_hEEG_filters.mat       out1..out3 FIR coefficients MATLAB's highpass(x,1,fs),
%                                  lowpass(x,100,fs) and bandstop(x,[49 51],fs) designed
%       ref_decodeAuxEEG.mat       out1 onsets of VR synchroSignal, out2/out3 stimulus onsets
%                                  and values of mm, out4 stimulus offsets of mm
%       ref_load_hEyeTracking.mat  out1 hEEGdata (struct)
%
%   The MATLAB readers take an ExpID (they look their files up in ExpLog), not a file name.
%   This script never queries the database: fill in the variables below. A reader whose
%   sample file or ExpID is missing is skipped.
%
%   Sample files in sample_dir are COPIES of the files the MATLAB reader opens for that
%   ExpID / stackID, named as follows (the Python tests look for these names):
%       S1-T<id>_EEG.dat, .oebin, .npy   load_EEG: pick an ExpID with ONE stack, so the
%                                        sample is small and the Python reader gets one .dat
%       S1-T<id>.txt and S1-T<id>.json   load_hEEG: same stack, same base name
%       S1-T<id>_eye.json                load_hEyeTracking: copy of that stack's S1-T<id>.json,
%                                        renamed so it is not mistaken for the hEEG log

% ======================================================================================
% FILL IN: leave a variable empty ([]) to skip the reader that needs it.
% ======================================================================================
exp_log      = [];   % ExpLog extract that the loaders take as argument. Obtain it yourself,
                     % for example  exp_log = getExpLog;  (that is the database query)
eeg_expid    = [];   % load_EEG: ExpID whose single stack was copied to sample_dir
heeg_expid   = [];   % load_hEEG: ExpID of the hEEG sample
heeg_stackid = [];   % load_hEEG: stackID of the hEEG sample
eye_expid    = [];   % load_hEyeTracking: ExpID of the eye-tracking sample
eye_stackid  = [];   % load_hEyeTracking: stackID of the eye-tracking sample
% ======================================================================================

%% load_EEG  ->  [EEGdata, EEGsync, SampsPerStack]
f = find_sample(sample_dir, '_EEG.dat');
if ~isempty(f)
    if isempty(eeg_expid) || isempty(exp_log)
        fprintf('load_EEG: eeg_expid / exp_log not filled in ref_eeg.m, skipped\n');
    else
        [EEGdata, EEGsync, SampsPerStack] = load_EEG(eeg_expid, exp_log);
        save_ref(out_dir, 'load_EEG', f, EEGdata, EEGsync, SampsPerStack);
    end
end

%% load_hEEG  ->  hEEGdata (written into the caller's workspace)
f = find_sample(sample_dir, '.txt');
if ~isempty(f)
    if isempty(heeg_expid) || isempty(heeg_stackid) || isempty(exp_log)
        fprintf('load_hEEG: heeg_expid / heeg_stackid / exp_log not filled in ref_eeg.m, skipped\n');
    else
        ws = run_load_hEEG(heeg_expid, heeg_stackid, exp_log);
        hEEGdata = ws.hEEGdata;
        save_ref(out_dir, 'load_hEEG', f, hEEGdata);

        % The filters highpass / lowpass / bandstop design themselves from the signal length
        % (FIR or IIR). Ask MATLAB for the filters it used on a signal as long as the real one.
        try
            x = hEEGdata.activity(1, :)';
            fs = 250;
            [~, d_high] = highpass(x, 1, fs);
            [~, d_low] = lowpass(x, 100, fs);
            [~, d_stop] = bandstop(x, [49 51], fs);
            save_ref(out_dir, 'hEEG_filters', f, d_high.Coefficients, d_low.Coefficients, ...
                d_stop.Coefficients);
        catch err
            fprintf(2, 'hEEG_filters skipped: %s\n', err.message);
        end

        % decodeAuxEEG / decodeAuxOffsetEEG (not in calliope r2003): save their outputs on
        % the VR log so the Python edge-detector assumption can be checked.
        [folder, base] = fileparts(f);
        f_json = fullfile(folder, [base '.json']);
        try
            vr = jsondecode(fileread(f_json));
            vr_table = struct2table(vr.records);
            [onsetVR, ~] = decodeAuxEEG(vr_table.synchroSignal, 0.5, 2);
            [stimOnset, stimValue] = decodeAuxEEG(vr_table.mm, 0.5, 2);
            stimOffset = decodeAuxOffsetEEG(vr_table.mm, 0.5, 2);
            save_ref(out_dir, 'decodeAuxEEG', f_json, onsetVR, stimOnset, stimValue, ...
                stimOffset);
        catch err
            fprintf(2, 'decodeAuxEEG reference skipped: %s\n', err.message);
        end
    end
end

%% load_hEyeTracking  ->  hEEGdata (written into the caller's workspace)
f = find_sample(sample_dir, '_eye.json');
if ~isempty(f)
    if isempty(eye_expid) || isempty(eye_stackid) || isempty(exp_log)
        fprintf('load_hEyeTracking: eye_expid / eye_stackid / exp_log not filled in ref_eeg.m, skipped\n');
    else
        ws = run_load_hEyeTracking(eye_expid, eye_stackid, exp_log);
        hEEGdata = ws.hEEGdata;
        save_ref(out_dir, 'load_hEyeTracking', f, hEEGdata);
    end
end
end

function ws = run_load_hEEG(ExpID, stackID, ExpLog)
% load_hEEG returns nothing: it does assignin('caller', 'hEEGdata', ...). Call it here, so
% this function's workspace receives the variable, then collect it with who.
load_hEEG(ExpID, stackID, ExpLog, 'caller');
ws = collect_assigned(who);
end

function ws = run_load_hEyeTracking(ExpID, stackID, ExpLog)
% Same as run_load_hEEG for load_hEyeTracking (it also assigns hEEGdata).
load_hEyeTracking(ExpID, stackID, ExpLog, 'caller');
ws = collect_assigned(who);
end

function ws = collect_assigned(names)
% Struct with one field per variable the loader put into the calling function's workspace.
% names = who, taken in that function right after the loader returned; the loader's own
% inputs are not part of the result.
inputs = {'ExpID', 'stackID', 'ExpLog'};
ws = struct;
for k = 1:numel(names)
    if ~any(strcmp(names{k}, inputs))
        ws.(names{k}) = evalin('caller', names{k});
    end
end
end
