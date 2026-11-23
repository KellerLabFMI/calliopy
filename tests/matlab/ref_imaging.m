function ref_imaging(sample_dir, out_dir)
% REF_IMAGING  MATLAB outputs of the imaging and video readers, for calliopy's tests.
%
%   ref_imaging(sample_dir, out_dir)
%
%   Calls the ORIGINAL calliope r2003 readers on one sample file per format and saves
%   everything they return with save_ref as ref_<reader>.mat (input_file, out1, out2, ...
%   in MATLAB's output order). tests/test_io_imaging.py::test_ref_* load these files.
%   Formats without a sample file in sample_dir are skipped.
%
%   Samples are taken by extension (first file found):
%       .bin  load_bin                 .wid  load_wid_data
%       .roi  load_roi_data  (its .wii must be in the same folder)
%       .wii  extract_wii_file_platform
%       .eye  load_eye_monitor_data    .tif  imread
%       one of vid_exts (below)  load_vid_data
%       auxvid_ext (below)       load_auxVid_data
%
%   Keep small samples: wid/roi/eye/vid/bin data are loaded completely and saved.
%
%   Extra references (more than one frames option per reader) are saved under
%   ref_<reader>_<option>.mat. When a reader is called with options, they are saved as
%   one more output after MATLAB's own (a struct), so the Python test can repeat them.

% ---- settings you may need to change ---------------------------------------------------
nbr_piezo_layers = 1;      % piezo layers of the .bin sample (piezo.nbrlayers in its .ini);
                           % load_bin does NOT read the .ini when it is given a file name
vid_exts = {'.run', '.vid', '.vid2', '.ton'};   % load_vid_data: the first extension that has
                                                % a sample is used
auxvid_ext = '';           % FILL IN: file extension of an auxiliary-video sample for
                           % load_auxVid_data (it is not named anywhere in the sources);
                           % '' skips that reader
auxvid_bit_depth = 'uint16';   % pixel type of that sample: 'uint16' or 'uint8'
% ----------------------------------------------------------------------------------------

%% load_bin ---------------------------------------------------------------------------------
f = find_sample(sample_dir, '.bin');
if ~isempty(f)
    % MATLAB applies a file-name hack here (circshift of the layers when the name contains
    % S1-T231419); the Python test passes layer_shift=-1 for that stack.
    [data, nbr_frames] = load_bin(f, nbr_piezo_layers);
    opts = struct('nbr_piezo_layers', nbr_piezo_layers);   % not a MATLAB output
    save_ref(out_dir, 'load_bin', f, data, nbr_frames, opts);
end

%% load_wid_data ------------------------------------------------------------------------------
f = find_sample(sample_dir, '.wid');
if ~isempty(f)
    [wid_data, wid_meta_info] = load_wid_data({f}, 'verbose', 0);
    save_ref(out_dir, 'load_wid_data', f, wid_data, stamps_to_char(wid_meta_info));
    wid_data = load_wid_data({f}, 'frames', 'firstlast', 'verbose', 0);
    save_ref(out_dir, 'load_wid_data_firstlast', f, wid_data);
    [~, wid_meta_info] = load_wid_data({f}, 'frames', 'meta', 'verbose', 0);
    save_ref(out_dir, 'load_wid_data_meta', f, stamps_to_char(wid_meta_info));
end

%% load_roi_data (needs the .wii next to the .roi) ----------------------------------------------
f = find_sample(sample_dir, '.roi');
if ~isempty(f)
    [roi_data, roi_meta_info] = load_roi_data({f}, 'verbose', 0);
    save_ref(out_dir, 'load_roi_data', f, roi_data, stamps_to_char(roi_meta_info));
    [~, roi_meta_info] = load_roi_data({f}, 'frames', 'meta', 'verbose', 0);
    save_ref(out_dir, 'load_roi_data_meta', f, stamps_to_char(roi_meta_info));
end

%% extract_wii_file_platform (python: load_wii_data) ----------------------------------------------
f = find_sample(sample_dir, '.wii');
if ~isempty(f)
    [wii_config, wii_mean_proj, frame_nonrotated] = extract_wii_file_platform(f);
    save_ref(out_dir, 'extract_wii_file_platform', f, wii_config, wii_mean_proj, frame_nonrotated);
end

%% load_eye_monitor_data ---------------------------------------------------------------------
f = find_sample(sample_dir, '.eye');
if ~isempty(f)
    [idata, imeta_info] = load_eye_monitor_data(f, 1);
    save_ref(out_dir, 'load_eye_monitor_data', f, idata, imeta_info);
    [idata, imeta_info] = load_eye_monitor_data(f, 0);      % meta info only
    save_ref(out_dir, 'load_eye_monitor_data_noimg', f, idata, imeta_info);
end

%% load_vid_data (99_Archive/LoadSaveView; load_exp uses it for .run .vid .vid2 .ton) ------------
if isempty(which('load_vid_data'))
    fprintf('load_vid_data is not on the path (99_Archive/LoadSaveView), skipped\n');
else
    f = '';
    for k = 1:numel(vid_exts)
        f = find_sample(sample_dir, vid_exts{k});
        if ~isempty(f), break, end
    end
    if ~isempty(f)
        [vdata, vmeta_info] = load_vid_data(f, 'int16');
        save_ref(out_dir, 'load_vid_data', f, vdata, vmeta_info);
    end
end

%% load_auxVid_data -----------------------------------------------------------------------------
if isempty(auxvid_ext)
    fprintf('auxvid_ext is not set in ref_imaging.m, load_auxVid_data skipped\n');
else
    f = find_sample(sample_dir, auxvid_ext);
    if ~isempty(f)
        opts = struct('bit_depth', auxvid_bit_depth);          % not a MATLAB output
        [vid_data, vid_meta_info] = load_auxVid_data({f}, 'verbose', 0, 'bit_depth', auxvid_bit_depth);
        save_ref(out_dir, 'load_auxVid_data', f, vid_data, vid_meta_info, opts);
        [~, vid_meta_info] = load_auxVid_data({f}, 'frames', 'meta', 'verbose', 0, ...
            'bit_depth', auxvid_bit_depth);
        save_ref(out_dir, 'load_auxVid_data_meta', f, vid_meta_info, opts);
    end
end

%% tif: the imread in load_exp.m ---------------------------------------------------------------
f = find_sample(sample_dir, '.tif');
if ~isempty(f)
    tif_data = imread(f);
    save_ref(out_dir, 'tif', f, tif_data);
end
end


function m = stamps_to_char(m)
% STAMPS_TO_CHAR  datetime objects cannot be read from v7.3 files by calliopy: store the
% timestamps of a meta-info struct array as text ('yyyy-MM-dd HH:mm:ss.SSSSSS').
if isstruct(m) && isfield(m, 'timestamp')
    for k = 1:numel(m)
        if isdatetime(m(k).timestamp)
            m(k).timestamp = char(m(k).timestamp);
        end
    end
end
end
