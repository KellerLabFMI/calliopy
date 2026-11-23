function ref_aux(sample_dir, out_dir)
% REF_AUX  Save MATLAB outputs of the auxiliary-file readers for calliopy's tests.
%
%   ref_aux(sample_dir, out_dir)
%
%   Called by make_reference.m, with calliope r2003 on the path. For every reader
%   it takes the first sample file with the right extension (find_sample), runs the
%   ORIGINAL MATLAB function on it and saves all outputs as ref_<name>.mat (save_ref):
%
%       ref_readini.mat          readini          .ini            out1 = struct
%       ref_load_lvd.mat         load_lvd         .lvd            out1..out5 = data, scanrateA,
%                                                                 numchannels, timestamp, inputrange
%       ref_load_wcp.mat         load_wcp         .wcp            out1..out5 as load_lvd
%       ref_load_ach_chans.mat   load_ach_chans   .ach            out1 = cell array of labels
%       ref_parse_ach.mat        parse_ach        .ach            out1 = pdef struct
%       ref_parse_vrd.mat        parse_vrd        .vrd            out1..out3 = sections, raw_txt,
%                                                                 vrd_processed
%       ref_sidecar_csv.mat      readtable        _min.csv        out1 = csv_names, out2 = csv_values
%                                                 (else _beh1.csv)
%
%   A format without sample file is skipped. A reader that fails is reported and
%   the next one still runs, so one bad sample does not block the others.
%
%   Notes on how the readers are called:
%     - 'status' of load_lvd / load_wcp is not saved (0 when it works).
%     - load_wcp takes the format from the ExpID in the file name S1-T<ExpID>.wcp;
%       name the sample accordingly.
%     - parse_ach(pdef, proj, acode, ExpLog) asks getExpLog (the database) for an
%       ExpLog when none is passed, even for a file name. A dummy non-empty ExpLog
%       is passed below, so the file path is used and the database is never queried.
%     - parse_vrd returns a datetime array in vrd_processed.rd.read, which the v7.3
%       file would store as an opaque object. It is saved as datenum; the Python test
%       converts calliopy's datetimes in the same way.

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% readini  (.ini)
f = find_sample(sample_dir, '.ini');
if ~isempty(f)
    try
        out1 = readini(f);
        save_ref(out_dir, 'readini', f, out1);
    catch err
        fprintf(2, 'readini failed: %s\n', err.message);
    end
end

%% load_lvd  (.lvd)
f = find_sample(sample_dir, '.lvd');
if ~isempty(f)
    try
        [data, scanrateA, numchannels, timestamp, inputrange] = load_lvd(f);
        save_ref(out_dir, 'load_lvd', f, data, scanrateA, numchannels, timestamp, inputrange);
    catch err
        fprintf(2, 'load_lvd failed: %s\n', err.message);
    end
end

%% load_wcp  (.wcp)
f = find_sample(sample_dir, '.wcp');
if ~isempty(f)
    try
        [data, scanrateA, numchannels, timestamp, inputrange] = load_wcp(f);
        save_ref(out_dir, 'load_wcp', f, data, scanrateA, numchannels, timestamp, inputrange);
    catch err
        fprintf(2, 'load_wcp failed: %s\n', err.message);
    end
end

%% load_ach_chans  (.ach)
f = find_sample(sample_dir, '.ach');
if ~isempty(f)
    try
        chans = load_ach_chans(f);
        save_ref(out_dir, 'load_ach_chans', f, chans);
    catch err
        fprintf(2, 'load_ach_chans failed: %s\n', err.message);
    end

    %% parse_ach  (.ach)
    try
        dummy_explog = struct();   % non-empty: stops parse_ach from calling getExpLog
        pdef = parse_ach(f, [], [], dummy_explog);
        save_ref(out_dir, 'parse_ach', f, pdef);
    catch err
        fprintf(2, 'parse_ach failed: %s\n', err.message);
    end
end

%% parse_vrd  (.vrd)
f = find_sample(sample_dir, '.vrd');
if ~isempty(f)
    try
        try
            [sections, raw_txt, vrd_processed] = parse_vrd(f);
            vrd_processed.rd.read = datenum(vrd_processed.rd.read);   % datetime -> datenum
        catch err
            fprintf(2, 'parse_vrd: third output failed (%s); saving the first two only\n', err.message);
            [sections, raw_txt] = parse_vrd(f);
            vrd_processed = [];   % the Python test skips the third comparison
        end
        save_ref(out_dir, 'parse_vrd', f, sections, raw_txt, vrd_processed);
    catch err
        fprintf(2, 'parse_vrd failed: %s\n', err.message);
    end
end

%% sidecar csv  (readtable + table2array, as in load_exp.m lines 502-504 and 525-527)
f = find_sample(sample_dir, '_min.csv');
if isempty(f)
    f = find_sample(sample_dir, '_beh1.csv');
end
if ~isempty(f)
    try
        min_csv_file = readtable(f);
        csv_names = min_csv_file.Properties.VariableNames;
        csv_values = table2array(min_csv_file);
        save_ref(out_dir, 'sidecar_csv', f, csv_names, csv_values);
    catch err
        fprintf(2, 'sidecar_csv failed: %s\n', err.message);
    end
end
end
