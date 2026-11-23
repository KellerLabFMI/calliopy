function ref_containers(sample_dir, out_dir)
% REF_CONTAINERS  MATLAB reference values for load_metafile and Adata/load_adata.
%
%   Saves summaries computed by MATLAB's own reading of the sample files, so the
%   Python readers are checked against MATLAB, not against themselves.
%   Samples: one metafile '*_meta.mat' and one adata file '*S1-T*.mat'.

f = find_sample(sample_dir, '_meta.mat');
if ~isempty(f)
    s = load(f, 'proj_meta');
    pm = s.proj_meta;
    n_sites = numel(pm);
    rd_size = cell(1, n_sites);
    act_sum = cell(1, n_sites);
    for isite = 1:n_sites
        rd = pm(isite).rd;
        rd_size{isite} = size(rd);
        sums = nan(size(rd));
        if isfield(rd, 'act')
            for il = 1:size(rd, 1)
                for ie = 1:size(rd, 2)
                    a = rd(il, ie).act;
                    if iscell(a), a = a{1}; end
                    sums(il, ie) = sum(double(a(:)), 'omitnan');
                end
            end
        end
        act_sum{isite} = sums;
    end
    rd_fields = fieldnames(pm(1).rd)';
    save_ref(out_dir, 'load_meta', f, n_sites, rd_size, rd_fields, act_sum);
end

f = find_sample(sample_dir, 'S1-T*.mat');
if ~isempty(f)
    a = Adata(f);
    save_ref(out_dir, 'Adata', f, char(a.type), a.expId, who('-file', f)');
end
end
