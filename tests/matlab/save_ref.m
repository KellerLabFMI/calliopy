function save_ref(out_dir, name, input_file, varargin)
% SAVE_REF  Save reader outputs as ref_<name>.mat (v7.3).
%
%   save_ref(out_dir, name, input_file, out1, out2, ...)
%   The file holds input_file (char) and out1, out2, ... in the order the
%   MATLAB reader returns them. calliopy's tests read it with load_mat.

s.input_file = input_file;
for k = 1:numel(varargin)
    s.(sprintf('out%d', k)) = varargin{k};
end
save(fullfile(out_dir, ['ref_' name '.mat']), '-struct', 's', '-v7.3');
fprintf('saved ref_%s.mat\n', name);
end
