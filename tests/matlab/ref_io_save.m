function ref_io_save(sample_dir, out_dir)
% REF_IO_SAVE  Check in MATLAB that calliopy.io.save_mat files load, for calliopy's tests.
%
%   ref_io_save(sample_dir, out_dir)
%
%   Before running, write the Python sample into sample_dir with
%       python tests/test_io_save.py <sample_dir>
%   which saves save_mat_python.mat (the variables of matlab_sample() in
%   tests/test_io_save.py). This function builds the same variables in MATLAB,
%   loads the Python file with load() and saves with save_ref as ref_io_save.mat:
%     out1 = struct of the variables built in MATLAB (input_file = the Python file)
%     out2 = struct of isequal(python_value, matlab_value) per variable
%     out3 = struct of class(python_value) per variable
%     out4 = struct of size(python_value) per variable
%     out5 = '' when the Python file loaded, otherwise the error message
%     out6 = struct of issparse(python_value) per variable
%   isequal ignores the numeric class, which is why out3 is compared separately.
%   sa1 and pm are 1x1 structs that Python writes in the struct-array layout
%   (fields as reference datasets), the layout save_mat uses for proj_meta.
%   tests/test_io_save.py::test_matlab_reads_python_file reads it.

m.d = [1 2 3; 4 5 6];
m.s = single([1.5 -2.25]);
m.i8 = int8([-3 4]);
m.u16 = uint16([7; 9]);
m.i64 = int64(-5);
m.b = [true false true];
m.c = ['h' char(233) 'llo'];
m.e = zeros(0, 3);
m.z = [1+2i, -3.5i];
m.st = struct('a', 1, 'txt', 'abc', 'inner', struct('v', 0:3));
m.sa = struct('name', {'x', 'yy'}, 'val', {[1 2], []}, 'tags', {{'a', 'bb'}, {}});
m.ce = {1, 'a'; [1 2 3], {}};
m.sp = sparse([1 3 2], [1 2 4], [1.5 -2 3], 3, 4);
m.sa1 = struct('k', 5);
m.pm = struct('animal', 'a1', 'rd', struct('act', {single([1 2]), single([])}, 'drug', {'x', {}}));

python_file = fullfile(sample_dir, 'save_mat_python.mat');
[flags, classes, sizes, sparsity] = deal(struct());
err = '';
if ~isfile(python_file)
    err = 'missing sample save_mat_python.mat (run python tests/test_io_save.py <sample_dir>)';
else
    try
        py = load(python_file);
        names = fieldnames(m);
        for k = 1:numel(names)
            n = names{k};
            if ~isfield(py, n)
                flags.(n) = false;
                classes.(n) = 'missing';
                sizes.(n) = [0 0];
                sparsity.(n) = false;
                continue
            end
            flags.(n) = isequal(py.(n), m.(n));
            classes.(n) = class(py.(n));
            sizes.(n) = size(py.(n));
            sparsity.(n) = issparse(py.(n));
        end
    catch me
        err = sprintf('load failed: %s (%s)', me.message, me.identifier);
    end
end
save_ref(out_dir, 'io_save', python_file, m, flags, classes, sizes, err, sparsity);
end
