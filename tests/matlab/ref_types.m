function ref_types(~, out_dir)
% REF_TYPES  Write calliopy_types_v73.mat with one variable per MATLAB type.
%
%   tests/test_io_mat.py::test_real_matlab_types_file checks that load_mat reads
%   every variable as documented. Keep the two in step.

t.dbl = reshape(1:24, 2, 3, 4);
t.scalar = 3.5;
t.i16 = int16([1 -2 3]);
t.sgl = single([1.5 2.5]);
t.lgc = logical([1 0; 0 1]);
t.cpx = [1+2i, 3-4i];
t.chr = 'hello';
t.chrmat = ['ab'; 'cd'];
t.uni = char([181 109]);          % micro sign + m
t.empty_dbl = zeros(0, 3);
t.empty_chr = '';
t.empty_cell = {};
t.cel = {1, 'txt'; [1 2 3], {2}};
t.st = struct('b', 1, 'a', 'x');  % field order b, a
t.nested.inner.value = 7;
for i = 1:4                       % like proj_meta.rd: layers x ExpIDs
    for j = 1:2
        t.starr(i, j).act = (10*i + j) * ones(3, 5);
        t.starr(i, j).id = i * j;
    end
end
t.sp = sparse([1 0 0; 0 0 2]);
t.splog = sparse(logical([1 0; 0 1]));
t.tbl = table([1; 2], {'a'; 'b'});
t.str = "string scalar";
save(fullfile(out_dir, 'calliopy_types_v73.mat'), '-struct', 't', '-v7.3');
fprintf('saved calliopy_types_v73.mat\n');
end
