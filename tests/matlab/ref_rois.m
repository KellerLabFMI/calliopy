function ref_rois(sample_dir, out_dir) %#ok<INUSL>
% REF_ROIS  MATLAB outputs of the RoiQuest functions (08_RoiQuest), for calliopy's tests.
%
%   ref_rois(sample_dir, out_dir)
%
%   Run in MATLAB with calliope r2003 on the path, including the folder 08_RoiQuest (so that
%   the packages +utils, +gfx and +clbk resolve); Image Processing Toolbox needed, and
%   affinetform2d (R2022b or later). Every block builds a small input, calls the ORIGINAL
%   MATLAB function and saves the inputs followed by the outputs with save_ref as
%   ref_rq_<name>.mat (out1, out2, ... in the order given in the comment of the block;
%   input_file is 'synthetic'). tests/test_rois_ref.py loads these files. No lab data is
%   needed, so sample_dir is not used.
%
%   NOT covered (need the GUI app object, ExpLog or adata files): saveRoisAcute,
%   saveRoisChronic, wfSaveRois, autoRoiMatching, autodetectCellpose. Their computational
%   parts are covered through autoAdjustRois, checkRoiMatching, wf_transformROis,
%   filterTemplate, adjustImgContrast and the Roi classes.
%
%   A block that fails is reported and skipped; the others still run.

if isempty(which('utils.Roi'))
    error('ref_rois: put calliope 08_RoiQuest on the MATLAB path first');
end
rng(2);
blocks = {@rq_builtins, @rq_roi, @rq_roiwf, @rq_gfx, @rq_adjust, @rq_match, @rq_wf};
for k = 1:numel(blocks)
    try
        blocks{k}(out_dir);
    catch err
        fprintf(2, '%s failed: %s\n', func2str(blocks{k}), err.message);
    end
end
end

%% Image Processing Toolbox functions reproduced in calliopy.rois._builtins ------------------
function rq_builtins(out_dir)
% fspecial disk: out1 = radii, out2..out6 = fspecial('disk', r) for each radius
r = [1 2 2.5 7 26];
save_ref(out_dir, 'rq_fspecial', 'synthetic', r, fspecial('disk', r(1)), ...
    fspecial('disk', r(2)), fspecial('disk', r(3)), fspecial('disk', r(4)), fspecial('disk', r(5)));

% bwmorph and friends: out1 = BW, out2 = thin 1, out3 = thin Inf, out4 = thicken 1,
% out5 = thicken 3, out6 = spur 1, out7 = bwareaopen(BW, 5), out8 = imclose(BW, strel('disk',2)),
% out9 = bwperim(BW)
BW = imgaussfilt(rand(40, 50), 2) > 0.5;
BW(5, 5) = true; BW(20, 1:10) = true;
save_ref(out_dir, 'rq_bwmorph', 'synthetic', BW, bwmorph(BW, 'thin', 1), bwmorph(BW, 'thin', Inf), ...
    bwmorph(BW, 'thicken', 1), bwmorph(BW, 'thicken', 3), bwmorph(BW, 'spur', 1), ...
    bwareaopen(BW, 5), imclose(BW, strel('disk', 2)), bwperim(BW));

% top-hat: out1 = I, out2 = imtophat(I, strel('disk',2)), out3 = imtophat(I, strel('disk',5)),
% out4 = imtophat(I, strel('disk',5,0))
I = imgaussfilt(rand(40, 50), 1.5) * 100;
save_ref(out_dir, 'rq_tophat', 'synthetic', I, imtophat(I, strel('disk', 2)), ...
    imtophat(I, strel('disk', 5)), imtophat(I, strel('disk', 5, 0)));

% mat2gray / quantile: out1 = A, out2 = mat2gray(A), out3 = mat2gray(A, [0.2 0.7]),
% out4 = quantile(A(:), [0.005 0.995]), out5 = mat2gray(ones(3)*0.4, [0.5 0.5])
A = rand(9, 7); A(2, 3) = NaN;
save_ref(out_dir, 'rq_mat2gray', 'synthetic', A, mat2gray(A), mat2gray(A, [0.2 0.7]), ...
    quantile(A(:), [0.005 0.995]), mat2gray(ones(3) * 0.4, [0.5 0.5]));
end

%% utils.Roi ----------------------------------------------------------------------------------
function rq_roi(out_dir)
hw = [12 15];
% move: out1 = indices, out2 = indices after move(2,-3), out3 = shift,
% out4 = indices after a further move(-1, 0, [1 1]), out5 = shift
idx = sub2ind(hw, [1 5 12 6]', [1 7 15 2]');
R = utils.Roi(idx, hw);
R.move(2, -3); i2 = R.indices; s2 = R.shift;
R.move(-1, 0, [1 1]);
save_ref(out_dir, 'rq_roi_move', 'synthetic', idx, i2, s2, R.indices, R.shift);

% expand/shrink: out1 = indices (block), out2 = expand(1), out3 = expand(2), out4 = shrink(1)
[rr, cc] = ndgrid(4:8, 5:9);
blk = sub2ind(hw, rr(:), cc(:));
R = utils.Roi(blk, hw); R.expand(1); e1 = R.indices;
R = utils.Roi(blk, hw); R.expand(2); e2 = R.indices;
R = utils.Roi(blk, hw); R.shrink(1); s1 = R.indices;
save_ref(out_dir, 'rq_roi_morph', 'synthetic', blk, e1, e2, s1);

% label type images: out1 = idxA, out2 = idxB, out3 = typeA, out4 = A.toLabelTypeImage(),
% out5 = B.toLabelTypeImage() (empty type), out6 = [A B].toLabelTypeImage()
A = utils.Roi(blk, hw, 3); B = utils.Roi(sub2ind(hw, 11, 2), hw); B.type = [];
save_ref(out_dir, 'rq_roi_labels', 'synthetic', blk, B.indices, 3, A.toLabelTypeImage(), ...
    B.toLabelTypeImage(), [A B].toLabelTypeImage());

% struct round trip: out1 = rois2struct([A C]), out2 = rois2struct(struct2rois(out1, hw))
C = utils.Roi(idx, hw, 1); C.move(1, 1);
s = utils.rois2struct([A C]);
save_ref(out_dir, 'rq_roi_struct', 'synthetic', s, utils.rois2struct(utils.struct2rois(s, hw)));
end

%% utils.Roi_wf ---------------------------------------------------------------------------------
function rq_roiwf(out_dir)
% out1 = mask indices, out2 = A (affinetform2d.A), out3 = green2red(mask), out4 = red2green(mask),
% out5 = single index, out6 = green2red(single), out7 = red2green(single),
% out8 = toLabelTypeImage of two ROIs with Allen_index 50 and 9 (second overlapping the first)
hw = [20 24];
A = [cosd(5) -sind(5) 1.5; sind(5) cosd(5) -2.25; 0 0 1];
t = affinetform2d(A);
[rr, cc] = ndgrid(6:10, 4:9);
idx = sub2ind(hw, rr(:), cc(:));
one = sub2ind(hw, 9, 12);
R = utils.Roi_wf(idx, hw); R.transform_green2red(t); g2r = R.indices;
R = utils.Roi_wf(idx, hw); R.transform_red2green(t); r2g = R.indices;
R = utils.Roi_wf(one, hw); R.transform_green2red(t); g2r1 = R.indices;
R = utils.Roi_wf(one, hw); R.transform_red2green(t); r2g1 = R.indices;
P = utils.Roi_wf(idx, hw, 1, 1, 1, 'a', 'x', 'y', 50);
Q = utils.Roi_wf(sub2ind(hw, [8 15]', [5 20]'), hw, 1, 1, 1, 'b', 'x', 'y', 9);
save_ref(out_dir, 'rq_roiwf', 'synthetic', idx, A, g2r, r2g, one, g2r1, r2g1, [P Q].toLabelTypeImage());
end

%% gfx ----------------------------------------------------------------------------------------
function rq_gfx(out_dir)
% filterTemplate: out1 = T, out2 = legacy r=4, out3 = updated r=4, out4 = updated r=2
T = imgaussfilt(rand(48, 40), 2) * 1000 + 200;
save_ref(out_dir, 'rq_filter', 'synthetic', T, gfx.filterTemplate(T, 4, 'legacy'), ...
    gfx.filterTemplate(T, 4, 'updated'), gfx.filterTemplate(T, 2, 'updated'));

% adjustImgContrast: out1 = img, out2 = minMax, out3 = bestFit, out4 = custom [0.1 0.6]
img = T; img(3, 4) = NaN;
save_ref(out_dir, 'rq_contrast', 'synthetic', img, gfx.adjustImgContrast(img, 'minMax'), ...
    gfx.adjustImgContrast(img, 'bestFit'), gfx.adjustImgContrast(img, 'custom', [0.1 0.6]));

% makeCross etc.: out1 = centerRC (1-based), out2 = imgHW, out3 = makeCross(c, hw, 10),
% out4 = makeCross([1 1], hw, 10), out5 = getLabelsColormap(), out6 = uniformRgbImage([3 4])
c = [10 21]; hw = [30 40];
save_ref(out_dir, 'rq_cross', 'synthetic', c, hw, gfx.makeCross(c, hw, 10), ...
    gfx.makeCross([1 1], hw, 10), gfx.getLabelsColormap(), gfx.uniformRgbImage([3 4]));

% segmentNeuron with a struct standing in for the app: out1 = img (uint8),
% out2 = XLim, out3 = YLim, out4 = validArea, out5 = mask (imgIdx 1), out6 = mask (imgIdx 3)
img = gfx.adjustImgContrast(T, 'minMax');
app.imgs = {img; img; img}; app.imgIdx = 1; app.currentZ = 1;
app.gHandles.ax_img.XLim = [5 30]; app.gHandles.ax_img.YLim = [3 40];
app.thr = [160 160 160]; app.minRoiPix = 20;
va = true(size(img)); va(1:4, :) = false; app.validArea = va; app.ROIs = {[]};
m1 = gfx.segmentNeuron(app);
app.imgIdx = 3;
m3 = gfx.segmentNeuron(app);
save_ref(out_dir, 'rq_segment', 'synthetic', img, app.gHandles.ax_img.XLim, ...
    app.gHandles.ax_img.YLim, va, m1, m3);
end

%% clbk.autoAdjustRois ---------------------------------------------------------------------------
function rq_adjust(out_dir)
% out1 = ref, out2 = cur, out3 = indices ROI 1, out4 = indices ROI 2,
% out5..out9 = [dxdy, globalDxdy, isOutOfBounds, startCentroids, displacement],
% out10..out14 = the same with ignore_global = true
ref = imgaussfilt(randn(128, 128), 2) * 100 + 500;
cur = circshift(ref, [3 -2]);
cur(40:50, 60:70) = circshift(cur(40:50, 60:70), [1 0]);
hw = size(ref);
i1 = sub2ind(hw, [44 45 44]', [64 64 65]');
i2 = sub2ind(hw, 30, 90);
R = [utils.Roi(i1, hw), utils.Roi(i2, hw)];
[a1, a2, a3, a4, a5] = clbk.autoAdjustRois(R, cur, ref);
[b1, b2, b3, b4, b5] = clbk.autoAdjustRois(R, cur, ref, true);
save_ref(out_dir, 'rq_autoadjust', 'synthetic', ref, cur, i1, i2, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5);
end

%% checkRoiMatching / roiMatchMessage --------------------------------------------------------------
function rq_match(out_dir)
% out1 = origROIs (cell of struct), out2 = testRois, out3 = roiTrans, out4 = imgHW,
% out5..out7 = [bool, isOverlapping, isTypeMatched]
hw = [20 20];
o = struct('indices', {sub2ind(hw, [6 6 7]', [6 7 6]'), sub2ind(hw, 11, 11), sub2ind(hw, 3, 3)}, ...
    'shift', {[0 0], [0 0], [0 0]}, 'type', {1, 0, 0});
t = struct('indices', {sub2ind(hw, [8 8 9]', [5 6 5]'), sub2ind(hw, 13, 11), sub2ind(hw, 9, 9)}, ...
    'shift', {[0 -1], [0 0], [0 0]}, 'type', {1, 2, 0});
origROIs = {o}; testRois = {t}; roiTrans = {[2; 0; 0]};
[b, ov, ty] = utils.checkRoiMatching(origROIs, testRois, roiTrans, hw);
save_ref(out_dir, 'rq_checkmatch', 'synthetic', origROIs, testRois, roiTrans, hw, b, ov, ty);

% roiMatchMessage with overlap errors only (the type branch of the MATLAB code is not used):
% out1 = isOverlapping, out2 = isTypeMatched, out3 = msg
ov = {[true false false], [true true]}; ty = {[true true true], [true true]};
save_ref(out_dir, 'rq_matchmsg', 'synthetic', ov, ty, utils.roiMatchMessage(ov, ty));
end

%% wf_transformROis ------------------------------------------------------------------------------
function rq_wf(out_dir)
% out1 = ROIs_orig (struct), out2 = size_template_orig, out3 = A (affinetform2d.A),
% out4 = wf_transformROis(ROIs_orig, size, affinetform2d(A))
hw = [40 50];
[rr, cc] = ndgrid(10:16, 20:27);
ROIs = struct('indices', {sub2ind(hw, 5, 10), sub2ind(hw, 30, 12), sub2ind(hw, rr(:), cc(:)), ...
    sub2ind(hw, [1 2]', [1 1]')}, 'shift', {[0 0], [0 0], [0 0], [0 0]}, 'type', {0, 0, 0, 0}, ...
    'name', {'lambda', 'bregma', 'V1', 'edge'});
A = [cosd(-3) -sind(-3) 2.4; sind(-3) cosd(-3) 1.3; 0 0 1];
save_ref(out_dir, 'rq_wftransform', 'synthetic', ROIs, hw, A, ...
    utils.wf_transformROis(ROIs, hw, affinetform2d(A)));
end
