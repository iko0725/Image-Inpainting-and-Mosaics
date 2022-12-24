clear all;
clc;

r = 8;

tiles = cell([1, r]);
% for i = 1:r
%     tiles{i} = imread(sprintf('../../test_tiles/circles_16/circle_%d.png',i));
% end

for i = 1:r
    tiles{i} = imread(sprintf('../../test_tiles/symbols_64/symbol_%d.png', i));
end

l1 = [5 10 20 40];

l2 = [8 16 32 64];

for i = 1:length(l1)
    blockSize = l1(i)
    img = imread('../../test_images/640_640_lion.png');
    [image_mosaic, sol, iter, isEqual, fval, result, degenerate] = mosaic(img, tiles, blockSize);
    [fval_dual, iter_dual] = mosaic_dual(image_path, tiles, blockSize);
    iter
    iter_dual
    duality_gap = fval - (-fval_dual)
end