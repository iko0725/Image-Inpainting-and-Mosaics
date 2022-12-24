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
    iter
    result
    degenerate
%     imwrite(image_mosaic, sprintf("../../mosaics/640_640_lion_mosaic_symbol_%d.png", blockSize));
end

