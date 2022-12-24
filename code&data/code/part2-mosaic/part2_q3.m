clear all;
clc;

r = 8;

tiles = cell([1, r]);

image = imread('../../test_images/640_640_lion.png');
for i = 1:r
    tiles{i} = imread(sprintf('../../test_tiles/circles_16/circle_%d.png',i));
end


blockSize = 10;
[image_mosaic, sol, iter, ifEqual, fval, result, degenerate] = mosaic(image, tiles, blockSize);

ifEqual