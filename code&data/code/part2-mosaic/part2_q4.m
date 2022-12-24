clear all;
clc;

r = 10;

tiles = cell([1, r]);

image_path = imread('../../test_images/640_640_lion.png');
for i = 1:r
    tiles{i} = imread(sprintf('../../own_tiles/tile_%d.png',i));
end

blockSize = 10;
[image_mosaic, sol, iter, fval, ifEqual, result, degenerate] = mosaic(image_path, tiles, blockSize);

imshow(image_mosaic);
ifEqual