% Part II - Generating Mosaics
function [image_mosaic, x, iter, equal, fval, result, degenerate] = mosaic(image, tiles,  blockSize)

    % Read images
    image_rgb = double(image) / 255;
    image_gray = rgb2gray(image_rgb);
    image_gray2 = (image_rgb(:,:,1) + image_rgb(:,:,2) + image_rgb(:,:,3)) / 3;
    [m, n] = size(image_gray);
    if size(image_rgb, 3) == 3 
        image_gray = rgb2gray(image_rgb);
    end
    [m, n] = size(image_gray);
    
    
    % define constants
    [~, r] = size(tiles);       % number of tiles
    l = blockSize;              % block size
    blockRows = m / l;          % #blocks per row
    blockCols = n / l;          % #blocks per column
    B = blockRows * blockCols;  % #total blocks 
    S = r * B;                  % x_{i,j,k} size
    C = B / r;                  % #copies of each tiles
    
   
    % Read tiles and compute brightness c_k

    tile_brightness = zeros(r, 1);
    
    for i = 1:r
        tiles{i} = imresize(tiles{i}, [l, l]);
        tiles{i} = rgb2gray(tiles{i});
        tiles{i} = double(tiles{i}) / 255;
        tile_brightness(i) = (i - 1) / (r - 1);
    end

    % image partition
    % blockproc handler to compute mean brightness 
    avg_brightness = @(block_struct) mean(block_struct.data, 'all');
    block_brightness = blockproc(image_rgb, [l, l], avg_brightness);
    block_brightness_rgb = blockproc(image_gray2, [l, l], avg_brightness);
    

    % Squared error (c_k - β_{i,j})^2 construction
    sqr_brightness = zeros(blockRows, blockCols, r);
    sqr_brightness_rgb = zeros(blockRows, blockCols, r);
    
    for i = 1:blockRows
        for j = 1:blockCols
            for k = 1:r
                sqr_brightness_rgb(i, j, k) = (tile_brightness(k) - block_brightness_rgb(i,j))^2;
                sqr_brightness(i, j, k) = (tile_brightness(k) - block_brightness(i,j))^2;
            end
        end
    end

    s = reshape(sqr_brightness, [], 1);
    s_rgb = reshape(sqr_brightness_rgb, [], 1);

    % selection matrices A_1, A_2 construction:

    % A_1: 
    % ensure we exactly place [m / l * n / l] / r copies of each tile
    % <=> \sum_i \sum_j x_{i,j,k} = [m / l * n / l] / r  for any k

    A1_i = repelem(1:r, B);
    A1_j = 1:(S);
    A1_v = ones(1, S);

    A1 = sparse(A1_i, A1_j, A1_v, r, S);
    c = C * ones(r, 1);

    % A_2:
    % ensure we exactly place one (of the r many)) tiles in block (i, j)
    % <=> \sum_k x_{i,j,k} = 1 for any i, j

    A2_i = repelem(1:B, r);
    A2_j = reshape(permute(reshape(1:S, B, []), [2, 1]), 1, []);
    A2_v = ones(1, S);

    A2 = sparse(A2_i, A2_j, A2_v, B, S);
    b2 = ones(B, 1);

    % optimization reformulation
    % --------------------------
    % A1: R^{r * S}; A2: R^{B * S}; c: R^{r * 1}; b: R^{(r + B) * 1} 
    %  s: R^{S * 1}
    % --------------------------
    % min   s^T x
    % s.t.
    %       A1 * x = c --|
    %                    | => Ax = b 
    %       A2 * x = 1 --|  
    %       0 ≤ x ≤ 1

    A = [A1; A2];
    b = [ c; b2];
    ub = ones(S, 1);
    lb = zeros(S, 1);

    options = optimoptions('linprog');
    % options.Display = 'iter';

    tic
    [x, fval, exitflag, output] = linprog(s', [], [], A, b, lb, ub, options);
    toc

    [x_rgb, fval_rgb, exitflag_rgb, output_rgb] = linprog(s_rgb', [], [], A,b, lb, ub, options);
    
    equal = isequal(x, x_rgb);
    
    iter = output.iterations;
    
    % reshape x to x_{i,j,k}
    x = reshape(x, blockRows, blockCols, r);
    
    % check if the solution is the solution of original integer problem
    if abs(x - ceil(x)) <= 1e-5
        result = 1;
    else
        result = 0;
    end
    
    % check if the solution is degenerate
    if nnz(x) < size(A, 1)
        degenerate = 1;
    else
        degenerate = 0;
    end
    
    x = round(x);

    % mosaic construction
    image_mosaic = zeros(m, n);

    for i = 1:blockRows
        for j = 1:blockCols
            for k = 1:r
                if x(i, j, k) == 1
                    image_mosaic((i - 1) * l + 1: i * l, (j - 1) * l + 1: j * l) = tiles{k};
                end
            end
        end
    end
    
    
end