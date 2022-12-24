function [fval, iter] = mosaic_dual(image_path, tiles, blockSize)

    % Read images
    image_orig = imread(image_path);
    if size(image_orig, 3) == 3 
        image_orig = rgb2gray(image_orig);
    end
    image_orig = double(image_orig) / 255;
    [m, n] = size(image_orig);
    
    
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
        tile_brightness(i) = mean(tiles{i}, 'all');
    end

    % image partition
    % blockproc handler to compute mean brightness 
    avg_brightness = @(block_struct) mean(block_struct.data, 'all');
    block_brightness = blockproc(image_orig, [l, l], avg_brightness);

    % Squared error (c_k - β_{i,j})^2 construction
    sqr_brightness = zeros(blockRows, blockCols, r);
    for i = 1:blockRows
        for j = 1:blockCols
            for k = 1:r
                sqr_brightness(i, j, k) = (tile_brightness(k) - block_brightness(i,j))^2;
            end
        end
    end

    s = reshape(sqr_brightness, [], 1);


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
    % A = (A1 A2)': R^{S * (B + r)}; c: R^{r * 1}; b: R^{(r + B) * 1} 
    %            s: R^{S * 1}      ; I: R^{S * S}
    % --------------------------
    % y1: R^{(B + r) * 1}; y2: R^{S * 1}
    % --------------------------
    % min  - (b' 1') (y1 y2)' 
    % s.t.
    %       (A' I) * (y1 y2)' ≤ s <=> AT * y ≤ s
    %        y1 free, y2 ≤ 0

    A = [A1; A2];
    b = [ c; b2];

    f  = -[b' ones(1, S)];
    AT = [A1' A2' speye(S)];
    ub = [ Inf(B + r, 1); sparse(S, 1)];
    lb = [-Inf(B + r, 1);   -Inf(S, 1)]; 

    options = optimoptions('linprog');
%     options.Display = 'iter';

    tic
    [y, fval, exitflag, output] = linprog(f, AT, s, [], [], lb, ub, options);
    toc

    iter = output.iterations;
    
end