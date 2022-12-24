% This code is for the Reformulation 1 in the report
% You can just run run the code for mask random 30 50 60

% For other masks that range from 0 to 255
% Please change the code for s into s = sum(ind/255, 'all');)
% We already out the code in the file, but it is annotation now.
% We set the max number of iterations be 50.
clear all;
clc;


fun = @(block_struct) reshape(block_struct.data, [], 1);
bsz = 8;

% construction of u
image = imread('../../test_images/512_512_circles.png');
image_gray_orig = rgb2gray(image);
image_gray_orig = double(image_gray_orig) / 255;
image_gray_orig = imresize(image_gray_orig, 1);
[m, n] = size(image_gray_orig);
mn = m * n;
image_gray = blockproc(image_gray_orig, [bsz, bsz], fun);

u = reshape(image_gray, [], 1);

% construction of A

mask_orig = imread('../../test_masks/512_512_random50.png');
mask_orig = imresize(mask_orig, 1);
mask = blockproc(mask_orig, [bsz, bsz], fun);


ind = reshape(mask, [], 1);

%s = sum(ind/255, 'all');
s = sum(ind, 'all');
I = find(ind);

% damage_image is just the damage image
damage_image = image_gray_orig;
damage_image(I) = 0;

A = sparse(1:s, I, ones(1, s), s, mn);

b = A * u;
Psi = get_Psi(m, n, bsz);     
f = Psi;

[row, col] = size(A);

delta = 0.06 * ones(row, 1);

lb = [zeros(2 * mn, 1); -Inf(mn, 1)];
ub = [   Inf(3* mn, 1)];

f = [ones(1, mn) ones(1, mn) zeros(1, mn)];

nzEq = 2 * nnz(-speye(mn)) + nnz(Psi);
nz = 2 * nnz(A);


bigAeq = spalloc(mn, 3 * mn, nzEq);
bigAeq(1:mn, 1:mn) = speye(mn);
bigAeq(1:mn, mn + 1:2 * mn) = -speye(mn);
bigAeq(1:mn, 2 * mn + 1:end) = -Psi;

bigA = spalloc(2 * row, 3 * mn, nz);
bigA(1:row, 1:mn) = sparse(row, mn);
bigA(1:row, mn + 1:2 * mn) = sparse(row, mn);
bigA(1:row, 2 * mn + 1:end) = A;
bigA(row + 1:2 * row, 1:mn) = sparse(row, mn);
bigA(row + 1:2 * row, mn + 1:2 * mn) = sparse(row, mn);
bigA(row + 1:2 * row, 2 * mn + 1:end) = -A;

bigBeq = sparse(zeros(mn, 1));
bigB = sparse([b + delta; delta - b]);

sol = reshape(image_gray, [], 1);
sol_z = abs(Psi * sol);

% set the terminate parameter
fprintf("Standard: %f", norm(Psi * sol, 1));
options = optimoptions('linprog', 'Algorithm', 'interior-point');
options.ConstraintTolerance = 1e-4;
%options.OptimalityTolerance = 1e-4;
options.Display = 'iter';
options.MaxIterations = 50;
tic
[h, fval, exitflag, output, lambda] = linprog(f, bigA, bigB, bigAeq, bigBeq, lb, ub, options);
toc

z = h([1:mn]);
x = h([2 * mn + 1:end]);
x1 = reshape(x, [], n / bsz);
fun1 = @(block_struct) reshape(block_struct.data, bsz, bsz);
x2 = blockproc(x1, [bsz * bsz, 1], fun1);


imshow(x2); 
   
PSNR = 10*log10(m*n/norm(x2 - image_gray_orig,2));

function Psi = get_Psi(m,n,dsz)           
    
    % fix dct-block size to 8
    if nargin <= 2
        dsz   = 8;
    end
    
    % display error if the image size is not compatible with the dct-block
    % size
    if mod(m,dsz) > 0 || mod(n,dsz) > 0
        error(strcat('Image size not a multiple of dsz = ',num2str(dsz,'%i')));
        Psi = [];
        return
    end
    
    % build Psi 
    D           = dctmtx(dsz); 
    Bdct        = kron(D',D);
    
    sz          = (m/dsz)*(n/dsz);
    Psi         = kron(speye(sz),Bdct);
end
