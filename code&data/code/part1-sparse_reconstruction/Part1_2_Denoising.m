% This code is for the Denoising Setting in the report
% You can just run run the code to test duck image.
% Noise setting is sigma 0.075 and delta = 0.9sigma now.

clear all;
clc;

bsz = 8;

fun = @(block_struct) reshape(block_struct.data, [], 1);
fun1 = @(block_struct) reshape(block_struct.data, bsz, bsz);

% construction of u

image = imread('../../test_images/512_512_ducks.png');
image_gray_orig = rgb2gray(image);
image_gray_orig = double(image_gray_orig) / 255;
image_gray_orig = imresize(image_gray_orig, 1);
[m, n] = size(image_gray_orig);
mn = m * n;
image_gray = blockproc(image_gray_orig, [bsz, bsz], fun);

u = reshape(image_gray, [], 1);

Psi = get_Psi(m, n, bsz);     


% noise 
sigma = 0.075;
delta_coef = 0.9;

% A is I now
A = speye(mn);
[row, col] = size(A);
u_noise = u + sigma * randn(mn, 1);
b = u + sigma * randn(mn, 1);

% calculate delta
delta = delta_coef * sigma * ones(row, 1);

lb = [zeros(2 * mn, 1); -Inf(mn, 1)];
ub = [   Inf(3* mn, 1)];

f = [ones(1, mn) ones(1, mn) zeros(1, mn)];
f_noise = f;

nzEq = 2 * nnz(-speye(mn)) + nnz(Psi);
nz = 2 * nnz(A);

bigAeq = spalloc(mn, 3 * mn, nzEq);
bigAeq(1:mn, 1:mn) = speye(mn);
bigAeq(1:mn, mn + 1:2 * mn) = -speye(mn);
bigAeq(1:mn, 2 * mn + 1:end) = -Psi;

bigA = spalloc(2 * row, 3 * mn, nz);
bigA(1:row, 1:mn) = sparse(row, mn);
bigA(1:row, mn + 1:2 * mn) = sparse(row, mn);
bigA(1:row, 2 * mn + 1:end) = speye(mn);
bigA(row + 1:2 * row, 1:mn) = sparse(row, mn);
bigA(row + 1:2 * row, mn + 1:2 * mn) = sparse(row, mn);
bigA(row + 1:2 * row, 2 * mn + 1:end) = -speye(mn);

bigBeq = sparse(zeros(mn, 1));
bigB = sparse([b + delta; delta - b]);

options = optimoptions('linprog', 'Algorithm', 'interior-point');
options.ConstraintTolerance = 1e-4;
options.Display = 'iter';
options.MaxIterations = 50;

tic
[h, fval, exitflag, output, lambda] = linprog(f, bigA, bigB, bigAeq, bigBeq, lb, ub, options);
toc

x = h([2 * mn + 1:end]);
x1 = reshape(x, [], n /bsz);
reconstructed_image = blockproc(x1, [bsz * bsz, 1], fun1);

b1 = reshape(b, [], n / bsz);
noisy_image = blockproc(b1, [bsz * bsz, 1], fun1);
   
PSNR_noise = 10*log10(m*n/norm(noisy_image - image_gray_orig, 2))
PSNR_noise_reconstruction = 10*log10(m*n/norm(reconstructed_image - image_gray_orig, 2))
PSNR_diff = PSNR_noise_reconstruction - PSNR_noise

% reconstructed_image is the reconstructed image
% noisy_image is the image with noise

imshow(reconstructed_image); 

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