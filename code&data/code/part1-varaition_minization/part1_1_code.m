clear all;
clc;

% construction of utur
image = imread('../../test_images/640_640_eagle.png');%picture RGB. Path of image.
image = double(image) / 255;
image_gray = rgb2gray(image);
[m, n] = size(image_gray);
u = reshape(image_gray, [], 1);


% **************************************************
% construction of A
mask = imread('../../test_masks/640_640_random70.png');%Path of mask.
Ind = mask;

ind = vec(Ind);
s = sum(ind, 'all');
% s = sum(ind/255, 'all');%if mask == screatches_01 02
I = find(ind);

A = spalloc(s, m * n, s);
for i = 1:s
    index = I(i);
    A(i,index) = 1;
end


% **************************************************
%get b
b = A * u;


% **************************************************
% construction of D
kappa = 2*m*n-m-n;

D = spalloc(kappa, m*n , 2*kappa);
for k = 1:m-1
    for l = 1:n
        D((l-1)*(m-1)+k,(l-1)*m+k) = -1;
        D((l-1)*(m-1)+k,(l-1)*m+k+1) = 1;
    end
end
for k_1 = 1:m
    for l_1 = 1:n-1
        D(n*(m-1)+(l_1-1)*m+k_1,(l_1-1)*m+k_1) = -1;
        D(n*(m-1)+(l_1-1)*m+k_1,l_1*m+k_1) = 1;
    end
end

% **************************************************
% reformulation and calculation
% see report
f = [ones(1,2*kappa) zeros(1,m*n)];
Aeq_1 = [speye(kappa) speye(kappa)*(-1) -D];
beq_1 = zeros(kappa,1);
y_1 = sparse(s,kappa,0);
z_1 = sparse(s,kappa,0);
Aeq_2 = [y_1 z_1 A];
beq_2 = b;
lb = sparse(2*kappa+m*n,1,0);
ub = ones(2*kappa+m*n,1);

% **************************************************
% dual simplex method
options = optimoptions('linprog','Algorithm','dual-simplex');
options.ConstraintTolerance = 1e-3;
options.Display = 'Iter';

tic
[res,y] = linprog(f,[],[],[Aeq_1;Aeq_2],[beq_1;beq_2],lb,ub,options);
toc
res_x = res(2*kappa+1:2*kappa+m*n,1);%get x
res_i = reshape(res_x,[m,n]);%get image
imshow(res_i);
PSNR = 10*log10(m*n/norm(res_x - u,2));%psnr

% **************************************************
% Dual calculation:
[dual, dual_c] = linprog([beq_1;-beq_2]',[Aeq_1;Aeq_2]',f');
dual_res = -dual_c;%dual optimal value

% **************************************************
% interior-point method
options_2 = optimoptions('linprog','Algorithm','interior-point');
options_2.ConstraintTolerance = 1e-3;
options_2.Display = 'iter';

tic
[res_2,y_2] = linprog(f,[],[],[Aeq_1;Aeq_2],[beq_1;beq_2],lb,ub,options_2);
toc
res_x_2 = res_2(2*kappa+1:2*kappa+m*n,1);%get x
res_i_2 = reshape(res_x_2,[m,n]);%get image
imshow(res_i_2);
PSNR_2 = 10*log10(m*n/norm(res_x_2 - u,2));%get psnr

