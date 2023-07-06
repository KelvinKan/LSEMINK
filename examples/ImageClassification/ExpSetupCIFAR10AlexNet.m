clear all; close all;
rng("default")
rng(1)

N = 50000; Nval = 10000; 

dataset = 'CIFAR-10';

layer = 'pool5';
[Z1,C1,Z2,C2,Zv,Cv] = getCIFAR10AlexNet(N, Nval, layer);

Zt = cat(2, Z1, Z2); clear Z1 Z2;
Ct = cat(2, C1, C2); clear C1 C2;

n_f = size(Zt, 1);
n_c = size(Ct, 1);

Zt = normalizeData(Zt, n_f);
Zv = normalizeData(Zv, n_f);

his = {};
elapsed_time = {};

init_x = randn(n_c*n_f, 1);
B = 0;
w = 1/N * ones(N, 1);

Jvec = @(x, varargin) Jvec_MLR(Zt, x, varargin{:});
Jvec_val = @(x, varargin) Jvec_MLR(Zv, x, varargin{:});

options.alpha = 0;

options.fun_train_accuracy = @(x) CE_accuracy(Jvec, x, Ct, B);
options.fun_test_accuracy = @(x) CE_accuracy(Jvec_val, x, Cv, B);