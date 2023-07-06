clear all; close all;
rng("default")
rng(1)

dataset = 'CIFAR-10';
N = 100; Nval = 10000; n_f = 1000;

his = {};
elapsed_time = {};

[Zt, Zv, Ct, Cv] = getRFM(dataset, N, Nval, n_f);

n_c = size(Ct, 1);

init_x = randn(n_c*n_f, 1);
B = 0;
w = 1/N * ones(N, 1);

Jvec = @(x, varargin) Jvec_MLR(Zt, x, varargin{:});
Jvec_val = @(x, varargin) Jvec_MLR(Zv, x, varargin{:});

options.max_cost = 3000;
options.xtol = 1;

options.fun_train_accuracy = @(x) CE_accuracy(Jvec, x, Ct, B);
options.fun_test_accuracy = @(x) CE_accuracy(Jvec_val, x, Cv, B);