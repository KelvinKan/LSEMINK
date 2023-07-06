clear all; close all;
rng("default")
rng(1)

% maximum work unit
options.max_cost = 10000;
options.max_Newton_iter = 2000;

his = {};
elapsed_time = {};

m = 100; % number of rows
n = 20; % dimension of solution (variable)

% randomly generate J and b
J = randn(m,n);
B = randn(m,1);

% c is absent
C = 0;

init_w = rand(n,1);