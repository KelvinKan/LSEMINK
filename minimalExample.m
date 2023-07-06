% make a matrix of random numbers J of size 20 by 10 and a random vectors b and c of size 20 by 1 and call the lsemink function
% to solve the problem min_x log(sum(exp(J*x-b))) - c'*J*x

clear all; close all;
rng("default")
rng(1)

m = 20; n = 10;

J = randn(m,n); b = randn(m,1);
c = rand(m,1); c = c/sum(c);
x0 = rand(n,1);
x = lsemink(J,x0,c,b);