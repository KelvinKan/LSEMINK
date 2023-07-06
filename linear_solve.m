function [delta_W_vectorized, CGiter, relres] = linear_solve(Hess_vectorized, grad_vectorized, CG_tol, max_CG_iter, linear_solver)
    
    if nargin==0
        linsolvetest();
        return;
    end

    switch linear_solver
        case "CG"
            solver = steihaugPCG('maxIter',max_CG_iter,'tol',CG_tol);
            [delta_W_vectorized, para] = solver.solve(Hess_vectorized,-grad_vectorized,0*grad_vectorized);
            CGiter = numel(para.resvec)-1;
            relres = para.relres;
        case "MINRES"
            [delta_W_vectorized, flag, relres, CGiter, resvec] = minres(Hess_vectorized, -grad_vectorized, CG_tol, max_CG_iter);
        case "GMRES"
            [delta_W_vectorized, flag, relres, CGiter, resvec] = gmres(Hess_vectorized, -grad_vectorized, [], CG_tol, max_CG_iter);
            CGiter = CGiter(2);
        case "Lanczos"
            [T,V] = lanczosTridiag(Hess_vectorized,grad_vectorized,max_CG_iter+1,CG_tol,0,1);
            delta_W_vectorized = -V*(T\(V'*grad_vectorized));
            CGiter = size(T, 2);
            relres = norm(Hess_vectorized(delta_W_vectorized)+grad_vectorized)/norm(grad_vectorized);
        case "MatlabCG"
            [delta_W_vectorized, flag, relres, CGiter, resvec] = pcg(Hess_vectorized, -grad_vectorized, CG_tol, max_CG_iter);
        case "exact"
            m = size(grad_vectorized, 1);
            explicit_Hess = zeros(m, m);
            for i = 1: m
                basis_vector = zeros(m, 1);
                basis_vector(i) = 1;
                explicit_Hess(:, i) = Hess_vectorized(basis_vector);
            end
            explicit_Hess = 0.5 * (explicit_Hess' + explicit_Hess);

            opts.SYM = true;
            delta_W_vectorized = linsolve(explicit_Hess, -grad_vectorized, opts);
            CGiter = NaN; 
            relres = norm(Hess_vectorized(delta_W_vectorized)+grad_vectorized)/norm(grad_vectorized);
        otherwise
            error("linear solver not supported")
            return;
    end
end

function result = linsolvetest()
    clear all; close all;
    rng("default")
    rng(1)

    n = 100;
    tol = 1e-1;
    max_iter = 50;

    H = randn(n);
    H = H*H' + eye(n);
    H = @(x) H*x;

    b = randn(n, 1);

    fprintf("max_iter: %d, tol: %d \n", max_iter, tol)

    solver_list = ["CG", "MINRES", "GMRES", "Lanczos", "MatlabCG", "exact"];

    for solver = solver_list
        [~, iter, relres] = linear_solve(H, b, tol, max_iter, solver);
        fprintf("Solver: %s, no. iter: %d, relres: %d \n", solver, iter, relres)
    end
end