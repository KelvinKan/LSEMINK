% function [x, his] = lsemink(J, init_x, C, B, w, options)  
function [x, his] = lsemink(J, init_x, C, B, varargin)  

    % Newton-Krylov methods for log-sum-exp minimization of the form
    % \sum_{i=1}^N log(1^T*exp(J_i*x-b)) - c_i^T*J_i*x

    % Four solvers provided:
    % 1. (NCG) Standard Newton-CG with Armijo line search
    % 2. (NGD) Natural gradient descent
    % 3. (Identity Shift) Modified Newton-Krylov with a M=I shifting term
    % 4. (Jacobian Shift) Modified Newton-Krylov with a M=J shifting term

    % Inputs:
    % 
    % J     - matrix or function handle to perform matrix-vector multiplication 
    %         w -> J*x (Jvec(x)) and w -> J^T*x (Jvec(x, "transpose"))
    %         where J = [J_1; J_2; ...; J_N]
    %
    % x     - vector containing initial guess for x
    %
    % C     - C = [c_1, c_2, ..., c_N]
    %
    % b     - B = [b_1, b_2, ..., b_N]
    %
    % w     - vector containing weights w_i's
    %
    % options:
    % 
    % init_beta        - initial value for beta to perform line search
    %
    % gtol             - tolerance for gradient
    %
    % xtol             - tolerance for relative change (||x_{i+1}-x_i||_2/||x_i||_2)
    %
    % max_Newton_iter  - max number of Newton iterations
    %
    % max_CG_iter      - max number of CG iterations
    %
    % CG_tol           - CG tolerance
    %
    % LanczosCoeff_tol - tolerance for Lanczos tridiagonalization,
    %                    which is used for the standard modified Newton
    %
    % max_cost         - max number of matrix-vector multiplications with
    %                    J and J^T
    %
    % alpha            - regularization parameter with 
    %                    \frac{\alpha}{2} ||w||_2^2
    % 
    % gamma            - line search parameter
    %
    % linear_solver    - linear solver for Newton equation
    %                  - "CG" or "MINRES" or "GMRES" or "Lanczos"
    %                  - "MatlabCG" or "exact"
    %
    % method           - Newton-Krylov method
    %                  "Newton-CG" or "natural gradient descent"
    %                  "standard modified Newton" or "proposed modified Newton"
    %
    % fun_train_accuracy - function to compute train/test accuracy
    % fun_test_accuracy  - fun_train_accuracy(w) outputs accuracy
    %

    % Outputs:
    %
    % x   - optimal variable
    % his - structure containing the history of iterations
    %       his.str: corresponding value in each column
    %       his.obj: values of history of iterations,
    %       each row is one iteration and is
    %       [objtive function value, norm of restricted gradient, ...
    %       percentage of active variables, CG relative error, ...
    %       training error, validation error, mu, number of function evaluations];

    nInputs = nargin;

    if nInputs > 4
        w = varargin{1};
    else
        nExamples = size(C, 2);
        w = 1/nExamples * ones(nExamples, 1);
    end

    if nInputs > 5
        options = varargin{2};
    else
        options = struct([]);
    end
 
    % if Jvec is numeric, make it a function handle
    if isnumeric(J)
        Jvec = @(x, varargin) Jfun(J, x, varargin{:});
    else
        Jvec = J;
    end


    % default options
    if isfield(options, 'init_beta')
        beta = options.init_beta;
    else
        beta = 1e-2;
    end

    if isfield(options, 'gtol')
        gtol = options.gtol;
    else
        gtol = 1e-14;
    end

    if isfield(options, 'xtol')
        xtol = options.xtol;
    else
        xtol = 1e-6;
    end

    if isfield(options, 'max_Newton_iter')
        max_Newton_iter = options.max_Newton_iter;
    else
        max_Newton_iter = 100;
    end

    if isfield(options, 'max_CG_iter')
        max_CG_iter = options.max_CG_iter;
    else
        max_CG_iter = 20 * ones(1, max_Newton_iter);
    end

    if isfield(options, 'CG_tol')
        CG_tol = options.CG_tol;
    else
        CG_tol = 1e-3 * ones(1, max_Newton_iter);
    end

    if isfield(options, 'LanczosCoeff_tol')
        LanczosCoeff_tol = options.LanczosCoeff_tol;
    else
        LanczosCoeff_tol = 1e-16 * ones(1, max_Newton_iter);
    end

    if isfield(options, 'max_cost')
        max_cost = options.max_cost;
    else
        max_cost = 2000;
    end

    if isfield(options, 'alpha')
        alpha = options.alpha;
    else
        alpha = 0;
    end

    if isfield(options, 'gamma')
        gamma = options.gamma;
    else
        gamma = 0.05;
    end

    if isfield(options, 'linear_solver')
        linear_solver = options.linear_solver;
    else
        linear_solver = "CG";
    end

    if isfield(options, 'solver')
        solver = options.solver;
    else
        solver = "Jacobian Shift";
    end

    if isfield(options, 'fun_train_accuracy')
        fun_train_accuracy = options.fun_train_accuracy;
        fun_test_accuracy = options.fun_test_accuracy;
    end

    if ~(strcmp(solver, "NCG") || strcmp(solver, "NGD") || ...
            strcmp(solver, "Identity Shift") || strcmp(solver, "Jacobian Shift"))
        error("line search not supported")
        return;
    end

    if strcmp(solver, "NCG")
        beta = 0;
    end

    % objective function
    fun = @(x, beta) ...
        LSE(Jvec, x, C, B, w, beta, alpha);

    x = init_x;

    [obj, grad] = fun(x, beta);

    old_obj = inf; % initialize into the loop
    relres = 0; CGiter = 0; LS_iter = 0; % initialize for printing
    Newton_iter = 1;
    step_size = 1;

    % counting the number of matrix vector products with J or J^T
    cost = 0;

    switch solver
        case {"NCG", "NGD"}
            his.str = 'Newton_iter: %d, work unit: %d, obj: %d, norm(grad): %d, relres: %d, CG_iter: %d, step_size: %d, LS_iter: %d, train_accuracy: %d, test_accuracy: %d \n';
        otherwise
            his.str = "Newton_iter: %d, work unit: %d, obj: %d, norm(grad): %d, relres: %d, CG_iter: %d, beta: %d, LS_iter: %d, train_accuracy: %d, test_accuracy: %d \n";
    end

    his.obj = zeros(max_Newton_iter+1, 9);
    % If no validation, then print val_obj as NaN
    val_obj = NaN;
    train_accuracy = NaN; test_accuracy = NaN;
    
    while ((norm(grad, 2) > gtol || abs(obj - old_obj)/obj > xtol)) ...
            && cost <= max_cost && Newton_iter <= max_Newton_iter

        %------------------storing and printing----------------------------
        if exist('fun_train_accuracy', 'var')
            train_accuracy = fun_train_accuracy(x);
            test_accuracy = fun_test_accuracy(x);
        end

        switch solver
            case {"NCG", "NGD"}
                his.obj(Newton_iter, :) = [cost, obj, norm(grad, 2), relres, CGiter, step_size, LS_iter, train_accuracy, test_accuracy];
                if LS_iter==1; step_size = min(step_size*2, 1); end
            otherwise
                his.obj(Newton_iter, :) = [cost, obj, norm(grad, 2), relres, CGiter, beta, LS_iter, train_accuracy, test_accuracy];
                if LS_iter==1; beta = beta/2; end
        end
        fprintf(his.str, Newton_iter - 1, his.obj(Newton_iter, :));
        %------------------storing and printing----------------------------

        cost = cost + 2; % obj and grad
        
        switch solver
            case "NCG"
                [obj, grad, Hess] = fun(x, beta);
                [delta_w, CGiter, relres] = linear_solve(Hess, grad, CG_tol(Newton_iter), max_CG_iter(Newton_iter), linear_solver);

                cost = cost + 2*CGiter; % Hess-vec

                % Armijo line search
                LS_iter = 1;

                cost = cost + 1; % obj_trial evaluation

                while fun(x + step_size*delta_w, beta) > obj + gamma * step_size * grad'*delta_w %sum(grad.*delta_w, "all")
                    step_size = step_size * 0.5;
                    LS_iter = LS_iter + 1;

                    cost = cost + 1; % obj evaluation
                end

            case "Jacobian Shift"

                for LS_iter = 1:50
                    [obj, grad, Hess] = fun(x, beta);

                    [delta_w, CGiter, relres] = linear_solve(Hess, grad, CG_tol(Newton_iter), max_CG_iter(Newton_iter), linear_solver);

                    cost = cost + 2*CGiter + 1; % Hess-vec and obj_trial

                    entropy_trial = fun(x + delta_w, beta);
                    if entropy_trial <= obj + gamma * grad'*delta_w 
                        break
                    end
                    beta = beta * 2; % decrease step size
                end

            case "NGD"
                [obj, grad, ~, NGD_Hess] = fun(x, beta);

                % Gauss-Newton-Krylov
                [delta_w, CGiter, relres] = linear_solve(NGD_Hess, grad, CG_tol(Newton_iter), max_CG_iter(Newton_iter), linear_solver);

                cost = cost + 2*CGiter;
                
                for LS_iter = 1:50
                    entropy_trial = fun(x + step_size*delta_w, beta);

                    cost = cost + 1;

                    if entropy_trial <= obj + gamma * step_size * grad'*delta_w
                        break
                    end
                    step_size = step_size/2;
                end

            case "Identity Shift"

                LS_iter = 1;

                [obj, grad, Hess] = fun(x, 0);

                Hess_noreg = @(v) Hess(v) - alpha * v;

                % max_CG_iter + 1 to be consistent with MATLAB's CG
                % see the minimal example in linear_solve
                [T, V, relres] = lanczosTridiag(Hess_noreg, grad, max_CG_iter(Newton_iter) + 1, CG_tol(Newton_iter), LanczosCoeff_tol(Newton_iter), 1);
                CGiter = size(T, 1);

                cost = cost + 2*CGiter; % Hess-vec

                cost = cost + 1; % obj_trial evaluation

                opts.SYM = true;

                minus_VT_grad = - V'*grad;

                delta_w = V*linsolve(T + (alpha + beta) *eye(size(T)), minus_VT_grad, opts);

                while fun(x + delta_w, beta) > obj + gamma * grad'*delta_w
                    beta = beta * 2;
                    LS_iter = LS_iter + 1;

                    cost = cost + 1; % obj evaluation

                    minus_VT_grad = - V'*grad;
                    delta_w = V*linsolve(T + (alpha + beta) *eye(size(T)), minus_VT_grad, opts);
                end

        end

        x = x + step_size*delta_w;
        old_obj = obj;
        
        [obj, grad] = fun(x, beta);

        Newton_iter = Newton_iter + 1;

    end

    %------------------storing and printing----------------------------
    if exist('fun_train_accuracy', 'var')
        train_accuracy = fun_train_accuracy(x);
        test_accuracy = fun_test_accuracy(x);
    end

    switch solver
        case {"NCG", "NGD"}
            his.obj(Newton_iter, :) = [cost, obj, norm(grad, 2), relres, CGiter, step_size, LS_iter, train_accuracy, test_accuracy];
            if LS_iter==1; step_size = min(step_size*2, 1); end
        otherwise
            his.obj(Newton_iter, :) = [cost, obj, norm(grad, 2), relres, CGiter, beta, LS_iter, train_accuracy, test_accuracy];
            if LS_iter==1; beta = beta/2; end
    end
    fprintf(his.str, Newton_iter - 1, his.obj(Newton_iter, :));
    %------------------storing and printing----------------------------

    his.obj(Newton_iter+1:end, :) = [];
end
