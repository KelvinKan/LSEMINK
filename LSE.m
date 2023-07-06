function varargout = LSE(J, x, C, B, w, beta, alpha)

    % objective function evaluation for
    % log-sum-exp (LSE)
    % f(w) = \sum_k w_k * [ log(1^T exp(J_k*x+b_k)) - c_k^T*J_k*x ]

    % [obj, grad, Hess, NGD_Hess] = LSE(Jvec, x, C, B, w, beta, alpha)

    % Inputs
    %
    % Jvec  - matrix or function handle to perform matrix-vector multiplication 
    %         w -> J*x (Jvec(x)) and w -> J^T*x (Jvec(x, "transpose"))
    %         where J = [J_1; J_2; ...; J_N]
    %
    % x     - variable to be evaluated at
    %
    % C     - C = [c_1, c_2, ..., c_N]
    %
    % b     - B = [b_1, b_2, ..., b_N]
    %
    % w     - vector containing weights w_i's
    %
    % beta  - penalty coefficient for NGD/LSEMINK
    %
    % alpha - Tikhonov regularization parameter 


    % Outputs
    %
    % obj       - objective function value
    %
    % grad      - gradient
    %
    % Hess      - Hessian (function handle)
    %
    % NGD_Hess  - Hessian for natural gradient descent (function handle)

    %% perform gradient test if no input is provided
    if nargin == 0
        gradtest();
        return;
    end

    if isnumeric(J)
        Jvec = @(x, varargin) Jfun(J, x, varargin{:});
    else
        Jvec = J;
    end

    nOutputs = nargout;

    % reshape into row vector
    w = reshape(w, 1, []);
    N = size(C, 2);

    % Check if columns of C sum to 1
    % Allow single precision error
    if norm(sum(C, 1) - ones(1, N))/N < 1e-8
        C_UnitCol = true;
    else
        C_UnitCol = false;
    end


    S = Jvec(x) + B(:);
    S = reshape(S, [], N);
    maxS = max(S, [], 1);
    % numerical stability
    S = S - maxS;
    expS = exp(S);
    colsum_expS = sum(expS, 1);
    P = expS./colsum_expS;

    % objective function
    if nOutputs >= 1

        LogSumExpS = sum(w.*log(colsum_expS), 2);
        if ~C_UnitCol
            % If the objective function is not multinomial logistic
            % regression (i.e. columns of C don't sum to 1), then
            % it is not shift invariant
            LogSumExpS = LogSumExpS + sum(w.*maxS, 2);
        end

        varargout{1} = (-sum(w.*C.*S, 'all') + LogSumExpS) + alpha/2*norm(x, 'fro')^2;
        
    end

    % gradient
    if nOutputs >= 2
        weighted_PC = w.*(P-C);
        varargout{2} = Jvec(weighted_PC, "transpose") + alpha*x;
    end

    % Hessian (function handle)
    if nOutputs >= 3
        varargout{3} = @(v) Hess_temp(Jvec, P, beta, alpha, N, w, v);
    end

    % L2 natural gradient descent's Hessian (function handle)
    if nOutputs >=4 
        varargout{4} = @(v) NGD_temp(Jvec, alpha, N, w, v);
    end
end

function result = Hess_temp(Jvec, P, beta, alpha, N, s, v)
    VA = s.*reshape(Jvec(v), [], N);
    U =  P.*VA - P .* sum(P.*VA, 1)  + beta*VA;
    result = Jvec(U(:), "transpose") + alpha*v;
end

function result = NGD_temp(Jvec, alpha, N, s, v)
    U = s.*reshape(Jvec(v), [], N);
    result = Jvec(U(:), "transpose") + alpha*v;
end

% gradient test
function results = gradtest()
    clear all; close all;
    rng("default")
    rng(1)

    n = 100;
    m = 20;
    N = 10;
    
    beta = 0;
    alpha = 1e-2;
    
    J = randn(m,n);
    B = randn(m,1);
    C = rand(m,1); C = C ./ sum(C, 1);
    w = 1;

    x = randn(n, 1);
    rand_direction = randn(n, 1);
    steps = 20;

    [f, grad_f, Hess] = LSE(J, x, C, B, w, beta, alpha);

    results = zeros(steps, 2);
    
    for step = 1:steps
        step_size = 2^(-step);
        y = x + step_size * rand_direction;
        
        f_prime = LSE(J, y, C, B, w, beta, alpha);
        results(step, 1) = abs(f_prime - f - step_size*grad_f'*rand_direction);
        results(step, 2) = abs(f_prime - f - step_size*grad_f'*rand_direction ...
                        - 0.5*step_size^2*rand_direction'*Hess(rand_direction));
    end
    
    semilogy(results)
end