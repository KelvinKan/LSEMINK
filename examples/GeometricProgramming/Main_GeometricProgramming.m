%
% REQUIREMENTS: This code requires the Meganet and CVX MATLAB package. 
% 
% git pull https://github.com/XtractOpen/Meganet.m
%
% and add to your MATLAB path for Meganet.
%
% download CVX on http://cvxr.com/cvx/download/

ExpSetupGeometricProgramming

save_figures = true;

% We minimize the log-sum-exp objective function
% min_x 1/\eta * log(\sum_{i=1}^n \exp((J_i'*x-b_i)/\eta))
% where J and b are randomly generated

% following the setting in
%
% 1. Kim, D., & Fessler, J. A. (2018). Adaptive restart of the 
% optimized gradient method for convex optimization. Journal of 
% Optimization Theory and Applications, 178(1), 240-263.
%
% 2. O’donoghue, B., & Candes, E. (2015). Adaptive restart for accelerated 
% gradient schemes. Foundations of computational mathematics, 15(3), 
% 715-732.

% Parameter for proximal term \beta/2 * ||Jd||_2^2
% This is the initial value of beta, line search will be performed 
% to increase beta (decrease step size) if line search condition is 
% not satisfied

% eta_list = [1e-6, 1e-4, 1e-2, 1];
eta_list = [1e-5, 1e-3, 1e-1, 1];

% Performs four different methods
% 1. (NCG) Standard Newton-CG with Armijo line search
% 2. (NGD) Natural gradient descent
% 3. (Identity Shift) Modified Newton-Krylov with a M=I shifting term
% 4. (Jacobian Shift) Modified Newton-Krylov with a M=J shifting term
solver_list = ["NCG", "NGD", "Identity Shift", "Jacobian Shift"];
marker_list = {'o', '*', '<', 'x'};

for eta = eta_list

    fprintf("perform experiments for eta = %d \n", eta);
        
    for i = 1: numel(solver_list)
        options.solver = solver_list(i);
    
        fprintf(strcat("Performing ", options.solver, "\n"))
    
        tic;
        [~, his{i}] = lsemink(J/eta, init_w, C, B/eta, eta, options);
        elapsed_time{i} = toc;
    
        figure(1)
        semilogy(his{i}.obj(:,1), his{i}.obj(:,2), 'DisplayName', options.solver, "LineWidth", 3, 'Marker', marker_list{i}, 'MarkerSize', 10)
        hold on
        figure(2)
        semilogy(his{i}.obj(:,1), his{i}.obj(:,3), 'DisplayName', options.solver, "LineWidth", 3, 'Marker', marker_list{i}, 'MarkerSize', 10)
        hold on
    end
    
    figure(1)
    if eta == eta_list(1)
        legend
    end
    ax=gca;
    ax.FontSize=20;
    
    if save_figures
        figure(1)
        figure_name = strcat('GP_obj_', 'eta=', num2str(eta), '.tex');
        matlab2tikz(figure_name)
    
        figure(2)
        figure_name = strcat('GP_grad_', 'eta=', num2str(eta), '.tex');
        matlab2tikz(figure_name)
    end
    
    close all
    
    num_CVXsolvers = 3;
    
    tic;
    % CVX solver 1: SeDuMi (free-to-use)
    cvx_begin
        cvx_precision best
        cvx_solver SeDuMi 
        variable x(n, 1)
        minimize( eta * log_sum_exp((J*x+B)/eta) )
    cvx_end
    elapsed_time{numel(solver_list) + 1} = toc;
    [obj, grad] = LSE(Jvec, x, C, B/eta, eta, 0, 0);
    his{numel(solver_list) + 1}.obj = [0, obj, norm(grad)];
    % 
    tic;
    % % CVX solver 2: SDPT3 (free-to-use)
    cvx_begin
        cvx_precision best
        cvx_solver SDPT3 
        variable x(n, 1)
        minimize( eta * log_sum_exp((J*x+B)/eta) ) 
    cvx_end
    elapsed_time{numel(solver_list) + 2} = toc;
    [obj, grad] = LSE(Jvec, x, C, B/eta, eta, 0, 0);
    his{numel(solver_list) + 2}.obj = [0, obj, norm(grad)];
    % 
    tic;
    % % CVX solver 3: MOSEK (commercial)
    cvx_begin
        cvx_precision best
        cvx_solver mosek 
        variable x(n, 1)
        minimize( eta * log_sum_exp((J*x+B)/eta) )
    cvx_end
    elapsed_time{numel(solver_list) + 3} = toc;
    [obj, grad] = LSE(Jvec, x, C, B/eta, eta, 0, 0);
    his{numel(solver_list) + 3}.obj = [0, obj, norm(grad)];
    
    all_method_list = [solver_list, "SeDuMi", "SDPT3", "mosek"];
    for i=1: numel(all_method_list)
        fprintf("method: %s, objective: %.2e, norm(grad): %.2e, elapsed time: %.2f s \n", all_method_list(i), his{i}.obj(end, 2), his{i}.obj(end, 3), elapsed_time{i})
    end

end