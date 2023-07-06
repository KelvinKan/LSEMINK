%
% REQUIREMENTS: This code requires the Meganet and CVX MATLAB package. 
% 
% git pull https://github.com/XtractOpen/Meganet.m
%
% and add to your MATLAB path for Meganet.
%
% download CVX on http://cvxr.com/cvx/download/

ExpSetupSmallCIFAR10

save_figures = true;

% Performs four different methods
% 1. (NCG) Standard Newton-CG with Armijo line search
% 2. (NGD) Natural gradient descent
% 3. (Identity Shift) Modified Newton-Krylov with a M=I shifting term
% 4. (Jacobian Shift) Modified Newton-Krylov with a M=J shifting term

solver_list = ["NCG", "NGD", "Identity Shift", "Jacobian Shift"];
marker_list = {'o', '*', '<', 'x'};

for i = 1: numel(solver_list)
    options.solver = solver_list(i);
    fprintf(strcat("Performing ", options.solver, "\n"))

    tic;
    [x, his{i}] = lsemink(Jvec, init_x, Ct, B, w, options);
    elapsed_time{i} = toc;
end

figure(1)
for i = 1: numel(solver_list)
    semilogy(his{i}.obj(:,1),his{i}.obj(:,2), 'DisplayName', solver_list(i), "LineWidth", 3, 'Marker', marker_list{i}, 'MarkerSize', 10)
    hold on
end
figure(2)
for i = 1: numel(solver_list)
    semilogy(his{i}.obj(:,1),his{i}.obj(:,3), 'DisplayName', solver_list(i), "LineWidth", 3, 'Marker', marker_list{i}, 'MarkerSize', 10)
    hold on
end
figure(3)
for i = 1: numel(solver_list)
    plot(his{i}.obj(:,1),his{i}.obj(:,end-1), 'DisplayName', solver_list(i), "LineWidth", 3, 'Marker', marker_list{i}, 'MarkerSize', 10)
    hold on
end
figure(4)
for i = 1: numel(solver_list)
    plot(his{i}.obj(:,1),his{i}.obj(:,end), 'DisplayName', solver_list(i), "LineWidth", 3, 'Marker', marker_list{i}, 'MarkerSize', 10)
    hold on
end

figure(1)
legend
ax=gca;
ax.FontSize=20;

if save_figures
    figure(1)
    figure_name = strcat('SmallMLR_training_', dataset, '.tex');
    matlab2tikz(figure_name)

    figure(2)
    figure_name = strcat('SmallMLR_grad_', dataset, '.tex');
    matlab2tikz(figure_name)

    figure(3)
    figure_name = strcat('SmallMLR_train_acc_', dataset, '.tex');
    matlab2tikz(figure_name)

    figure(4)
    figure_name = strcat('SmallMLR_test_acc_', dataset, '.tex');
    matlab2tikz(figure_name)
end

fprintf("Begin CVX \n")

tic;
% CVX solver 1: SeDuMi (free-to-use)
cvx_begin
    cvx_precision best
    cvx_solver SeDuMi 
    variable X(n_c, n_f)
    % CVX does not allow sum over 'all', the 2nd argument must be int
    minimize( -1/N*sum(sum(Ct.*(X*Zt), 1), 2) + 1/N*sum(log_sum_exp(X*Zt, 1),2) )
cvx_end
elapsed_time{numel(solver_list)+1} = toc;
[obj, grad] = LSE(Jvec, X(:), Ct, B, w, 0, 0);
his{numel(solver_list)+1}.obj = [0, obj, norm(grad)];

tic;
% CVX solver 2: SDPT3 (free-to-use)
cvx_begin
    cvx_precision best
    cvx_solver SDPT3 
    variable X(n_c, n_f)
    minimize( -1/N*sum(sum(Ct.*(X*Zt), 1), 2) + 1/N*sum(log_sum_exp(X*Zt, 1),2) )
cvx_end
elapsed_time{numel(solver_list)+2} = toc;
[obj, grad] = LSE(Jvec, X(:), Ct, B, w, 0, 0);
his{numel(solver_list)+2}.obj = [0, obj, norm(grad)];

tic;
% CVX solver 3: MOSEK (commercial)
cvx_begin
    cvx_precision best
    cvx_solver mosek 
    variables X(n_c, n_f)
    minimize( -1/N*sum(sum(Ct.*(X*Zt), 1), 2) + 1/N*sum(log_sum_exp(X*Zt, 1),2) )
cvx_end
elapsed_time{numel(solver_list)+3} = toc;
[obj, grad] = LSE(Jvec, X(:), Ct, B, w, 0, 0);
his{numel(solver_list)+3}.obj = [0, obj, norm(grad)];

all_method_list = [solver_list, "SeDuMi", "SDPT3", "mosek"];
for i=1: numel(all_method_list)
    fprintf("method: %s, objective: %.2e, norm(grad): %.2e, elapsed time: %.2f s \n", all_method_list(i), his{i}.obj(end, 2), his{i}.obj(end, 3), elapsed_time{i})
end