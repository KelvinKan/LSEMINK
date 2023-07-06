%
% REQUIREMENTS: This code requires the Meganet and CVX MATLAB package. 
% 
% git pull https://github.com/XtractOpen/Meganet.m
%
% and add to your MATLAB path for Meganet.
%
% download CVX on http://cvxr.com/cvx/download/

ExpSetupRFMMNIST

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
    figure_name = strcat('MLR(RFM)_training_', dataset, 'alpha=', num2str(options.alpha), '.tex');
    matlab2tikz(figure_name)

    figure(2)
    figure_name = strcat('MLR(RFM)_grad_', dataset, 'alpha=', num2str(options.alpha), '.tex');
    matlab2tikz(figure_name)

    figure(3)
    figure_name = strcat('MLR(RFM)_train_acc_', dataset, 'alpha=', num2str(options.alpha), '.tex');
    matlab2tikz(figure_name)

    figure(4)
    figure_name = strcat('MLR(RFM)_test_acc_', dataset, 'alpha=', num2str(options.alpha), '.tex');
    matlab2tikz(figure_name)
end