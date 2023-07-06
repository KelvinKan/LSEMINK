function [Zt, Zv, Ct, Cv] = getRFM(dataset, N, Nval, n_f)

% activation, number of example, dataset
    switch dataset
        case "MNIST"
            [Y,C] = setupMNIST(N+Nval);
        case {"CIFAR10", "CIFAR-10"}
            [Y,C] = setupCIFAR10(N+Nval);
    end

    dim1 = size(Y,1);dim2=size(Y,2);dim3=size(Y,3);
    Y    = normalizeData(Y,dim1*dim2*dim3);

    id = randperm(size(C,2));
    idt = id(1:N);
    idv = id(N+1:end);

    Yt  = reshape(Y(:,:,:,idt),dim1*dim2*dim3,[]); Ct = C(:,idt);
    Yv  = reshape(Y(:,:,:,idv),dim1*dim2*dim3,[]); Cv = C(:,idv);

    K = sampleSd(dim1*dim2*dim3,n_f-1);
    b = sampleSd(n_f-1,1)';

    Zt = [max(K*Yt+b,0); ones(1,size(Yt,2))];
    Zv = [max(K*Yv+b,0); ones(1,size(Yv,2))];
end

function X = sampleSd(d,n)
%
% create n i.i.d samples uniformly on S_{d-1}
%
% Input:
%   d - spatial dimension
%   n - number of points
%
% Output:
%   X - matrix of samples, size(X)=[n,d]

if nargin==0   
    d = 3;
    n = 2000;
    X = feval(mfilename,d,n);    
    if not(all(abs(sum(X.^2,2)-1.0)<1e-3))
        error('points not on sphere')
    end    
    if d==3
        figure(d); clf;
        plot3(X(:,1),X(:,2),X(:,3),'.r','MarkerSize',20);
    elseif d==2
        figure(d); clf;    
        plot(X(:,1),X(:,2),'.r','MarkerSize',20);
    end
    return
end

X = randn(n,d);
r = sqrt(sum(X.^2,2));
X = X./r;

end