% function result = Jvec_MLR(A, w, is_trans)
function result = Jvec_MLR(A, w, varargin)

    if nargin>2 && strcmp(varargin{1}, "transpose")
        % Get number of samples
        n = size(A, 2);
        W = reshape(w, [], n);
        result = W*A';
    else
        % Get number of features
        m = size(A, 1);
        W = reshape(w, [], m);
        result = W*A;
    end

    result = result(:);
end