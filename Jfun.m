% function result = Jvec_GP(J, w, is_trans)
function result = Jvec_GP(J, w, varargin)

    if nargin>2 && strcmp(varargin{1}, "transpose")
        result = J'*w;
    else
        result = J*w;
    end
end