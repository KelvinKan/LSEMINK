function accuracy = CE_accuracy(Jvec, w, C, B)

    N = size(C, 2);

    S = Jvec(w, 0) - B(:);
    S = reshape(S, [], N);
    S = S - max(S, [], 1);
        
    expS = exp(S);
    colsum_expS = sum(expS, 1);
    P = expS./colsum_expS;

    [~, prediction] = max(P, [], 1);
    [~, ground_truth] = max(C, [], 1);

    accuracy = sum(prediction == ground_truth)/size(prediction, 2);

end
