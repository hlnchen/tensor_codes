% Parameters:
n = 11;                     % dimension of original vector
k = 9;                      % rank
% m =  [20, 50, 100, 200, 500, 1000];                 % number of measurements
m =  [2000];
% Generate random 4-tensor(in d^2 by d^2 matrix) of rank k
X = zeros(n^2,n^2);
for i = 1:k
    A = randn(n,n)/sqrt(n);
%     A = 0.5 * (A+A');
%     A = A + n * eye(n);
    A = A * A';
    vec_A = reshape(A,[],1);
    X = X + vec_A * vec_A';
end

err = [];
for i = 1: length(m)
    % Generate measurements
    measure_vecs = zeros(n^4, m(i));
    measurements = zeros(m(i),1);
    for j = 1:m(i)
        theta = randn(n,1);
        theta = theta/norm(theta);
        theta = reshape(theta * theta',[],1);
        measure_vecs(:,j) = reshape(theta * theta',[],1);
        vec_X  = reshape(X,[],1);
        measurements(j) = measure_vecs(:,j)' * vec_X;
    end

    % Solve nuclear norm minimization problem
    cvx_begin
        variable Y(n^2,n^2)  symmetric
        Y == semidefinite(n^2);
        minimize trace(Y) %norm_nuc(Y)
        subject to
            measure_vecs' * reshape(Y,[],1) == measurements;
    cvx_end
    disp(['m = ', num2str(m(i)), ' and relative error = ', num2str(norm(X-Y,'fro')/norm(X,'fro'))]);
    err = [err, norm(X-Y,'fro')/norm(X,'fro')];
end
