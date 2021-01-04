% Parameters:
n = 5;                     % dimension of original vector
k = 2;                      % rank
% m =  [20, 50, 100, 200, 500, 1000];                 % number of measurements
m = 1000;
% Generate random 4-tensor(in d^2 by d^2 matrix) of rank k
X = zeros(n^2,n^2);
for i = 1:k
    A = randn(n,n);
    vec_A = reshape(A* A',[],1);
    X = X + vec_A * vec_A';
end
err = [];
for i = 1: length(m)
    % Generate measurements
    measure_vecs = zeros(n^2, m(i));
    measurements = zeros(m(i),1);
    for j = 1:m(i)
        theta = randn(n,1);
        measure_vecs(:,j) = reshape(theta * theta',[],1);
        measurements(j) = measure_vecs(:,j)' * X * measure_vecs(:,j);
    end

    % GD intialization
    learning_rate = 1e-6;
    num_steps = 100000;
    S = zeros(n^2,k);
    for l = 1:k
        A = randn(n,n);
        S(:,l) = reshape(A * A',[],1);
    end
    % GD
    for r = 1:num_steps
%         grad = zeros(n^2,n^2);
%         for j = 1:m(i)
%             grad = grad + (measurements(j) - norm(measure_vecs(:,j)'*S,2)^2) * measure_vecs(:,j) * measure_vecs(:,j)';
%         end
%         grad = grad/m(i);
        grad = measure_vecs * diag(vecnorm(S'*measure_vecs)'.^2 - measurements) * measure_vecs'/ m(i);
        S = S - learning_rate * grad * S;
        if mod(r,500) == 0
            disp(['step = ', num2str(r)]);
            disp(['norm of gradient = ', num2str(norm(grad))]);
        end
        if norm(grad) < 1e-6
            disp(['converged with step = ', num2str(r)]);
            disp(['norm of gradient = ', num2str(norm(grad))]);
            break;
        end
    end
    Y = S*S';
    disp(['m = ', num2str(m(i)), ' and relative error = ', num2str(norm(X-Y,'fro')/norm(X,'fro'))]);
    err = [err, norm(X-Y,'fro')/norm(X,'fro')];
end
