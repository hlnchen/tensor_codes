% b1 = 3*rand();
% b2 = 3*rand();
% c1 = randn()+0.5;
% S1 = [b1;c1;c1;b2];
% 
% d1 = 3*rand();
% d2 = 3*rand();
% c2 = randn()+0.5;
% S2 = [d1;c2;c2;d2];
% 
% X = S1 * S1' + S2 * S2';

k = 3;
d = 3;
X = zeros(d^2,d^2);
for i = 1:k
    A = randn(d,d);
    A = A*A';
    AA = reshape(A,[],1);
    X = X + AA*AA';
end

% E = [0 0 0 -2; 0 1 1 0; 0 1 1 0; -2 0 0 0];

E = [0 0 0 0 -2 0 0 0 2;0 1 0 1 0 0 0 0 0;
0 0 -1 0 0 0 -1 0 0;0 1 0 1 0 0 0 0 0;
-2 0 0 0 0 0 0 0 0;
0 0 0 0 0 0 0 0 0;0 0 -1 0 0 0 -1 0 0;
0 0 0 0 0 0 0 0 0;2 0 0 0 0 0 0 0 0];

t = -5:0.001:5;

rank_X = [];
trace_X = [];
norm_X = [];
eigmin = [];

for i = 1:length(t)
    norm_X = [norm_X, norm(X+t(i)*E)];
    rank_X = [rank_X, rank(X+t(i)*E,1e-8)];
    trace_X = [trace_X, trace(X+t(i)*E)];
    eigmin = [eigmin, min(nonzeros(eig(X+t(i)*E)))];
end

% plot(t,rank_X);
plot(t, eigmin,'-*', t, rank_X,'-o', t, trace_X,'-+');
legend("eigmin","rank", "trace");