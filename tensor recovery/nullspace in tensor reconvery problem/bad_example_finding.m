b1 = 5*rand();
c1 = 2*randn();
b2 = 5*rand();
c2 = 2*randn();
S1 = [b1;c1;c1;b1];
S2 = [b2;c2;c2;b2];

% A1 = randn(2,2);
% A2 = randn(2,2);
% 
% S1 = reshape(A1 * A1',[],1);
% b1 = S1(1);
% c1 = S1(2);
% S2 = reshape(A2 * A2',[],1);
% b2 = S2(1);
% c2 = S2(2);

X = S1*S1' + S2*S2';

E = [0 0 0 -2; 0 1 1 0; 0 1 1 0; -2 0 0 0];

M = (b1^2 + b2^2 + c1^2 + c2^2)/(b1*c1+b2*c2);

alpha1 = (M + sqrt(M^2-4))/4;
alpha2 = (M - sqrt(M^2-4))/4;

a1 = 2 * (b1*c1+b2*c2) * alpha1 - c1^2 - c2^2;
a2 = 2 * (b1*c1+b2*c2) * alpha2 - c1^2 - c2^2;

Y1 = X + a1*E;
Y2 = X + a2*E;

disp(eig(Y1));
disp(trace(Y1) - trace(X));
disp(eig(Y2));
disp(trace(Y2) - trace(X));
% D1 = det(lambda * eye(4) - Y1);
% D1 = simplify(D1,'Steps',50)

% syms C a alpha
% A = [C/(2*alpha)+a C C C/(2*alpha)-a; C 2*alpha*C 2*alpha*C C;C 2*alpha*C 2*alpha*C C;C/(2*alpha)-a C C C/(2*alpha)+a]
 