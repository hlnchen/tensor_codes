b1 = 3*rand();
b2 = 3*rand();
c1 = randn()+0.5;
S1 = [b1;c1;c1;b2];

d1 = 3*rand();
d2 = 3*rand();
c2 = randn()+0.5;
S2 = [d1;c2;c2;d2];

X = S1 * S1' + S2 * S2';

E = [0 0 0 -2; 0 1 1 0; 0 1 1 0; -2 0 0 0];

K1 = b1^2 + d1^2;
K2 = b1*b2 + d1*d2;
K3 = b1*c1 + d1*c2;
K4 = c1*b2 + c2*d2;
K5 = b2^2 + d2^2;
K6 = c1^2 + c2^2;

syms x;
a_possible = vpasolve(x^3-(K2-K6)*x^2 + (K2^2/4 -K2*K6-K1*K5/4+K3*K4)*x - K2*K3*K4/2 - K1*K5*K6/4 + K1*K4^2/4 + K2^2*K6/4+K3^2*K5/4 == 0 ,x);

for i = 1:3
    Y = X + a_possible(i)*E;
    disp(eig(Y));
    disp(trace(Y) - trace(X));
end
