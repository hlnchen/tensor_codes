syms a b c d e f real
% E = [0 0 0 -0.5*a -d -0.5*b;0 a d 0 e f;0 d b -e -f 0;-0.5*a 0 -e 0 0 -0.5*c;-d e -f 0 c 0;-0.5*b f 0 -0.5*c 0 0];

E1 = [ 0 0 0 0 -2*a 0 0 0 0; 0 a 0 a 0 0 0 0 0;0 0 0 0 0 0 0 0 0; 0 a 0 a 0 0 0 0 0; -2*a 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0];

E2 = [ 0 0 0 0 0 0 0 0 -2*b; 0 0 0 0 0 0 0 0 0;0 0 b 0 0 0 b 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 b 0 0 0 b 0 0; 0 0 0 0 0 0 0 0 0; -2*b 0 0 0 0 0 0 0 0];

E3 = [ 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 -2*c; 0 0 0 0 0 c 0 c 0; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 c 0 c 0; 0 0 0 0 -2*c 0 0 0 0];

E4 = [ 0 0 0 0 0 -2*d 0 -2*d 0; 0 0 d 0 0 0 d 0 0;0 d 0 d 0 0 0 0 0; 0 0 d 0 0 0 d 0 0; 0 0 0 0 0 0 0 0 0; -2*d 0 0 0 0 0 0 0 0; 0 d 0 d 0 0 0 0 0; -2*d 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0];

E5 = [ 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 e 0 e 0;0 0 0 0 -2*e 0 0 0 0; 0 0 0 0 0 e 0 e 0; 0 0 -2*e 0 0 0 -2*e 0 0; 0 e 0 e 0 0 0 0 0; 0 0 0 0 -2*e 0 0 0 0; 0 e 0 e 0 0 0 0 0; 0 0 0 0 0 0 0 0 0];

E6 = [ 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 -2*f;0 0 0 0 0 f 0 f 0; 0 0 0 0 0 0 0 0 -2*f; 0 0 0 0 0 0 0 0 0; 0 0 f 0 0 0 f 0 0; 0 0 0 0 0 f 0 f 0; 0 0 f 0 0 0 f 0 0; 0 -2*f 0 -2*f 0 0 0 0 0];

N = {E1,E2,E3,E4,E5,E6};

E = zeros(9,9,'sym');

F = cell(1,6);

for i = 1:6
    E = E + N{i};
    F{i} = matlabFunction(N{i});
end

FE = matlabFunction(E);

% d=4 case
% EE = zeros(16,16,'sym');
% EE(2,12) = a;EE(2,15) =a;EE(5,12) = a;EE(5,15) = a; EE(12,2) =a;EE(12,5)=a;EE(15,2)=a;EE(15,5)=a
% EE(3,8) =b;EE(3,14) =b;EE(9,8) =b;EE(9,14) =b;EE(8,3) =b; EE(8,9) =b;EE(14,3) =b;EE(14,9) =b
% EE(4,7) = -a-b; EE(4,10) = -a-b;EE(13,7) = -a-b;EE(13,10) = -a-b;EE(7,4) = -a-b;EE(7,13) = -a-b;EE(10,4) = -a-b;EE(10,13) = -a-b
