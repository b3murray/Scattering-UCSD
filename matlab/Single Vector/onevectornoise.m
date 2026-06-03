clear; clc; close all;

% ------------------------------------------------
% TRUE VECTOR
% ------------------------------------------------
m_true = [9; -1; 3];

figure
quiver3(0,0,0,m_true(1),m_true(2),m_true(3),'LineWidth',2)
grid on
axis equal
title('True Vector')
xlabel('x'); ylabel('y'); zlabel('z')

% ------------------------------------------------
% NOISE CONTROL
% ------------------------------------------------
epsilon = 0;    % measurement noise
lambda  = 0.0; % regularization strength

% ------------------------------------------------
% ANGLE SAMPLING
% ------------------------------------------------
theta_list = linspace(0.2,1.2,8);
phi_list   = linspace(0,2*pi,8);

A = [];
b = [];

% ------------------------------------------------
% BUILD SYSTEM
% ------------------------------------------------
for j = 1:length(phi_list)

    phi = phi_list(j);

    for i = 1:length(theta_list)

        theta = theta_list(i);

        k = [
            sin(theta)*cos(phi)
            sin(theta)*sin(phi)
            cos(theta)
        ];

        s_true  = dot(k,m_true);
        s_noisy = s_true + epsilon*randn;

        A = [A ; k'];
        b = [b ; s_noisy];

    end
end

% ------------------------------------------------
% REGULARIZED SOLUTION
% ------------------------------------------------
m_rec = (A'*A + lambda*eye(3)) \ (A'*b);

disp('True vector:')
disp(m_true)

disp('Recovered vector:')
disp(m_rec)

% ------------------------------------------------
% PLOT RECONSTRUCTED VECTOR
% ------------------------------------------------
figure
quiver3(0,0,0,m_rec(1),m_rec(2),m_rec(3),'LineWidth',2)
grid on
axis equal
title('Recovered Vector')
xlabel('x'); ylabel('y'); zlabel('z')