syms m n e tau B real

% Define sigma_0 = m/(n e^2 tau)
rho0 = m/(n*e^2*tau);

% Resistivity matrix in symbolic form
rho = [ rho0,      B/(n*e);
       -B/(n*e),   rho0 ];

disp('----------------------------------------');
disp('Symbolic Resistivity Matrix ρ:');
pretty(rho)

% Symbolic diagonalization
[V, D] = eig(rho);

disp('----------------------------------------');
disp('Symbolic Eigenvalues (diagonal matrix D):');
pretty(D)

disp('----------------------------------------');
disp('Symbolic Eigenvectors (columns of V):');
pretty(V)

disp('----------------------------------------');
