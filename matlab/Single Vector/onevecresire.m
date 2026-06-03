clear; clc; close all;

% ============================================================
% XMCD COHERENT X-RAY VECTOR TOMOGRAPHY
%
% Physical model:
%   - Coherent X-ray beam with switchable circular polarization
%   - Left/right polarization gives XMCD contrast
%   - Diffraction patterns collected at each tilt angle
%   - Scalar reconstruction from sum signal gives support
%   - Vector reconstruction from difference signal via RESIRE-V
%     gradient descent (no phase retrieval needed for magnetic part)
%
% Forward model (after RESIRE-V eq. 3):
%   I+  = |FFT2D[ integral_z ( O + alpha*Mx + beta*My + gamma*Mz ) dz ]|^2
%   I-  = |FFT2D[ integral_z ( O - alpha*Mx - beta*My - gamma*Mz ) dz ]|^2
%   b-  = 0.5*(I+ - I-)   <- magnetic signal (signed)
%   b+  = 0.5*(I+ + I-)   <- structural signal
%
% Reconstruction:
%   Step 1: scalar RESIRE on b+ to get 3D support
%   Step 2: RESIRE-V gradient descent on b- to get Mx, My, Mz
% ============================================================


%% ===================== USER CONTROLS ========================

% ----- Grid -----
Nx = 32;
Ny = 32;
Nz = 32;

% ----- True single vector value -----
vec_true = [1.0, 0.5, -0.8];   % [vx vy vz]

% ----- Blob center -----
cx = 32; cy = 32; cz = 32;

% ----- Blob width -----
sigma = 3.0;

% ----- Non-magnetic structure amplitude -----
% In real XMCD, O >> M (structural signal much larger than magnetic)
O_amplitude = 5.0;

% ----- X-ray energy / wavelength -----
lambda = 1.0;       % arbitrary units
k0     = 2*pi/lambda;

% ----- Tilt series geometry -----
% Two in-plane rotation angles (phi) each with a tilt series (theta)
% Following RESIRE-V recommendation: phi = 0 and 90 degrees
phi_series_deg   = [0, 90];        % in-plane rotations
theta_range_deg  = -60:12:60;            % tilt range for each series
psi              = 0;                    % X-axis rotation (0 for now)

% ----- Noise -----
add_noise   = true;
photon_flux = 1e7;   % photons per pixel (controls SNR)

% ----- RESIRE scalar controls (for support) -----
scalar_iters   = 100;
scalar_step    = 0.5;
support_thresh = 0.05;   % threshold for support mask (fraction of max)

% ----- RESIRE-V vector controls -----
vector_iters = 150;
t_step       = 1.0;      % normalized step size (RESIRE-V eq. algorithm 1)
eps_reg      = 1e-3;     % regularization

% ----- Display -----
zslice = round(Nz/2);


%% ===================== MAKE TRUE OBJECT =====================

fprintf('Building true object...\n');

[xg,yg,zg] = ndgrid(1:Nx,1:Ny,1:Nz);

blob = exp(-((xg-cx).^2 + (yg-cy).^2 + (zg-cz).^2)/(2*sigma^2));

% Non-magnetic structure
O_true  = O_amplitude * blob;

% Magnetic vector components
mx_true = vec_true(1) * blob;
my_true = vec_true(2) * blob;
mz_true = vec_true(3) * blob;

% True support
support_true = blob > 0.05;


%% ===================== BUILD ANGLE BANK =====================

fprintf('Building angle bank...\n');

phi_list   = deg2rad(phi_series_deg);
theta_list = deg2rad(theta_range_deg);

Nphi   = numel(phi_list);
Ntheta = numel(theta_list);
Nproj  = Nphi * Ntheta;   % total number of projection angles

% For each projection store:
%   R       : 3x3 rotation matrix
%   n_hat   : [alpha, beta, gamma] last column of R^T (beam direction in sample frame)
%   phi,theta,psi : Euler angles

proj_list(Nproj) = struct('R',[],'n_hat',[],'phi',0,'theta',0,'psi',0);

pidx = 0;
for ip = 1:Nphi
    phi_p = phi_list(ip);
    for it = 1:Ntheta
        theta_p = theta_list(it);
        pidx = pidx + 1;

        % Rotation matrices (ZYX convention, RESIRE-V)
        Rz = [cos(phi_p)  -sin(phi_p)  0;
              sin(phi_p)   cos(phi_p)  0;
              0            0           1];

        Ry = [cos(theta_p)   0   sin(theta_p);
              0               1   0;
             -sin(theta_p)   0   cos(theta_p)];

        Rx = [1   0          0;
              0   cos(psi)  -sin(psi);
              0   sin(psi)   cos(psi)];

        R = Rz * Ry * Rx;

        % Normal vector = last column of R^T = last row of R
        % This is the beam direction in sample frame
        % For ZYX: n = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
        n_hat = R(3,:);   % [alpha, beta, gamma]

        proj_list(pidx).R      = R;
        proj_list(pidx).n_hat  = n_hat;
        proj_list(pidx).phi    = phi_p;
        proj_list(pidx).theta  = theta_p;
        proj_list(pidx).psi    = psi;
    end
end

fprintf('Total projections: %d (%d phi x %d theta)\n', Nproj, Nphi, Ntheta);


%% ===================== FORWARD PROJECTION ===================
% Radon transform: integrate along beam direction (z after rotation)
% Uses bilinear interpolation following RESIRE-V algorithm

fprintf('Computing forward projections...\n');

% Storage for measurements
b_plus_meas  = zeros(Nx, Ny, Nproj);   % structural signal
b_minus_meas = zeros(Nx, Ny, Nproj);   % magnetic signal

for pidx = 1:Nproj
    R     = proj_list(pidx).R;
    n_hat = proj_list(pidx).n_hat;   % [alpha, beta, gamma]
    alpha = n_hat(1);
    beta  = n_hat(2);
    gamma = n_hat(3);

    % ----- Compute rotated projections via Radon transform -----
    % For each (x,y) in detector plane, integrate along beam (z direction
    % after rotation R is applied to sample coordinates)

    proj_O  = radon_forward(O_true,  R, Nx, Ny, Nz);
    proj_Mx = radon_forward(mx_true, R, Nx, Ny, Nz);
    proj_My = radon_forward(my_true, R, Nx, Ny, Nz);
    proj_Mz = radon_forward(mz_true, R, Nx, Ny, Nz);

    % ----- Magnetic projection (RESIRE-V eq. 3) -----
    proj_mag = alpha*proj_Mx + beta*proj_My + gamma*proj_Mz;

    % ----- Left and right polarization projections -----
    proj_plus  = proj_O + proj_mag;
    proj_minus = proj_O - proj_mag;

    % ----- Coherent diffraction: 2D FFT of each projection -----
    F_plus  = fftshift(fft2(ifftshift(proj_plus)));
    F_minus = fftshift(fft2(ifftshift(proj_minus)));

    % ----- Intensities -----
    I_plus  = abs(F_plus).^2;
    I_minus = abs(F_minus).^2;

    % ----- Add Poisson noise if requested -----
    if add_noise
        % normalize to photon counts, add Poisson noise, renormalize
        scale   = photon_flux / max(I_plus(:) + I_minus(:));
        I_plus  = poissrnd(I_plus  * scale) / scale;
        I_minus = poissrnd(I_minus * scale) / scale;
    end

    % ----- XMCD signals -----
    b_plus_meas(:,:,pidx)  = 0.5*(I_plus + I_minus);   % structural
    b_minus_meas(:,:,pidx) = 0.5*(I_plus - I_minus);   % magnetic

end

fprintf('Forward projections done.\n\n');


%% ===================== STEP 1: SCALAR RECONSTRUCTION ========
% Use b_plus (structural signal) to reconstruct O and get support
% This uses phase retrieval (HIO/RAAR) on the sum signal
% For now we use a simplified RESIRE gradient descent on the
% real-space projections (assumes we can recover real-space projections
% from diffraction patterns via phase retrieval - simplified here)

fprintf('Step 1: Scalar reconstruction for support...\n');

% For the support estimation we use the back-projected sum signal
% In a real experiment you would run RAAR/HIO on b_plus first
% Here we use a simplified back-projection to estimate support

O_rec = zeros(Nx, Ny, Nz);
step_scalar = scalar_step / (Nproj * Nz);

for iter = 1:scalar_iters
    grad = zeros(Nx, Ny, Nz);

    for pidx = 1:Nproj
        R = proj_list(pidx).R;

        % Forward project current estimate
        proj_est = radon_forward(O_rec, R, Nx, Ny, Nz);

        % Coherent diffraction of estimate
        F_est  = fftshift(fft2(ifftshift(proj_est)));

        % Enforce measured magnitude from b_plus
        % (simplified: use sqrt of b_plus as magnitude target)
        mag_target = sqrt(abs(b_plus_meas(:,:,pidx)));
        F_new  = mag_target .* exp(1i*angle(F_est));

        % Back to real space
        proj_new = real(fftshift(ifft2(ifftshift(F_new))));

        % Residual
        dproj = proj_new - proj_est;

        % Back project residual
        grad = grad + radon_backward(dproj, R, Nx, Ny, Nz);
    end

    % Gradient step
    O_rec = O_rec + step_scalar * grad;

    % Positivity constraint (O is non-negative electron density)
    O_rec = max(O_rec, 0);

    if mod(iter,50) == 0
        fprintf('  Scalar iter %d/%d\n', iter, scalar_iters);
    end
end

% ----- Extract support from scalar reconstruction -----
O_norm   = O_rec / max(O_rec(:));
support  = O_norm > support_thresh;

fprintf('Support voxels: %d / %d\n\n', sum(support(:)), Nx*Ny*Nz);


%% ===================== STEP 2: VECTOR RECONSTRUCTION ========
% RESIRE-V gradient descent on b_minus (magnetic signal)
% Following RESIRE-V Algorithm 1 and eq. 8
%
% Gradient w.r.t. Mx (eq. 8):
%   dE/dMx = sum_theta alpha * Pi^T [ Pi(alpha*Mx + beta*My + gamma*Mz) - b- ]

fprintf('Step 2: Vector reconstruction (RESIRE-V)...\n');

% Step size from RESIRE-V: t / (sqrt(3) * n * Nz)
% where n = number of projections, Nz = thickness in pixels
step_vec = t_step / (sqrt(3) * Nproj * Nz);

Mx_rec = zeros(Nx, Ny, Nz);
My_rec = zeros(Nx, Ny, Nz);
Mz_rec = zeros(Nx, Ny, Nz);

% Track error over iterations
err_hist = zeros(vector_iters, 1);

for iter = 1:vector_iters

    grad_x = zeros(Nx, Ny, Nz);
    grad_y = zeros(Nx, Ny, Nz);
    grad_z = zeros(Nx, Ny, Nz);

    for pidx = 1:Nproj
        R     = proj_list(pidx).R;
        n_hat = proj_list(pidx).n_hat;
        alpha = n_hat(1);
        beta  = n_hat(2);
        gamma = n_hat(3);

        % ----- Forward projection of current M estimate -----
        proj_Mx = radon_forward(Mx_rec, R, Nx, Ny, Nz);
        proj_My = radon_forward(My_rec, R, Nx, Ny, Nz);
        proj_Mz = radon_forward(Mz_rec, R, Nx, Ny, Nz);

        % Magnetic projection
        proj_mag_est = alpha*proj_Mx + beta*proj_My + gamma*proj_Mz;

        % ----- Coherent scattering forward model -----
        % We need to compare against b_minus which came from intensities
        % Use current O estimate for structural part
        proj_O_est = radon_forward(O_rec, R, Nx, Ny, Nz);

        proj_plus_est  = proj_O_est + proj_mag_est;
        proj_minus_est = proj_O_est - proj_mag_est;

        F_plus_est  = fftshift(fft2(ifftshift(proj_plus_est)));
        F_minus_est = fftshift(fft2(ifftshift(proj_minus_est)));

        I_plus_est  = abs(F_plus_est).^2;
        I_minus_est = abs(F_minus_est).^2;

        b_minus_est = 0.5*(I_plus_est - I_minus_est);

        % ----- Residual in measurement space -----
        residual = b_minus_est - b_minus_meas(:,:,pidx);

        % ----- Back project residual to get gradient -----
        % For nonlinear model: linearize around current estimate
        % d(b-)/d(proj_mag) = 2*(F_plus.*conj(F_plus_est) - F_minus.*conj(F_minus_est))
        % Simplified gradient (first order):
        grad_proj = radon_backward(residual, R, Nx, Ny, Nz);

        % Accumulate gradients (RESIRE-V eq. 8)
        grad_x = grad_x + alpha * grad_proj;
        grad_y = grad_y + beta  * grad_proj;
        grad_z = grad_z + gamma * grad_proj;
    end

    % ----- Gradient step -----
    Mx_rec = Mx_rec - step_vec * grad_x;
    My_rec = My_rec - step_vec * grad_y;
    Mz_rec = Mz_rec - step_vec * grad_z;

    % ----- Support constraint -----
    Mx_rec(~support) = 0;
    My_rec(~support) = 0;
    Mz_rec(~support) = 0;

    % ----- Track error -----
    err_hist(iter) = 100 * norm([Mx_rec(:)-mx_true(:); ...
                                  My_rec(:)-my_true(:); ...
                                  Mz_rec(:)-mz_true(:)]) / ...
                          max(norm([mx_true(:);my_true(:);mz_true(:)]),eps);

    if mod(iter,50) == 0
        fprintf('  Vector iter %d/%d, error = %.4f%%\n', ...
                iter, vector_iters, err_hist(iter));
    end
end

fprintf('\n===== Final Reconstruction Errors =====\n');
err_x = 100*norm(Mx_rec(:)-mx_true(:))/max(norm(mx_true(:)),eps);
err_y = 100*norm(My_rec(:)-my_true(:))/max(norm(my_true(:)),eps);
err_z = 100*norm(Mz_rec(:)-mz_true(:))/max(norm(mz_true(:)),eps);
err_v = err_hist(end);
fprintf('Percent error X   = %.4f %%\n', err_x);
fprintf('Percent error Y   = %.4f %%\n', err_y);
fprintf('Percent error Z   = %.4f %%\n', err_z);
fprintf('Percent error ALL = %.4f %%\n', err_v);


%% ===================== FIGURES ==============================

% ----- Convergence curve -----
figure('Name','Convergence','Color','w');
semilogy(err_hist);
xlabel('Iteration'); ylabel('Error (%)');
title('RESIRE-V Convergence');
grid on;

% ----- True vs reconstructed slices -----
figure('Name','True Vector Components','Color','w');
subplot(1,3,1); imagesc(squeeze(mx_true(:,:,zslice))); 
axis image; colorbar; title(sprintf('True m_x'));
subplot(1,3,2); imagesc(squeeze(my_true(:,:,zslice))); 
axis image; colorbar; title(sprintf('True m_y'));
subplot(1,3,3); imagesc(squeeze(mz_true(:,:,zslice))); 
axis image; colorbar; title(sprintf('True m_z'));

figure('Name','Reconstructed Vector Components','Color','w');
subplot(1,3,1); imagesc(squeeze(Mx_rec(:,:,zslice))); 
axis image; colorbar; title(sprintf('Recon m_x, err=%.2f%%',err_x));
subplot(1,3,2); imagesc(squeeze(My_rec(:,:,zslice))); 
axis image; colorbar; title(sprintf('Recon m_y, err=%.2f%%',err_y));
subplot(1,3,3); imagesc(squeeze(Mz_rec(:,:,zslice))); 
axis image; colorbar; title(sprintf('Recon m_z, err=%.2f%%',err_z));

% ----- Error maps -----
figure('Name','Error Maps','Color','w');
subplot(1,3,1); 
imagesc(abs(squeeze(Mx_rec(:,:,zslice)-mx_true(:,:,zslice)))); 
axis image; colorbar; title('|m_x err|');
subplot(1,3,2); 
imagesc(abs(squeeze(My_rec(:,:,zslice)-my_true(:,:,zslice)))); 
axis image; colorbar; title('|m_y err|');
subplot(1,3,3); 
imagesc(abs(squeeze(Mz_rec(:,:,zslice)-mz_true(:,:,zslice)))); 
axis image; colorbar; title('|m_z err|');

% ----- Single vector at center -----
blob_center = blob(cx,cy,cz);
figure('Name','Center Vector Comparison','Color','w');
subplot(1,2,1);
quiver3(cx,cy,cz, mx_true(cx,cy,cz), my_true(cx,cy,cz), mz_true(cx,cy,cz), 1.5);
axis equal tight; grid on;
title(sprintf('True [%.2f, %.2f, %.2f]', vec_true(1), vec_true(2), vec_true(3)));
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1,2,2);
quiver3(cx,cy,cz, Mx_rec(cx,cy,cz), My_rec(cx,cy,cz), Mz_rec(cx,cy,cz), 1.5);
axis equal tight; grid on;
title(sprintf('Recon [%.2f, %.2f, %.2f], err=%.2f%%', ...
    Mx_rec(cx,cy,cz)/blob_center, ...
    My_rec(cx,cy,cz)/blob_center, ...
    Mz_rec(cx,cy,cz)/blob_center, err_v));
xlabel('x'); ylabel('y'); zlabel('z');


%% ===================== HELPER FUNCTIONS =====================

function proj = radon_forward(vol, R, Nx, Ny, Nz)
% Forward Radon transform: project 3D volume onto 2D detector
% using bilinear interpolation following RESIRE-V
%
% Integrates along the z-axis of the rotated coordinate system
% R rotates sample coordinates into beam coordinates

    proj = zeros(Nx, Ny);

    % Center of volume
    cx = (Nx+1)/2;
    cy = (Ny+1)/2;
    cz = (Nz+1)/2;

    % For each detector pixel (u,v), trace ray through volume
    for u = 1:Nx
        for v = 1:Ny
            % Detector coordinates (centered)
            ud = u - cx;
            vd = v - cy;

            % Integrate along beam direction (w axis in rotated frame)
            val = 0;
            for w = 1:Nz
                wd = w - cz;

                % Rotated coordinates back to sample frame
                % r_sample = R^T * [ud, vd, wd]
                rs = R' * [ud; vd; wd];

                % Sample coordinates (1-indexed)
                xs = rs(1) + cx;
                ys = rs(2) + cy;
                zs = rs(3) + cz;

                % Bilinear/trilinear interpolation
                val = val + interp3_safe(vol, xs, ys, zs, Nx, Ny, Nz);
            end
            proj(u,v) = val;
        end
    end
end


function vol = radon_backward(proj, R, Nx, Ny, Nz)
% Transpose of Radon transform: back-project 2D image into 3D volume
% Transpose of radon_forward (adjoint operator)

    vol = zeros(Nx, Ny, Nz);

    cx = (Nx+1)/2;
    cy = (Ny+1)/2;
    cz = (Nz+1)/2;

    for u = 1:Nx
        for v = 1:Ny
            ud = u - cx;
            vd = v - cy;
            pval = proj(u,v);

            for w = 1:Nz
                wd = w - cz;

                % Back rotate to sample frame
                rs = R' * [ud; vd; wd];

                xs = rs(1) + cx;
                ys = rs(2) + cy;
                zs = rs(3) + cz;

                % Distribute value back to nearest voxels
                vol = splat3_safe(vol, xs, ys, zs, pval, Nx, Ny, Nz);
            end
        end
    end
end


function val = interp3_safe(vol, x, y, z, Nx, Ny, Nz)
% Trilinear interpolation with boundary check

    x0 = floor(x); x1 = x0 + 1;
    y0 = floor(y); y1 = y0 + 1;
    z0 = floor(z); z1 = z0 + 1;

    % Check bounds
    if x0 < 1 || x1 > Nx || y0 < 1 || y1 > Ny || z0 < 1 || z1 > Nz
        val = 0;
        return;
    end

    % Fractional parts
    dx = x - x0; dy = y - y0; dz = z - z0;

    % Trilinear interpolation
    val = vol(x0,y0,z0)*(1-dx)*(1-dy)*(1-dz) + ...
          vol(x1,y0,z0)*dx    *(1-dy)*(1-dz) + ...
          vol(x0,y1,z0)*(1-dx)*dy    *(1-dz) + ...
          vol(x0,y0,z1)*(1-dx)*(1-dy)*dz     + ...
          vol(x1,y1,z0)*dx    *dy    *(1-dz) + ...
          vol(x1,y0,z1)*dx    *(1-dy)*dz     + ...
          vol(x0,y1,z1)*(1-dx)*dy    *dz     + ...
          vol(x1,y1,z1)*dx    *dy    *dz;
end


function vol = splat3_safe(vol, x, y, z, val, Nx, Ny, Nz)
% Trilinear splatting: distribute value to 8 neighboring voxels
% (adjoint of trilinear interpolation)

    x0 = floor(x); x1 = x0 + 1;
    y0 = floor(y); y1 = y0 + 1;
    z0 = floor(z); z1 = z0 + 1;

    if x0 < 1 || x1 > Nx || y0 < 1 || y1 > Ny || z0 < 1 || z1 > Nz
        return;
    end

    dx = x - x0; dy = y - y0; dz = z - z0;

    vol(x0,y0,z0) = vol(x0,y0,z0) + val*(1-dx)*(1-dy)*(1-dz);
    vol(x1,y0,z0) = vol(x1,y0,z0) + val*dx    *(1-dy)*(1-dz);
    vol(x0,y1,z0) = vol(x0,y1,z0) + val*(1-dx)*dy    *(1-dz);
    vol(x0,y0,z1) = vol(x0,y0,z1) + val*(1-dx)*(1-dy)*dz;
    vol(x1,y1,z0) = vol(x1,y1,z0) + val*dx    *dy    *(1-dz);
    vol(x1,y0,z1) = vol(x1,y0,z1) + val*dx    *(1-dy)*dz;
    vol(x0,y1,z1) = vol(x0,y1,z1) + val*(1-dx)*dy    *dz;
    vol(x1,y1,z1) = vol(x1,y1,z1) + val*dx    *dy    *dz;
end