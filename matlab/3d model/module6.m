% ============================================================
% ============================================================
%   MODULE 6 — RESIRE-V GRADIENT DESCENT REFINEMENT
%   Reconstructs 3D spin vector field from XMCD amplitudes
% ============================================================
%
% PURPOSE:
%   Takes the seed from the algorithm slot (module4) and refines
%   it by minimizing the mismatch between predicted and measured
%   XMCD scattering amplitudes using gradient descent.
%
% FORWARD MODEL (per iteration):
%   For each channel r with beam direction p_r = k_i_hat_r:
%     A_r(q) = sum_z [ p_r · M(q,z) * exp(i*q_z*z) * exp(-z/labs) ]
%     residual_r = |A_r_pred| - |A_r_meas|
%
% GRADIENT (Wirtinger calculus):
%   dE/dMx = sum_r p_r(1) * IFFT[ residual_r * A_r/(|A_r|+eps) ]
%   Same for My, Mz with p_r(2), p_r(3)
%
% OPTIMIZER:
%   Adam optimizer with momentum — fixes the oscillation seen
%   in all four draft codes which used plain gradient descent.
%   Adam adapts the step size per parameter and adds momentum
%   so it doesn't ping-pong across the minimum.
%
% SUPPORT CONSTRAINT:
%   Voxels outside the support mask are zeroed every iteration.
%   Support comes from pipeline.M.Wall_mask or full volume.
%
% INPUTS (from pipeline struct):
%   pipeline.seed.Mx/My/Mz      — initial estimate [Ny x Nx x Nz]
%   pipeline.meas.Amp_minus     — measured amplitudes [Npy x Npx x Nangles]
%   pipeline.meas.b_minus       — signed magnetic signal
%   pipeline.geom.angle_list    — geometry per channel
%   pipeline.sensing.Pmat       — sensing matrix
%   pipeline.M.x/y/z            — true field (for error tracking)
%   pipeline.geom.lambda_nm     — wavelength
%   pipeline.M.z_vec            — depth coordinates (nm)
%   pipeline.geom.absorption_length_nm
%
% OUTPUTS (added to pipeline struct):
%   pipeline.recon.Mx/My/Mz     — reconstructed field [Ny x Nx x Nz]
%   pipeline.recon.err_hist     — amplitude residual per iteration
%   pipeline.recon.ang_hist     — angular error vs true per iteration
%   pipeline.recon.l2_hist      — L2 error vs true per iteration
%   pipeline.recon.method       — 'resire_adam'
%
% FIGURES:
%   Figure 1 — Convergence: amplitude residual + angular error
%   Figure 2 — True vs reconstructed field (middle z-layer)
%   Figure 3 — Angular error map per z-layer
%
% ============================================================

if ~exist('pipeline','var') || ~isfield(pipeline,'seed')
    error('Modules 1-4 must be run first.');
end

fprintf('\n============================================================\n');
fprintf('MODULE 6 — RESIRE-V GRADIENT DESCENT (Adam optimizer)\n');
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 1 — USER CONTROLS
% ============================================================
% ============================================================

% Number of iterations
RES_iters = 300;

% Adam optimizer parameters
adam_lr       = 1e-5;
adam_beta1    = 0.9;
adam_beta2    = 0.999;
adam_eps      = 1e-8;

% Learning rate schedule
lr_schedule      = 'cosine';
lr_step_every    = 100;
lr_min_fraction  = 0.05;

% Support constraint
use_support = false;

% Regularization — L2 weight decay
% Set to 0 when testing with truefield seed (perfect seed, no regularization needed)
% Use 1e-4 for real reconstruction with noisy/imperfect seeds
lambda_reg = 0;

% Small epsilon
epsI = 1e-12;

% Report interval
report_every = 25;

% ============================================================
%   FIGURE TOGGLES — set false to skip individual figures
% ============================================================
show_convergence  = true;   % Fig 1: amplitude residual + angular error
show_comparison   = true;   % Fig 2: true vs reconstructed all z-layers
show_error_maps   = true;   % Fig 3: angular error maps per z-layer

% ============================================================
% ============================================================
%   SECTION 2 — RETRIEVE INPUTS
% ============================================================
% ============================================================

Mx_rec = pipeline.seed.Mx;
My_rec = pipeline.seed.My;
Mz_rec = pipeline.seed.Mz;

% Use b_minus (signed XMCD difference signal) as measurement target
% b_minus = 0.5*(I+ - I-) = 2*Re(A_charge* . A_mag) for weak magnetic signal
% This is what module2 actually computes and stores
b_meas_all = pipeline.meas.b_minus;   % [Npy x Npx x Nangles] — signed, real
N_angles   = pipeline.meas.N_angles;
angle_list = pipeline.geom.angle_list;
Pmat       = pipeline.sensing.Pmat;
z_vec_nm   = pipeline.M.z_vec * pipeline.geom.z_scale_nm;  % physical nm — uses same z_scale_nm as module2
labs       = pipeline.geom.absorption_length_nm;

Mx_true = pipeline.M.x;
My_true = pipeline.M.y;
Mz_true = pipeline.M.z;

[Ny_vol, Nx_vol, Nz_vol] = size(Mx_rec);
% Npy/Npx are determined per-channel from angle_list(r).q_x_map inside the loop.

% Build incidence phase for each channel and z-layer
% phase(u,v,z,r) = exp(i * q_z(u,v,r) * z)
% For efficiency: recompute q_z per channel inside loop

fprintf('RESIRE-V settings:\n');
fprintf('  Iterations    : %d\n', RES_iters);
fprintf('  Optimizer     : Adam (lr=%.1e, b1=%.3f, b2=%.3f)\n', ...
        adam_lr, adam_beta1, adam_beta2);
fprintf('  LR schedule   : %s\n', lr_schedule);
fprintf('  Support       : %s\n', mat2str(use_support));
fprintf('  Regularization: %.1e\n', lambda_reg);
fprintf('  Volume size   : [%d x %d x %d]\n', Ny_vol, Nx_vol, Nz_vol);
fprintf('  Channels      : %d\n\n', N_angles);

% Keep b_minus at native detector resolution
% Module6 forward model predicts b_minus = 2*Re(A_charge* . A_mag)
% and compares directly against measured b_minus from module2
Amp_meas_vol = b_meas_all;   % [Npy x Npx x N_angles] — signed real
fprintf('Detector b_minus size   : %d x %d x %d channels\n', ...
        size(Amp_meas_vol,1), size(Amp_meas_vol,2), N_angles);
fprintf('Volume grid             : %d x %d x %d layers\n', ...
        Ny_vol, Nx_vol, Nz_vol);
fprintf('Sample scale            : %.2f nm per unit\n', pipeline.M.sample_scale_nm);

% Print q-axis range for first angle to verify match with module2
scale_nm_check = pipeline.M.sample_scale_nm;
dx_check = (pipeline.M.x_vec(end)-pipeline.M.x_vec(1))/max(Nx_vol-1,1)*scale_nm_check;
dy_check = (pipeline.M.y_vec(end)-pipeline.M.y_vec(1))/max(Ny_vol-1,1)*scale_nm_check;
qx_check = 2*pi*(-Nx_vol/2:Nx_vol/2-1)/(Nx_vol*dx_check);
qx_det_max = max(abs(angle_list(1).q_x_map(:)));
qx_vol_max = max(abs(qx_check));
fprintf('Volume q-axis range     : qx=[-%.4f, %.4f] rad/nm\n', qx_vol_max, qx_vol_max);
fprintf('Detector q_x range      : [-%.4f, %.4f] rad/nm\n', qx_det_max, qx_det_max);

if qx_vol_max < qx_det_max * 0.8
    fprintf('\nWARNING: Volume q-range is smaller than detector q-range!\n');
    fprintf('         Detector pixels outside volume q-grid return zero.\n');
    recommended = pi / qx_det_max / dx_check * scale_nm_check;
    fprintf('         Recommended sample_scale_nm >= %.1f (currently %.1f)\n\n', ...
            recommended, scale_nm_check);
elseif qx_vol_max > qx_det_max * 3
    fprintf('\nNOTE: Volume q-range much larger than detector — interpolation uses small fraction.\n');
    fprintf('      Consider reducing sample_scale_nm for better resolution.\n\n');
else
    fprintf('Q-ranges overlap well — interpolation should be accurate.\n\n');
end

% Support mask
if use_support && isfield(pipeline.M, 'Wall_mask')
    support = any(pipeline.M.Wall_mask, 3) | ...
              any(pipeline.M.Mask_neel,  3) | ...
              any(pipeline.M.Mask_bloch, 3);
    support = repmat(support, [1, 1, Nz_vol]);
else
    support = true(Ny_vol, Nx_vol, Nz_vol);
end

% ============================================================
%   GRADIENT CHECK — verify forward model self-consistency
%   If seed = true field and noise=false, iteration 1 residual
%   should be near zero. Print a warning if it is not.
% ============================================================
fprintf('Running gradient self-consistency check...\n');
r_check = 1;
p_check = Pmat(r_check,:);
q_x_c = angle_list(r_check).q_x_map;
q_y_c = angle_list(r_check).q_y_map;
q_z_c = angle_list(r_check).q_z_map;
valid_c = angle_list(r_check).valid;
scale_c = pipeline.M.sample_scale_nm;
dx_c = (pipeline.M.x_vec(end)-pipeline.M.x_vec(1))/max(Nx_vol-1,1)*scale_c;
dy_c = (pipeline.M.y_vec(end)-pipeline.M.y_vec(1))/max(Ny_vol-1,1)*scale_c;
qx_c = 2*pi*(-Nx_vol/2:Nx_vol/2-1)/(Nx_vol*dx_c);
qy_c = 2*pi*(-Ny_vol/2:Ny_vol/2-1)/(Ny_vol*dy_c);

A_mag_c = zeros(size(q_x_c));
A_ch_c  = zeros(size(q_x_c));
for kz = 1:Nz_vol
    z_k = z_vec_nm(kz);
    att = exp(-2*z_k/max(labs,1e-12));
    proj_k = p_check(1)*Mx_rec(:,:,kz)+p_check(2)*My_rec(:,:,kz)+p_check(3)*Mz_rec(:,:,kz);
    Fk = fftshift(fft2(proj_k));
    F_det = interp2(qx_c, qy_c(:), Fk, q_x_c, q_y_c, 'linear', 0);
    phase_z = exp(1i*q_z_c*z_k);
    A_mag_c = A_mag_c + att * F_det .* phase_z;
    A_ch_c  = A_ch_c  + att * phase_z;
end
A_mag_c(~valid_c) = 0;
A_ch_c(~valid_c)  = 0;
xmcd_check = pipeline.geom.xmcd_asymmetry;
b_pred_c = 2 * xmcd_check * real(conj(A_ch_c) .* A_mag_c);
b_meas_c = Amp_meas_vol(:,:,r_check);
resid_check = norm(b_pred_c(:)-b_meas_c(:)) / max(norm(b_meas_c(:)),1e-12);
fprintf('Forward model self-check residual: %.6f\n', resid_check);
if resid_check < 0.01
    fprintf('PASS — forward model is self-consistent with measurements.\n\n');
elseif resid_check < 0.1
    fprintf('MARGINAL — small inconsistency (interpolation error).\n\n');
else
    fprintf('FAIL — forward model inconsistency (residual=%.3f).\n', resid_check);
    fprintf('       Check: noise off? sample_scale_nm? q-axes match?\n\n');
end

% ============================================================
%   SECTION 3 — ADAM OPTIMIZER STATE INITIALIZATION
% ============================================================

% First moment (momentum)
m1x = zeros(Ny_vol, Nx_vol, Nz_vol);
m1y = zeros(Ny_vol, Nx_vol, Nz_vol);
m1z = zeros(Ny_vol, Nx_vol, Nz_vol);

% Second moment (adaptive scaling)
m2x = zeros(Ny_vol, Nx_vol, Nz_vol);
m2y = zeros(Ny_vol, Nx_vol, Nz_vol);
m2z = zeros(Ny_vol, Nx_vol, Nz_vol);

% ============================================================
% ============================================================
%   SECTION 4 — RESIRE-V MAIN LOOP
% ============================================================
% ============================================================

err_hist = zeros(RES_iters, 1);
ang_hist = zeros(RES_iters, 1);
l2_hist  = zeros(RES_iters, 1);

fprintf('Running RESIRE-V...\n');
t_res = tic;

for it = 1:RES_iters

    % --------------------------------------------------------
    % Learning rate schedule
    % --------------------------------------------------------
    switch lower(lr_schedule)
        case 'cosine'
            lr_it = adam_lr * (lr_min_fraction + ...
                    0.5*(1-lr_min_fraction) * ...
                    (1 + cos(pi * it / RES_iters)));
        case 'step'
            lr_it = adam_lr * (0.5 ^ floor(it / lr_step_every));
        otherwise
            lr_it = adam_lr;
    end

    % --------------------------------------------------------
    % Gradient accumulation across all channels
    % --------------------------------------------------------
    gradx = zeros(Ny_vol, Nx_vol, Nz_vol);
    grady = zeros(Ny_vol, Nx_vol, Nz_vol);
    gradz = zeros(Ny_vol, Nx_vol, Nz_vol);
    total_err = 0;

    for r = 1:N_angles

        p      = Pmat(r,:);           % [px py pz] sensitivity vector
        k_i    = angle_list(r).k_i;
        k_norm = norm(k_i);
        if k_norm < 1e-12, continue; end

        % --------------------------------------------------------
        % Physically correct forward model:
        %
        % 1. FFT the [Ny_vol x Nx_vol] layer → F(qx,qy) on volume q-grid
        % 2. Interpolate F at the physical detector q-vectors (q_x_map, q_y_map)
        %    from module2 — these are the actual scattering vectors for each pixel
        % 3. Apply depth phase exp(i*q_z(u,v)*z) and absorption
        % 4. Sum over depth → predicted amplitude at each detector pixel
        %
        % Backward pass uses adjoint interpolation (griddata scatter)
        % to back-propagate detector-space gradients to volume q-grid,
        % then IFFT to get real-space volume gradient.
        % --------------------------------------------------------

        % Get physical q maps from module2 (detector geometry)
        q_x_map = angle_list(r).q_x_map;   % [Npy x Npx] rad/nm
        q_y_map = angle_list(r).q_y_map;
        q_z_map = angle_list(r).q_z_map;
        valid_r = angle_list(r).valid;

        Npy_det = size(q_x_map, 1);
        Npx_det = size(q_x_map, 2);

        % Volume q-grid axes in rad/nm — must match module2 exactly
        scale_nm = pipeline.M.sample_scale_nm;
        dx_vol_nm = (pipeline.M.x_vec(end)-pipeline.M.x_vec(1)) / max(Nx_vol-1,1) * scale_nm;
        dy_vol_nm = (pipeline.M.y_vec(end)-pipeline.M.y_vec(1)) / max(Ny_vol-1,1) * scale_nm;
        qx_axis = 2*pi * (-Nx_vol/2 : Nx_vol/2-1) / (Nx_vol * dx_vol_nm);  % rad/nm
        qy_axis = 2*pi * (-Ny_vol/2 : Ny_vol/2-1) / (Ny_vol * dy_vol_nm);  % rad/nm

        % Accumulate predicted magnetic and charge amplitudes
        A_mag_pred    = zeros(Npy_det, Npx_det);
        A_charge_pred = zeros(Npy_det, Npx_det);

        % Store per-layer FFTs for backward pass
        Fk_layers = zeros(Ny_vol, Nx_vol, Nz_vol, 'like', 1i);

        for kz = 1:Nz_vol
            z_k = z_vec_nm(kz);
            att = exp(-2 * z_k / max(labs, 1e-12));

            % Magnetic projection for this layer
            proj_k = p(1)*Mx_rec(:,:,kz) + ...
                     p(2)*My_rec(:,:,kz) + ...
                     p(3)*Mz_rec(:,:,kz);

            % FFT on volume grid
            Fk = fftshift(fft2(proj_k));
            Fk_layers(:,:,kz) = Fk;

            % Interpolate at detector q-vectors
            F_det = interp2(qx_axis, qy_axis(:), Fk, ...
                            q_x_map, q_y_map, 'linear', 0);

            % Depth phase
            phase_z = exp(1i * q_z_map * z_k);

            A_mag_pred    = A_mag_pred    + att * F_det .* phase_z;
            A_charge_pred = A_charge_pred + att * phase_z;
        end

        % Zero evanescent pixels
        A_mag_pred(~valid_r)    = 0;
        A_charge_pred(~valid_r) = 0;

        % Predicted b_minus = 2 * xmcd * Re(conj(A_charge) .* A_mag)
        % This matches module2: b_minus = 0.5*(I+ - I-)
        %   = 0.5*(|A_ch + xmcd*A_mag|^2 - |A_ch - xmcd*A_mag|^2)
        %   = 2*xmcd*Re(conj(A_charge).*A_mag)
        xmcd = pipeline.geom.xmcd_asymmetry;
        b_pred = 2 * xmcd * real(conj(A_charge_pred) .* A_mag_pred);

        % Measured b_minus for this channel
        b_meas = Amp_meas_vol(:,:,r);

        % Residual in b_minus space (linear — no amplitude loss problem)
        residual = b_pred - b_meas;
        total_err = total_err + mean(residual(:).^2);

        % Gradient of ||b_pred - b_meas||^2 w.r.t. A_mag
        % d/d(A_mag): b_pred = 2*xmcd*Re(conj(A_ch)*A_mag)
        %           = xmcd*(conj(A_ch)*A_mag + A_ch*conj(A_mag))
        % d(b_pred)/d(A_mag) = xmcd * conj(A_charge)  [Wirtinger]
        % full gradient: 2 * residual * xmcd * conj(A_charge)
        dA_mag = 2 * xmcd * residual .* conj(A_charge_pred);
        dA_mag(~valid_r) = 0;

        % --------------------------------------------------------
        % Backward pass: scatter dA_mag onto volume q-grid, IFFT
        % --------------------------------------------------------
        for kz = 1:Nz_vol
            z_k = z_vec_nm(kz);
            att = exp(-2 * z_k / max(labs, 1e-12));

            % Conjugate depth phase
            phase_z = exp(1i * q_z_map * z_k);
            dA_layer = conj(phase_z) .* dA_mag * att;

            % Adjoint of interp2 — scatter onto volume q-grid
            dqx = qx_axis(2) - qx_axis(1);
            dqy = qy_axis(2) - qy_axis(1);

            ix = round((q_x_map - qx_axis(1)) / dqx) + 1;
            iy = round((q_y_map - qy_axis(1)) / dqy) + 1;

            ix = max(1, min(Nx_vol, ix));
            iy = max(1, min(Ny_vol, iy));

            mask_v = valid_r(:);
            ix_v   = ix(mask_v);
            iy_v   = iy(mask_v);
            val_v  = dA_layer(mask_v);

            lin_idx = sub2ind([Ny_vol, Nx_vol], iy_v, ix_v);
            dFk = accumarray(lin_idx, val_v, [Ny_vol*Nx_vol, 1], @sum, 0+0i);
            dFk = reshape(dFk, Ny_vol, Nx_vol);

            % IFFT back to real space
            back = real(ifft2(ifftshift(dFk)));

            % Distribute to spin components
            gradx(:,:,kz) = gradx(:,:,kz) + p(1) * back;
            grady(:,:,kz) = grady(:,:,kz) + p(2) * back;
            gradz(:,:,kz) = gradz(:,:,kz) + p(3) * back;
        end

    end

    % L2 regularization gradient
    gradx = gradx + lambda_reg * Mx_rec;
    grady = grady + lambda_reg * My_rec;
    gradz = gradz + lambda_reg * Mz_rec;

    % --------------------------------------------------------
    % Adam update — skip if gradient is negligible
    % (prevents numerical instability when residual is near zero)
    % --------------------------------------------------------
    gnorm = sqrt(mean(gradx(:).^2) + mean(grady(:).^2) + mean(gradz(:).^2));

    if gnorm < 1e-15
        % Gradient is at machine precision — no meaningful update to make
        % This happens when the seed already perfectly satisfies measurements
        if mod(it, report_every) == 0
            fprintf('  it %4d/%d | gradient at machine precision — skipping update\n', it, RES_iters);
        end
    else
        m1x = adam_beta1*m1x + (1-adam_beta1)*gradx;
        m1y = adam_beta1*m1y + (1-adam_beta1)*grady;
        m1z = adam_beta1*m1z + (1-adam_beta1)*gradz;

        m2x = adam_beta2*m2x + (1-adam_beta2)*gradx.^2;
        m2y = adam_beta2*m2y + (1-adam_beta2)*grady.^2;
        m2z = adam_beta2*m2z + (1-adam_beta2)*gradz.^2;

        % Bias correction
        bc1 = 1 - adam_beta1^it;
        bc2 = 1 - adam_beta2^it;

        m1x_hat = m1x / bc1;  m2x_hat = m2x / bc2;
        m1y_hat = m1y / bc1;  m2y_hat = m2y / bc2;
        m1z_hat = m1z / bc1;  m2z_hat = m2z / bc2;

        Mx_rec = Mx_rec - lr_it * m1x_hat ./ (sqrt(m2x_hat) + adam_eps);
        My_rec = My_rec - lr_it * m1y_hat ./ (sqrt(m2y_hat) + adam_eps);
        Mz_rec = Mz_rec - lr_it * m1z_hat ./ (sqrt(m2z_hat) + adam_eps);
    end

    % --------------------------------------------------------
    % Support constraint
    % --------------------------------------------------------
    if use_support
        Mx_rec(~support) = 0;
        My_rec(~support) = 0;
        Mz_rec(~support) = 0;
    end

    % --------------------------------------------------------
    % Track errors
    % --------------------------------------------------------
    err_hist(it) = total_err;

    epsv   = 1e-12;
    rel_L2 = sqrt(norm(Mx_rec(:)-Mx_true(:))^2 + ...
                  norm(My_rec(:)-My_true(:))^2 + ...
                  norm(Mz_rec(:)-Mz_true(:))^2) / ...
             (sqrt(norm(Mx_true(:))^2+norm(My_true(:))^2+norm(Mz_true(:))^2)+epsv);
    l2_hist(it) = 100 * rel_L2;

    dotp   = Mx_true.*Mx_rec + My_true.*My_rec + Mz_true.*Mz_rec;
    normT  = sqrt(Mx_true.^2+My_true.^2+Mz_true.^2) + epsv;
    normR  = sqrt(Mx_rec.^2 +My_rec.^2 +Mz_rec.^2)  + epsv;
    cosang = max(min(dotp./(normT.*normR),1),-1);
    ang_hist(it) = mean(acosd(cosang(:)), 'omitnan');

    if it == 1 || mod(it, report_every) == 0
        fprintf('  it %4d/%d | amp_err=%.4e | L2=%.2f%% | ang=%.2f deg | lr=%.2e\n', ...
                it, RES_iters, total_err, l2_hist(it), ang_hist(it), lr_it);
    end

end

fprintf('\nRESIRE-V complete. Time: %.1f seconds\n\n', toc(t_res));

% ============================================================
% ============================================================
%   SECTION 5 — FINAL ERROR REPORT
% ============================================================
% ============================================================

z_check  = round(Nz_vol/2);
mx_true_z = Mx_true(:,:,z_check);
my_true_z = My_true(:,:,z_check);
mz_true_z = Mz_true(:,:,z_check);
mx_rec_z  = Mx_rec(:,:,z_check);
my_rec_z  = My_rec(:,:,z_check);
mz_rec_z  = Mz_rec(:,:,z_check);

epsv = 1e-12;
dotp_f  = mx_true_z.*mx_rec_z + my_true_z.*my_rec_z + mz_true_z.*mz_rec_z;
normT_f = sqrt(mx_true_z.^2+my_true_z.^2+mz_true_z.^2)+epsv;
normR_f = sqrt(mx_rec_z.^2 +my_rec_z.^2 +mz_rec_z.^2) +epsv;
ang_map = acosd(max(min(dotp_f./(normT_f.*normR_f),1),-1));

fprintf('============================================================\n');
fprintf('FINAL RECONSTRUCTION ERRORS\n');
fprintf('Seed method        : %s\n', pipeline.seed.method);
fprintf('============================================================\n');
fprintf('Relative L2 error  : %.2f %%\n', l2_hist(end));
fprintf('Mean angular error : %.2f deg\n', ang_hist(end));
fprintf('Median ang. error  : %.2f deg\n', median(ang_map(:),'omitnan'));
fprintf('Amplitude residual : %.4e\n', err_hist(end));
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 6 — PACK PIPELINE STRUCT
% ============================================================
% ============================================================

pipeline.recon.Mx       = Mx_rec;
pipeline.recon.My       = My_rec;
pipeline.recon.Mz       = Mz_rec;
pipeline.recon.err_hist = err_hist;
pipeline.recon.ang_hist = ang_hist;
pipeline.recon.l2_hist  = l2_hist;
pipeline.recon.method   = 'resire_adam';
pipeline.recon.seed_method = pipeline.seed.method;

% ============================================================
% ============================================================
%   SECTION 7 — FIGURES
% ============================================================
% ============================================================

step_q = max(1, round(Ny_vol/18));
[Xg, Yg] = meshgrid(pipeline.M.x_vec, pipeline.M.y_vec);
Xs = Xg(1:step_q:end,1:step_q:end);
Ys = Yg(1:step_q:end,1:step_q:end);
Zs = zeros(size(Xs));

% ---- Figure 1: Convergence curves ----
if show_convergence
    figure('Units','normalized','OuterPosition',[0.02 0.30 0.43 0.50],'Color','w');

    subplot(2,1,1);
    semilogy(err_hist, 'b-', 'LineWidth', 2);
    xlabel('Iteration','FontSize',10);
    ylabel('Amplitude residual','FontSize',10);
    title(sprintf('RESIRE-V Convergence  |  seed: %s', pipeline.seed.method),'FontSize',11);
    grid on;

    subplot(2,1,2);
    plot(ang_hist, 'r-', 'LineWidth', 2); hold on;
    plot(l2_hist,  'b-', 'LineWidth', 2);
    yline(90, 'k--', 'LineWidth', 1);
    text(5, 92, 'random baseline (90°)', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
    legend('Mean angular error (deg)', 'L2 error (%)', 'Location','northeast','FontSize',9);
    xlabel('Iteration','FontSize',10);
    ylabel('Error','FontSize',10);
    title('Reconstruction Error vs True Field','FontSize',11);
    grid on; hold off;
end

% ---- Figure 2: True vs reconstructed — ALL z-layers ----
if show_comparison
    figure('Units','normalized','OuterPosition',[0.02 0.02 0.95 0.45],'Color','w');

    for kz = 1:Nz_vol

        mx_t = Mx_true(:,:,kz); my_t = My_true(:,:,kz); mz_t = Mz_true(:,:,kz);
        mx_r = Mx_rec(:,:,kz);  my_r = My_rec(:,:,kz);  mz_r = Mz_rec(:,:,kz);

        % Angular error for this layer
        epsv2  = 1e-12;
        dotp_k = mx_t.*mx_r + my_t.*my_r + mz_t.*mz_r;
        nT_k   = sqrt(mx_t.^2+my_t.^2+mz_t.^2)+epsv2;
        nR_k   = sqrt(mx_r.^2+my_r.^2+mz_r.^2)+epsv2;
        ang_k  = mean(acosd(max(min(dotp_k./(nT_k.*nR_k),1),-1)),'all','omitnan');

        % True field — top row
        subplot(2, Nz_vol, kz); hold on;
        mxt = mx_t(1:step_q:end,1:step_q:end);
        myt = my_t(1:step_q:end,1:step_q:end);
        mzt = mz_t(1:step_q:end,1:step_q:end);
        quiver3(Xs,Ys,Zs,mxt,myt,mzt,0.6,'b','LineWidth',1.2);
        grid on; axis equal; view(35,25);
        xlabel('X'); ylabel('Y'); zlabel('Mz');
        title(sprintf('TRUE  z-layer %d  (z=%.1f)', kz, z_vec_nm(kz)),'FontSize',9);
        hold off;

        % Reconstructed — bottom row
        subplot(2, Nz_vol, kz + Nz_vol); hold on;
        mxr = mx_r(1:step_q:end,1:step_q:end);
        myr = my_r(1:step_q:end,1:step_q:end);
        mzr = mz_r(1:step_q:end,1:step_q:end);
        quiver3(Xs,Ys,Zs,mxr,myr,mzr,0.6,'r','LineWidth',1.2);
        grid on; axis equal; view(35,25);
        xlabel('X'); ylabel('Y'); zlabel('Mz');
        title(sprintf('RECON z-layer %d  ang=%.1f°', kz, ang_k),'FontSize',9);
        hold off;
    end

    sgtitle(sprintf('True (blue) vs Reconstructed (red)  |  %d z-layers  |  seed: %s  |  L2=%.1f%%  ang=%.1f°', ...
            Nz_vol, pipeline.seed.method, l2_hist(end), ang_hist(end)), 'FontSize',11);
end

% ---- Figure 3: Angular error maps — all z-layers ----
if show_error_maps
    figure('Units','normalized','OuterPosition',[0.52 0.30 0.45 0.28],'Color','w');

    for kz = 1:Nz_vol
        subplot(1, Nz_vol, kz);
        mx_t = Mx_true(:,:,kz); my_t = My_true(:,:,kz); mz_t = Mz_true(:,:,kz);
        mx_r = Mx_rec(:,:,kz);  my_r = My_rec(:,:,kz);  mz_r = Mz_rec(:,:,kz);
        epsv2 = 1e-12;
        dotp_k  = mx_t.*mx_r + my_t.*my_r + mz_t.*mz_r;
        normT_k = sqrt(mx_t.^2+my_t.^2+mz_t.^2)+epsv2;
        normR_k = sqrt(mx_r.^2+my_r.^2+mz_r.^2)+epsv2;
        ang_k   = acosd(max(min(dotp_k./(normT_k.*normR_k),1),-1));
        imagesc(ang_k); axis image; colorbar;
        colormap(gca,'hot'); clim([0 90]);
        title(sprintf('z=%d  mean=%.1f°', kz, mean(ang_k(:),'omitnan')),'FontSize',9);
        xlabel('x'); ylabel('y');
    end
    sgtitle('Angular Error (deg)  |  0°=perfect  90°=random','FontSize',10);
end

% ============================================================

fprintf('--- Module 6 complete ---\n');
fprintf('pipeline.recon populated.\n');
fprintf('  Method  : %s\n', pipeline.recon.method);
fprintf('  Seed    : %s\n', pipeline.recon.seed_method);
fprintf('  L2      : %.2f %%\n', l2_hist(end));
fprintf('  Angle   : %.2f deg\n', ang_hist(end));
fprintf('Pass pipeline to module8.m (error analysis).\n\n');