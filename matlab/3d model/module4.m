% ============================================================
% ============================================================
%   ALGORITHM SLOT — RAAR PHASE RETRIEVAL
%   Relaxed Averaged Alternating Reflections → LS Seed
% ============================================================
%
% PURPOSE:
%   Takes the measured XMCD diffraction amplitudes from Module 2
%   and recovers the complex exit-wave for each measurement channel
%   using RAAR phase retrieval. Then solves the per-pixel least
%   squares system using the sensing matrix from Module 3 to
%   produce an initial estimate of [Mx, My, Mz] — the seed that
%   gets passed to RESIRE-V in Module 6.
%
% WHAT RAAR DOES:
%   For each measurement channel r, we know |A_r(q)| (the amplitude)
%   but not the phase. RAAR alternates between two projections:
%     Pm — enforces measured Fourier amplitude
%     Ps — enforces real-space support constraint
%   The RAAR update rule is:
%     x_{k+1} = (beta/2)(R_S R_M x_k + x_k) + (1-beta) P_M x_k
%   where R_S, R_M are reflections about the support and magnitude sets.
%
% IMPORTANT NOTE ON ALGORITHM CHOICE:
%   RAAR is one option in the algorithm slot. It can be replaced
%   by algo_cnn.m (CNN prediction) or algo_ls_direct.m (direct LS)
%   without changing any other module. The only requirement is that
%   this script populates pipeline.seed with Mx_seed, My_seed, Mz_seed.
%
% INPUTS (from pipeline struct):
%   pipeline.meas.Amp_minus     — magnetic amplitudes [Npx x Npy x Nangles]
%   pipeline.meas.b_minus       — signed magnetic signal
%   pipeline.geom.angle_list    — angle geometry
%   pipeline.sensing.Pmat       — sensing matrix [Nangles x 3]
%   pipeline.sensing.Pinv       — pseudoinverse [3 x Nangles]
%   pipeline.M.x/y/z            — true field (for convergence monitoring)
%
% OUTPUTS (added to pipeline struct):
%   pipeline.seed.Mx            — [Ny x Nx x Nz] initial Mx estimate
%   pipeline.seed.My            — [Ny x Nx x Nz] initial My estimate
%   pipeline.seed.Mz            — [Ny x Nx x Nz] initial Mz estimate
%   pipeline.seed.method        — 'raar'
%   pipeline.seed.raar_err      — convergence history per channel
%   pipeline.seed.ls_err        — LS residual after seed solve
%
% FIGURES:
%   Figure 1 — RAAR convergence curves (all channels)
%   Figure 2 — Seed vector field vs true field (middle z-layer)
%   Figure 3 — Per-channel amplitude residual at convergence
%
% ============================================================

% ============================================================
%   CHECK PREREQUISITES
% ============================================================
if ~exist('pipeline','var') || ~isfield(pipeline,'sensing')
    error('Modules 1, 2, and 3 must be run first.');
end

fprintf('\n============================================================\n');
fprintf('ALGORITHM SLOT — RAAR PHASE RETRIEVAL\n');
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 1 — RAAR USER CONTROLS
% ============================================================
% ============================================================

% Number of RAAR iterations per channel
% More iterations = better phase convergence but slower
% Recommended: 200-400 for good results, 50 for quick test
RAAR_iters = 300;

% Relaxation parameter beta [0.5 .. 1.0]
% Higher beta = more aggressive updates, faster but less stable
% Lower beta = more conservative, slower but more reliable
% Recommended: 0.87 (standard value in CDI literature)
beta_raar  = 0.87;

% Shrink-wrap support refinement
% Dynamically tightens the support mask during RAAR
% Strongly recommended — prevents solution spreading into empty space
use_shrinkwrap  = true;
sw_every        = 25;      % update support every N iterations
sw_sigma        = 2.0;     % Gaussian blur sigma for shrink-wrap
sw_thresh       = 0.15;    % threshold as fraction of max blurred magnitude

% Support initialization
%   'full'    -> start with full support (no prior knowledge)
%   'tight'   -> start with a tight support from b+ structural signal
support_init = 'full';

% Progress reporting interval
report_every = 50;

% ---- Figure toggles ----
show_raar_convergence = true;   % Fig 1: RAAR convergence per channel
show_seed_comparison  = true;   % Fig 2: seed vs true field
show_ls_residual      = true;   % Fig 3: per-channel LS residual

% ============================================================
% ============================================================
%   SECTION 2 — RETRIEVE INPUTS
% ============================================================
% ============================================================

Amp_minus  = pipeline.meas.Amp_minus;   % signed amplitudes
b_minus    = pipeline.meas.b_minus;
N_angles   = pipeline.meas.N_angles;
Pmat       = pipeline.sensing.Pmat;
Pinv       = pipeline.sensing.Pinv;

% Volume dimensions from Module 1
Ny_vol = size(pipeline.M.x, 1);
Nx_vol = size(pipeline.M.x, 2);
Nz_vol = size(pipeline.M.x, 3);

% Detector dimensions
Npy = size(Amp_minus, 1);
Npx = size(Amp_minus, 2);

fprintf('RAAR settings:\n');
fprintf('  Iterations per channel : %d\n', RAAR_iters);
fprintf('  Beta                   : %.3f\n', beta_raar);
fprintf('  Shrink-wrap            : %s (every %d iters, sigma=%.1f, thresh=%.2f)\n', ...
        mat2str(use_shrinkwrap), sw_every, sw_sigma, sw_thresh);
fprintf('  Total channels         : %d\n', N_angles);
fprintf('  Detector size          : %d x %d pixels\n\n', Npy, Npx);

% ============================================================
% ============================================================
%   SECTION 3 — RAAR PHASE RETRIEVAL PER CHANNEL
% ============================================================
%
% Each channel gives us |A_r(q)|. We recover the complex A_r(x,y)
% in real space. The real part of this gives us the projected
% magnetic exit wave for that beam direction.
% ============================================================

% Storage for real-space reconstructions
a_realspace = zeros(Npy, Npx, N_angles);
raar_err    = zeros(RAAR_iters, N_angles);

fprintf('Running RAAR on %d channels...\n', N_angles);
t_raar = tic;

for r = 1:N_angles

    % Measured amplitude for this channel
    % Use absolute value of signed amplitude — RAAR works on magnitudes
    Amp = abs(Amp_minus(:,:,r));

    % Initialize support
    switch lower(support_init)
        case 'tight'
            % Use structural signal to initialize support
            b_plus_r = pipeline.meas.b_plus(:,:,r);
            support  = b_plus_r > (sw_thresh * max(b_plus_r(:)));
        otherwise
            support = true(Npy, Npx);
    end

    % Random phase initialization in Fourier domain
    phi0 = 2*pi * rand(Npy, Npx);
    F    = Amp .* exp(1i * phi0);
    xr   = ifft2(ifftshift(F));

    % RAAR iterations
    for it = 1:RAAR_iters

        % Fourier magnitude projection (Pm)
        Fk      = fftshift(fft2(xr));
        Fk_proj = Amp .* exp(1i * angle(Fk));
        Pm      = ifft2(ifftshift(Fk_proj));

        % Support projection and reflections
        RM            = 2*Pm - xr;
        RSRM          = RM;
        RSRM(~support) = -RM(~support);

        % RAAR update
        xr = (beta_raar/2) * (RSRM + xr) + (1 - beta_raar) * Pm;

        % Shrink-wrap support update
        if use_shrinkwrap && mod(it, sw_every) == 0
            mag_blur = imgaussfilt(abs(xr), sw_sigma);
            support  = mag_blur > (sw_thresh * max(mag_blur(:)));
        end

        % Track convergence: Fourier amplitude residual
        Fk_new = fftshift(fft2(xr));
        residual = abs(Fk_new) - Amp;
        raar_err(it,r) = norm(residual(:)) / max(norm(Amp(:)), 1e-12);

    end

    % Store real part — the magnetic projected amplitude
    a_realspace(:,:,r) = real(xr);

    % Progress report
    if mod(r, max(1, round(N_angles/10))) == 0 || r == N_angles
        fprintf('  Channel %3d / %d  |  final err = %.4f\n', ...
                r, N_angles, raar_err(end,r));
    end

end

fprintf('RAAR complete. Time: %.1f seconds\n\n', toc(t_raar));

% ============================================================
% ============================================================
%   SECTION 4 — LEAST SQUARES SEED
%
%   For each pixel, solve: Pmat * [mx; my; mz] = a_vec
%   where a_vec = [a_1(x,y), a_2(x,y), ..., a_nR(x,y)]^T
%   Solution: m = Pinv * a_vec
%
%   This gives us one 2D seed per (x,y) pixel.
%   For a 3D volume we solve independently per z-layer
%   using the RAAR reconstruction at each depth.
% ============================================================

fprintf('Building LS seed from RAAR reconstructions...\n');

% For 3D: distribute RAAR results across z-layers
% Current approximation: all channels contribute to all z-layers
% equally (depth-resolved LS is Stage 1 roadmap item)
% For now: build one 2D seed then replicate across z

Mx_seed_2d = zeros(Npy, Npx);
My_seed_2d = zeros(Npy, Npx);
Mz_seed_2d = zeros(Npy, Npx);

% Solve row by row for efficiency
for iy = 1:Npy
    % Stack all channel values for this row: [N_angles x Npx]
    a_row = squeeze(a_realspace(iy,:,:)).';   % [N_angles x Npx]

    % LS solve: [3 x Npx] = [3 x N_angles] * [N_angles x Npx]
    m_row = Pinv * a_row;   % [3 x Npx]

    Mx_seed_2d(iy,:) = m_row(1,:);
    My_seed_2d(iy,:) = m_row(2,:);
    Mz_seed_2d(iy,:) = m_row(3,:);
end

% Replicate 2D seed across all z-layers
% In a full depth-resolved pipeline this would differ per layer
Mx_seed = repmat(Mx_seed_2d, [1, 1, Nz_vol]);
My_seed = repmat(My_seed_2d, [1, 1, Nz_vol]);
Mz_seed = repmat(Mz_seed_2d, [1, 1, Nz_vol]);

% Crop or pad seed to match volume dimensions if detector != volume
if Npy ~= Ny_vol || Npx ~= Nx_vol
    fprintf('Note: Detector size (%dx%d) differs from volume size (%dx%d)\n', ...
            Npy, Npx, Ny_vol, Nx_vol);
    fprintf('      Resampling seed to match volume grid...\n');
    Mx_seed_r = zeros(Ny_vol, Nx_vol, Nz_vol);
    My_seed_r = zeros(Ny_vol, Nx_vol, Nz_vol);
    Mz_seed_r = zeros(Ny_vol, Nx_vol, Nz_vol);
    for kz = 1:Nz_vol
        Mx_seed_r(:,:,kz) = imresize(Mx_seed(:,:,kz), [Ny_vol, Nx_vol]);
        My_seed_r(:,:,kz) = imresize(My_seed(:,:,kz), [Ny_vol, Nx_vol]);
        Mz_seed_r(:,:,kz) = imresize(Mz_seed(:,:,kz), [Ny_vol, Nx_vol]);
    end
    Mx_seed = Mx_seed_r;
    My_seed = My_seed_r;
    Mz_seed = Mz_seed_r;
end

% LS residual — how well does the seed fit the measurements
ls_residual = zeros(N_angles, 1);
for r = 1:N_angles
    p    = Pmat(r,:);
    proj = p(1)*Mx_seed_2d + p(2)*My_seed_2d + p(3)*Mz_seed_2d;
    a_r  = a_realspace(:,:,r);
    ls_residual(r) = norm(proj(:) - a_r(:), 'fro') / ...
                     max(norm(a_r(:)), 1e-12);
end

fprintf('LS seed built.\n');
fprintf('Mean LS residual across channels: %.4f\n\n', mean(ls_residual));

% ============================================================
% ============================================================
%   SECTION 5 — QUICK SEED QUALITY CHECK VS TRUE FIELD
% ============================================================
% ============================================================

% Compare seed to true field (middle z-layer) for diagnostics
z_check = round(Nz_vol / 2);
mx_true = pipeline.M.x(:,:,z_check);
my_true = pipeline.M.y(:,:,z_check);
mz_true = pipeline.M.z(:,:,z_check);

mx_seed_check = Mx_seed(:,:,z_check);
my_seed_check = My_seed(:,:,z_check);
mz_seed_check = Mz_seed(:,:,z_check);

epsv = 1e-12;
rel_L2_seed = sqrt(norm(mx_seed_check(:)-mx_true(:))^2 + ...
                   norm(my_seed_check(:)-my_true(:))^2 + ...
                   norm(mz_seed_check(:)-mz_true(:))^2) / ...
              (sqrt(norm(mx_true(:))^2 + norm(my_true(:))^2 + norm(mz_true(:))^2) + epsv);

dotp = mx_true.*mx_seed_check + my_true.*my_seed_check + mz_true.*mz_seed_check;
normT = sqrt(mx_true.^2+my_true.^2+mz_true.^2) + epsv;
normS = sqrt(mx_seed_check.^2+my_seed_check.^2+mz_seed_check.^2) + epsv;
cosang = max(min(dotp./(normT.*normS), 1), -1);
ang_err_seed = acosd(cosang);

fprintf('============================================================\n');
fprintf('RAAR SEED QUALITY (z-layer %d/%d vs true field)\n', z_check, Nz_vol);
fprintf('============================================================\n');
fprintf('Relative L2 error  : %.2f %%\n', 100*rel_L2_seed);
fprintf('Mean angular error : %.2f deg\n', mean(ang_err_seed(:), 'omitnan'));
fprintf('Median ang. error  : %.2f deg\n', median(ang_err_seed(:), 'omitnan'));
fprintf('(For reference: random seed would give ~90 deg mean angle error)\n');
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 6 — PACK PIPELINE STRUCT
% ============================================================
% ============================================================

pipeline.seed.Mx          = Mx_seed;
pipeline.seed.My          = My_seed;
pipeline.seed.Mz          = Mz_seed;
pipeline.seed.method      = 'raar';
pipeline.seed.raar_err    = raar_err;
pipeline.seed.ls_residual = ls_residual;
pipeline.seed.rel_L2      = rel_L2_seed;
pipeline.seed.mean_ang_err = mean(ang_err_seed(:), 'omitnan');
pipeline.seed.a_realspace = a_realspace;   % keep for debugging

% ============================================================
% ============================================================
%   SECTION 7 — FIGURES
% ============================================================
% ============================================================

% ---- Figure 1: RAAR convergence curves ----
if show_raar_convergence
figure('Units','normalized','OuterPosition',[0.02 0.30 0.43 0.50],'Color','w');
cmap_ch = jet(N_angles);
hold on;
for r = 1:N_angles
    plot(raar_err(:,r), 'Color', [cmap_ch(r,:) 0.4], 'LineWidth', 0.8);
end
plot(mean(raar_err, 2), 'k-', 'LineWidth', 2.5);
hold off;
xlabel('Iteration', 'FontSize', 11);
ylabel('|| |F| - |A_{meas}| || / ||A_{meas}||', 'FontSize', 11);
title({'RAAR Convergence — All Channels','Black = mean  |  Colour = channel index'}, 'FontSize', 11);
grid on;
colormap(gca, jet(N_angles));
cb = colorbar; cb.Label.String = 'Channel index';
end % show_raar_convergence

% ---- Figure 2: Seed vs true field (middle z-layer) ----
if show_seed_comparison
step_q = max(1, round(Ny_vol / 20));
[Xg, Yg] = meshgrid(pipeline.M.x_vec, pipeline.M.y_vec);
Xs = Xg(1:step_q:end, 1:step_q:end);
Ys = Yg(1:step_q:end, 1:step_q:end);
Zs = zeros(size(Xs));

figure('Units','normalized','OuterPosition',[0.52 0.30 0.43 0.50],'Color','w');
subplot(1,2,1); hold on;
mxs_t = mx_true(1:step_q:end,1:step_q:end);
mys_t = my_true(1:step_q:end,1:step_q:end);
mzs_t = mz_true(1:step_q:end,1:step_q:end);
quiver3(Xs,Ys,Zs,mxs_t,mys_t,mzs_t,0.6,'b','LineWidth',1.2);
grid on; axis equal; view(35,25);
xlabel('X'); ylabel('Y'); zlabel('Mz');
title(sprintf('True Field  (z-layer %d)', z_check), 'FontSize',10);
hold off;

subplot(1,2,2); hold on;
mxs_s = mx_seed_check(1:step_q:end,1:step_q:end);
mys_s = my_seed_check(1:step_q:end,1:step_q:end);
mzs_s = mz_seed_check(1:step_q:end,1:step_q:end);
quiver3(Xs,Ys,Zs,mxs_s,mys_s,mzs_s,0.6,'r','LineWidth',1.2);
grid on; axis equal; view(35,25);
xlabel('X'); ylabel('Y'); zlabel('Mz');
title(sprintf('RAAR Seed  |  L2=%.1f%%  ang=%.1f°', ...
      100*rel_L2_seed, mean(ang_err_seed(:),'omitnan')), 'FontSize',10);
hold off;
sgtitle('RAAR Seed Quality vs True Field', 'FontSize', 12);
end % show_seed_comparison

% ---- Figure 3: Per-channel LS residual ----
if show_ls_residual
figure('Units','normalized','OuterPosition',[0.20 0.02 0.55 0.22],'Color','w');
bar(ls_residual, 'FaceColor',[0.3 0.6 0.9], 'EdgeColor','none');
hold on;
yline(mean(ls_residual), 'r--', 'LineWidth', 2);
text(N_angles*0.02, mean(ls_residual)*1.05, ...
     sprintf('mean = %.3f', mean(ls_residual)), ...
     'Color','r', 'FontSize', 9);
hold off;
xlabel('Channel index', 'FontSize', 10);
ylabel('LS residual', 'FontSize', 10);
title('Per-Channel Least Squares Residual After RAAR Seed Solve', 'FontSize', 11);
grid on;
end % show_ls_residual

% ============================================================

fprintf('--- algo_raar complete ---\n');
fprintf('pipeline.seed populated.\n');
fprintf('  Method     : RAAR + LS\n');
fprintf('  Seed size  : [%d x %d x %d]\n', Ny_vol, Nx_vol, Nz_vol);
fprintf('  L2 error   : %.2f %%\n', 100*rel_L2_seed);
fprintf('  Mean angle : %.2f deg\n', mean(ang_err_seed(:),'omitnan'));
fprintf('Pass pipeline to module6.m (RESIRE-V).\n\n');


% ============================================================
% ============================================================
%   ALGORITHM SLOT — DIRECT LS INVERSION
%   Q-space linear inversion from b_minus measurements
% ============================================================
%
% PURPOSE:
%   Bypasses phase retrieval entirely. For each spatial frequency
%   (qx, qy) on the volume grid, uses the 32 b_minus measurements
%   across all angles to solve directly for the depth-integrated
%   magnetization via the pseudoinverse of the sensing matrix.
%   IFFTs the result back to real space and normalizes to unit spins.
%
% WHY NOT RAAR:
%   b_minus = 2*xmcd*Re(conj(A_charge)*A_mag) is already a real-valued
%   linear measurement of FFT[p·M]. There is no missing phase — the
%   cross-term with A_charge collapses the complex phase into a real
%   number. RAAR solves the wrong problem here.
%
% FORWARD MODEL (per angle r, per q-pixel):
%   b_minus(q,r) = 2*xmcd * sum_z[ att(z) * (p_r · FFT[M(:,:,z)](q))
%                                          * cos(q_z(r)*z) ]
%
%   Approximation used here: depth-average the attenuation and phase,
%   then solve the 2D problem: b_minus(q,r) ≈ c_r * p_r · FM(q)
%   where FM(q) = FFT[M_eff(x,y)] is an effective 2D magnetization.
%   This is exact when Nz=1 and a good approximation when the film
%   is thin relative to the absorption length.
%
% APPROACH:
%   1. For each volume q-pixel, interpolate b_minus from detector grid
%   2. Build [N_angles x 1] measurement vector at that q
%   3. Solve: FM(q) = Pinv * b_meas(q)   [3 x 1]
%   4. IFFT FM back to real space -> Mx, My, Mz
%   5. Normalize to unit spins
%   6. Replicate across z-layers (depth resolution requires CNN/RESIRE)
%
% INPUTS:
%   pipeline.meas.b_minus       — [Npy x Npx x N_angles]
%   pipeline.geom.angle_list    — q-maps per angle
%   pipeline.sensing.Pinv       — pseudoinverse [3 x N_angles]
%   pipeline.sensing.Pmat       — sensing matrix [N_angles x 3]
%   pipeline.M.x/y/z            — true field (for seed quality check)
%   pipeline.geom.xmcd_asymmetry
%   pipeline.geom.z_scale_nm
%   pipeline.geom.absorption_length_nm
%
% OUTPUTS:
%   pipeline.seed.Mx/My/Mz     — [Ny x Nx x Nz] initial estimate
%   pipeline.seed.method       — 'ls_direct'
%
% ============================================================

if ~exist('pipeline','var') || ~isfield(pipeline,'sensing')
    error('Modules 1, 2, and 3 must be run first.');
end

fprintf('\n============================================================\n');
fprintf('ALGORITHM SLOT — DIRECT LS INVERSION\n');
fprintf('============================================================\n\n');

% ============================================================
%   RETRIEVE INPUTS
% ============================================================

b_minus_all = pipeline.meas.b_minus;       % [Npy x Npx x N_angles]
N_angles    = pipeline.meas.N_angles;
angle_list  = pipeline.geom.angle_list;
Pinv        = pipeline.sensing.Pinv;        % [3 x N_angles]
Pmat        = pipeline.sensing.Pmat;        % [N_angles x 3]
xmcd        = pipeline.geom.xmcd_asymmetry;
labs        = pipeline.geom.absorption_length_nm;
z_scale     = pipeline.geom.z_scale_nm;
z_vec_nm    = pipeline.M.z_vec * z_scale;

Ny_vol = size(pipeline.M.x, 1);
Nx_vol = size(pipeline.M.x, 2);
Nz_vol = size(pipeline.M.x, 3);

scale_nm  = pipeline.M.sample_scale_nm;
x_vec     = pipeline.M.x_vec;
y_vec     = pipeline.M.y_vec;
dx_vol_nm = (x_vec(end)-x_vec(1)) / max(Nx_vol-1,1) * scale_nm;
dy_vol_nm = (y_vec(end)-y_vec(1)) / max(Ny_vol-1,1) * scale_nm;
qx_axis   = 2*pi*(-Nx_vol/2:Nx_vol/2-1) / (Nx_vol*dx_vol_nm);  % rad/nm
qy_axis   = 2*pi*(-Ny_vol/2:Ny_vol/2-1) / (Ny_vol*dy_vol_nm);  % rad/nm

fprintf('Volume grid      : [%d x %d x %d]\n', Ny_vol, Nx_vol, Nz_vol);
fprintf('Channels         : %d\n', N_angles);
fprintf('Depth layers     : z = %s nm\n', num2str(z_vec_nm, '%.1f  '));
fprintf('\n');

% ============================================================
%   COMPUTE DEPTH-ATTENUATION WEIGHTS PER LAYER
%   b_minus(q,r) = 2*xmcd * sum_z[ w(z) * cos(q_z_center(r)*z)
%                                        * p_r · FM_z(q) ]
%   We fold the depth weighting into an effective scale per channel
%   and solve for a weighted average magnetization FM_eff(q).
% ============================================================

% Effective depth weight per channel: sum_z att(z) * mean_cos(q_z, z)
% Use mean q_z across detector pixels for each angle (scalar per angle)
eff_weight = zeros(N_angles, 1);
for r = 1:N_angles
    q_z_map  = angle_list(r).q_z_map;
    valid_r  = angle_list(r).valid;
    qz_mean  = mean(q_z_map(valid_r));
    w = 0;
    for kz = 1:Nz_vol
        z_k = z_vec_nm(kz);
        att = exp(-2*z_k / max(labs,1e-12));
        w   = w + att * cos(qz_mean * z_k);
    end
    eff_weight(r) = 2 * xmcd * w;
end

% Scale Pinv rows by 1/eff_weight so the LS solve accounts for depth
% Avoid divide-by-zero
eff_weight(abs(eff_weight) < 1e-12) = 1e-12;
Pinv_scaled = Pinv ./ eff_weight(:).';   % [3 x N_angles]

fprintf('Effective depth weights (per channel):\n');
fprintf('  min=%.4f  max=%.4f  mean=%.4f\n\n', ...
        min(eff_weight), max(eff_weight), mean(eff_weight));

% ============================================================
%   INTERPOLATE b_minus ONTO VOLUME Q-GRID PER CHANNEL
%   For each angle, resample the 512x512 detector b_minus
%   onto the 140x140 volume q-grid using the physical q-maps.
% ============================================================

fprintf('Interpolating b_minus onto volume q-grid...\n');
b_vol = zeros(Ny_vol, Nx_vol, N_angles);   % b_minus on volume q-grid

for r = 1:N_angles
    q_x_map = angle_list(r).q_x_map;   % [Npy x Npx]
    q_y_map = angle_list(r).q_y_map;
    valid_r = angle_list(r).valid;

    b_det = b_minus_all(:,:,r);         % [Npy x Npx]
    b_det(~valid_r) = 0;

    % Interpolate: for each volume q-pixel (qx_axis, qy_axis),
    % find nearest detector pixel and pull its b_minus value.
    % Use interp2 with the detector q-maps as the source grid.
    % Since detector q-maps are not on a regular grid, use griddata.
    qx_det = q_x_map(valid_r);
    qy_det = q_y_map(valid_r);
    b_det_v = b_det(valid_r);

    [QX_vol, QY_vol] = meshgrid(qx_axis, qy_axis);

    % griddata scattered interpolation from detector q-points to volume grid
    b_interp = griddata(qx_det, qy_det, double(b_det_v), ...
                        QX_vol, QY_vol, 'linear');
    b_interp(isnan(b_interp)) = 0;   % extrapolated points set to zero

    b_vol(:,:,r) = b_interp;
end
fprintf('Interpolation complete.\n\n');

% ============================================================
%   DIRECT LS SOLVE IN Q-SPACE
%   At each q-pixel: FM_eff(q) = Pinv_scaled * b_vec(q)
%   where b_vec(q) = [b_vol(q,1); ...; b_vol(q,N_angles)]
% ============================================================

fprintf('Solving LS system at each q-pixel...\n');

% Reshape b_vol to [N_angles x Ny_vol*Nx_vol]
b_mat = reshape(b_vol, Ny_vol*Nx_vol, N_angles).';   % [N_angles x Npix]

% LS solve: FM_eff = [3 x Npix]
FM_eff = Pinv_scaled * b_mat;   % [3 x Npix]

FMx = reshape(FM_eff(1,:), Ny_vol, Nx_vol);
FMy = reshape(FM_eff(2,:), Ny_vol, Nx_vol);
FMz = reshape(FM_eff(3,:), Ny_vol, Nx_vol);

fprintf('LS solve complete.\n\n');

% ============================================================
%   IFFT BACK TO REAL SPACE
% ============================================================

fprintf('IFFTing to real space...\n');

Mx_2d = real(ifft2(ifftshift(FMx)));
My_2d = real(ifft2(ifftshift(FMy)));
Mz_2d = real(ifft2(ifftshift(FMz)));

% ============================================================
%   NORMALIZE TO UNIT SPINS
% ============================================================

mag = sqrt(Mx_2d.^2 + My_2d.^2 + Mz_2d.^2);
mag(mag < 1e-12) = 1;
Mx_2d = Mx_2d ./ mag;
My_2d = My_2d ./ mag;
Mz_2d = Mz_2d ./ mag;

% Replicate across z-layers
% (depth resolution is handled by RESIRE-V, not this seed)
Mx_seed = repmat(Mx_2d, [1, 1, Nz_vol]);
My_seed = repmat(My_2d, [1, 1, Nz_vol]);
Mz_seed = repmat(Mz_2d, [1, 1, Nz_vol]);

% ============================================================
%   SEED QUALITY CHECK VS TRUE FIELD
% ============================================================

z_check = round(Nz_vol/2);
mx_true = pipeline.M.x(:,:,z_check);
my_true = pipeline.M.y(:,:,z_check);
mz_true = pipeline.M.z(:,:,z_check);

mx_s = Mx_seed(:,:,z_check);
my_s = My_seed(:,:,z_check);
mz_s = Mz_seed(:,:,z_check);

epsv = 1e-12;
rel_L2 = sqrt(norm(mx_s(:)-mx_true(:))^2 + ...
              norm(my_s(:)-my_true(:))^2 + ...
              norm(mz_s(:)-mz_true(:))^2) / ...
         (sqrt(norm(mx_true(:))^2+norm(my_true(:))^2+norm(mz_true(:))^2)+epsv);

dotp   = mx_true.*mx_s + my_true.*my_s + mz_true.*mz_s;
normT  = sqrt(mx_true.^2+my_true.^2+mz_true.^2)+epsv;
normS  = sqrt(mx_s.^2+my_s.^2+mz_s.^2)+epsv;
cosang = max(min(dotp./(normT.*normS),1),-1);
ang_err = acosd(cosang);

fprintf('============================================================\n');
fprintf('LS SEED QUALITY (z-layer %d/%d vs true field)\n', z_check, Nz_vol);
fprintf('============================================================\n');
fprintf('Relative L2 error  : %.2f %%\n', 100*rel_L2);
fprintf('Mean angular error : %.2f deg\n', mean(ang_err(:),'omitnan'));
fprintf('Median ang. error  : %.2f deg\n', median(ang_err(:),'omitnan'));
fprintf('(Random baseline   : ~90 deg)\n');
fprintf('============================================================\n\n');

% ============================================================
%   PACK PIPELINE STRUCT
% ============================================================

pipeline.seed.Mx           = Mx_seed;
pipeline.seed.My           = My_seed;
pipeline.seed.Mz           = Mz_seed;
pipeline.seed.method       = 'ls_direct';
pipeline.seed.rel_L2       = rel_L2;
pipeline.seed.mean_ang_err = mean(ang_err(:),'omitnan');
pipeline.seed.raar_err     = [];
pipeline.seed.ls_residual  = [];

fprintf('--- module4_ls complete ---\n');
fprintf('Seed: direct LS inversion  [%d x %d x %d]\n', Ny_vol, Nx_vol, Nz_vol);
fprintf('Pass pipeline to module6.m (RESIRE-V).\n\n');