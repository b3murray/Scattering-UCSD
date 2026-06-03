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
% ============================================================
 
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
 
eff_weight(abs(eff_weight) < 1e-12) = 1e-12;
Pinv_scaled = Pinv ./ eff_weight(:).';   % [3 x N_angles]
 
fprintf('Effective depth weights (per channel):\n');
fprintf('  min=%.4f  max=%.4f  mean=%.4f\n\n', ...
        min(eff_weight), max(eff_weight), mean(eff_weight));
 
% ============================================================
%   INTERPOLATE b_minus ONTO VOLUME Q-GRID PER CHANNEL
%
%   SPEED OPTIMIZATION: interpolation weights are precomputed
%   on the first call and cached in pipeline.sensing.interp_idx.
%   Subsequent calls (e.g. in generate_training_data.m loop)
%   skip the expensive griddata and use fast array indexing.
%   First call: ~15s. All subsequent calls: <1s.
% ============================================================
 
% Interpolate b_minus from detector grid onto volume q-grid using interp2.
% The volume q-axes are regular grids — interp2 handles this in one vectorized
% call per angle, no index map needed, no toolbox required.
fprintf('Interpolating b_minus onto volume q-grid...\n');

[QX_vol, QY_vol] = meshgrid(qx_axis, qy_axis);
Npix = Ny_vol * Nx_vol;

b_vol = zeros(Npix, N_angles);
for r = 1:N_angles
    % Physical detector q-vectors for this angle
    q_x_map = angle_list(r).q_x_map;   % [Npy_det x Npx_det]
    q_y_map = angle_list(r).q_y_map;
    valid_r = angle_list(r).valid;

    % Build regular axes for the detector q-grid (rows=qy, cols=qx)
    % Use the valid pixels only to get the source grid extent
    qx_det = q_x_map(1, :);   % one row — qx varies along columns
    qy_det = q_y_map(:, 1);   % one col — qy varies along rows

    b_det = b_minus_all(:,:,r);
    b_det(~valid_r) = 0;

    % interp2: query the detector b_minus at each volume q-pixel
    % QX_vol, QY_vol are [Ny_vol x Nx_vol] — same shape as output needed
    b_interp = interp2(qx_det, qy_det, b_det, QX_vol, QY_vol, 'linear', 0);

    b_vol(:, r) = b_interp(:);
end

fprintf('Interpolation complete.\n\n');
 
% ============================================================
%   DIRECT LS SOLVE IN Q-SPACE
%   At each q-pixel: FM_eff(q) = Pinv_scaled * b_vec(q)
%   where b_vec(q) = [b_vol(q,1); ...; b_vol(q,N_angles)]
% ============================================================
 
fprintf('Solving LS system at each q-pixel...\n');
 
% Reshape b_vol to [N_angles x Ny_vol*Nx_vol]
b_mat = b_vol.';   % [N_angles x Npix]
 
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