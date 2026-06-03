% ============================================================
% ============================================================
%   MODULE 2 — FORWARD SCATTERING MODEL
%   Reflection/Diffraction Geometry | XMCD | Ewald Sphere
% ============================================================
%
% PURPOSE:
%   Takes the true 3D magnetization volume from Module 1 and
%   simulates what a real detector at ALS COSMIC 7.0.1.1 would
%   measure in reflection/diffraction geometry with left and
%   right circularly polarized soft X-rays.
%
% PHYSICS:
%   For each (theta_inc, phi) angle pair:
%     1. Compute incident wavevector k_i
%     2. Map every detector pixel to its q-vector via Ewald sphere
%     3. For each depth z, compute phase = q_z * z and absorption
%     4. Sum over all depths to get complex scattered amplitude A
%     5. Apply XMCD: I+/I- from left/right circular polarization
%     6. Store b+ (structural) and b- (magnetic) signals
%
%   The depth-dependent phase is the key physical ingredient that
%   encodes depth information into the diffraction pattern.
%   Different (theta,phi) angles sample different Ewald sphere
%   slices, allowing depth reconstruction from the full dataset.
%
% GEOMETRY:
%   Reflection/diffraction — detector is NOT behind the sample.
%   Incident beam hits sample surface at grazing/moderate angle.
%   Scattered beam collected at 2theta from incident direction.
%   This is the geometry used at ALS COSMIC for magnetic films.
%
% INPUTS (from Module 1 via pipeline struct):
%   pipeline.M.x, .y, .z       — Mx, My, Mz [Ny x Nx x Nz]
%   pipeline.M.x_vec            — x coordinate vector
%   pipeline.M.y_vec            — y coordinate vector
%   pipeline.M.z_vec            — z coordinate vector (depth, nm)
%   pipeline.M.params           — domain parameters struct
%
% OUTPUTS (added to pipeline struct):
%   pipeline.meas.I_plus        — I+ patterns [Npx x Npy x Nangles]
%   pipeline.meas.I_minus       — I- patterns [Npx x Npy x Nangles]
%   pipeline.meas.b_plus        — structural signal
%   pipeline.meas.b_minus       — magnetic signal (what we invert)
%   pipeline.meas.Amp_plus      — sqrt(b+) amplitudes
%   pipeline.meas.Amp_minus     — signed sqrt(|b-|) amplitudes
%   pipeline.geom.q_x/y/z       — q-vectors per pixel per angle
%   pipeline.geom.two_theta     — 2theta map on detector
%   pipeline.geom.angles        — all (theta,phi) pairs used
%   pipeline.geom.k_i           — incident wavevectors
%   pipeline.ewald.qx/qy/qz    — all measured q-points (3D coverage)
%
% FIGURES:
%   Figure 1 — 3D Ewald sphere with measured q-points
%   Figure 2 — 2theta and q_z maps on detector
%   Figure 3 — Example I+, I-, b+, b- at one angle
%   Figure 4 — 3D reciprocal space coverage / missing wedge
%
% ============================================================

% ============================================================
%   CHECK MODULE 1 HAS BEEN RUN
% ============================================================
if ~exist('pipeline','var') || ~isfield(pipeline,'M')
    error(['Module 1 must be run first.\n' ...
           'Run module1.m and ensure it populates pipeline.M']);
end

fprintf('\n============================================================\n');
fprintf('MODULE 2 — FORWARD SCATTERING MODEL\n');
fprintf('Geometry: Reflection/Diffraction | XMCD | Ewald Sphere\n');
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 1 — BEAMLINE AND DETECTOR PARAMETERS
% ============================================================
% ============================================================
%
% Default values based on ALS COSMIC 7.0.1.1 published specs.
% All values are USER CONTROLLED — override as needed.
% See beamline reference footer for sources.

% ----- Photon energy -----
% Set to resonant absorption edge of your magnetic element:
%   Fe L3-edge : ~707 eV   (most common for iron-based magnets)
%   Co L3-edge : ~779 eV
%   Ni L3-edge : ~853 eV
%   Mn L3-edge : ~641 eV
photon_energy_eV = 707;         % eV — Fe L3-edge default

% ----- Detector geometry -----
% ALS COSMIC uses the VeryFastCCD detector (48 um pixels)
% These are published specs — verify against your logbook
detector_Npx     = 512;         % pixels along x (physical detector)
detector_Npy     = 512;         % pixels along y (physical detector)
pixel_size_um    = 48;          % um — VeryFastCCD pixel size
det_distance_mm  = 150;         % mm — sample to detector distance
beam_center_x    = 256;         % detector pixel — beam center x
beam_center_y    = 256;         % detector pixel — beam center y

% ----- Computation grid -----
% The forward model FFT is computed on this grid.
% MUST match the volume grid (Nx, Ny from module1) so that
% module6 RESIRE-V gradient is consistent with these measurements.
% Set to 0 to auto-match module1 volume size (recommended).
comp_Npx = 0;   % 0 = auto-match module1 Nx
comp_Npy = 0;   % 0 = auto-match module1 Ny

% ----- Angular scan — THETA (sample tilt) -----
% theta is the incident angle measured from the sample surface normal
% At ALS COSMIC this is the sample tilt stage
% Range 15-60 deg is typical for soft X-ray reflection
theta_min_deg    = 15;          % deg — minimum tilt
theta_max_deg    = 60;          % deg — maximum tilt
N_theta          = 8;           % number of theta steps to simulate
                                % (more = better depth resolution, slower)

% ----- Angular scan — PHI (azimuthal rotation) -----
% phi is the in-plane sample rotation
% At ALS COSMIC this is done manually between runs
% List every phi angle you will collect data at:
phi_angles_deg   = [0, 45, 90, 135];   % deg — manual phi positions

% ----- Absorption -----
% Soft X-ray absorption length in the magnetic layer (material dependent)
% At Fe L-edge in iron: ~20-50 nm
% At Co L-edge in cobalt: ~15-40 nm
% This controls how deep the beam penetrates — critical for depth reconstruction
absorption_length_nm = 30;      % nm — adjust for your material

% ----- Noise model -----
% Set to false when testing with truefield seed (module4_truefield)
% so the true field gives residual=0 at iteration 1.
% Set to true for realistic simulation once RESIRE-V is verified.
add_poisson_noise = false;      % true = realistic photon shot noise
photon_flux      = 1e8;         % photons/pixel — controls SNR

% ----- Polarization factor -----
% XMCD asymmetry factor — scales the magnetic contribution to scattering
% At resonance this can be large (0.1-0.5)
% Off resonance it is small (~0.01)
% Set to 1.0 for ideal case, reduce for realistic contrast
xmcd_asymmetry   = 0.3;        % dimensionless [0..1]

% ---- Figure toggles — set false to skip ----
if ~exist('show_ewald_sphere',    'var'), show_ewald_sphere    = true; end
if ~exist('show_detector_maps',   'var'), show_detector_maps   = true; end
if ~exist('show_example_patterns','var'), show_example_patterns = true; end
if ~exist('show_qspace_coverage', 'var'), show_qspace_coverage = true; end

% ============================================================
% ============================================================
%   SECTION 2 — DERIVED PHYSICAL QUANTITIES
% ============================================================
% ============================================================

% Convert photon energy to wavelength
hc_eV_nm         = 1239.84;                    % eV*nm
lambda_nm         = hc_eV_nm / photon_energy_eV; % nm
lambda_m          = lambda_nm * 1e-9;            % m
k_mag             = 2*pi / lambda_nm;            % rad/nm — wavevector magnitude

% Detector pixel size in nm (for q calculation)
pixel_size_nm     = pixel_size_um * 1e3;         % convert um to nm
det_distance_nm   = det_distance_mm * 1e6;       % convert mm to nm

% Build theta scan array
theta_scan_deg    = linspace(theta_min_deg, theta_max_deg, N_theta);
theta_scan_rad    = deg2rad(theta_scan_deg);

% All (theta, phi) combinations
phi_scan_rad      = deg2rad(phi_angles_deg);
N_phi             = numel(phi_angles_deg);
N_angles          = N_theta * N_phi;

% Print geometry summary
fprintf('--- Beamline Parameters ---\n');
fprintf('Photon energy       : %.1f eV\n', photon_energy_eV);
fprintf('Wavelength          : %.4f nm\n', lambda_nm);
fprintf('Wavevector |k|      : %.4f rad/nm\n', k_mag);
fprintf('\n--- Detector Parameters ---\n');
fprintf('Detector size       : %d x %d pixels\n', detector_Npx, detector_Npy);
fprintf('Pixel size          : %.1f um\n', pixel_size_um);
fprintf('Sample-detector dist: %.1f mm\n', det_distance_mm);
fprintf('Beam center         : (%d, %d) pixels\n', beam_center_x, beam_center_y);
fprintf('\n--- Angular Scan ---\n');
fprintf('Theta range         : %.1f to %.1f deg  (%d steps)\n', ...
        theta_min_deg, theta_max_deg, N_theta);
fprintf('Phi positions       : '); fprintf('%.1f  ', phi_angles_deg); fprintf('deg\n');
fprintf('Total angle pairs   : %d (theta) x %d (phi) = %d\n', ...
        N_theta, N_phi, N_angles);
fprintf('\n--- Material Parameters ---\n');
fprintf('Absorption length   : %.1f nm\n', absorption_length_nm);
fprintf('XMCD asymmetry      : %.2f\n', xmcd_asymmetry);
fprintf('Poisson noise       : %s\n', mat2str(add_poisson_noise));
fprintf('\n');

% ============================================================
% ============================================================
%   SECTION 3 — DETECTOR PIXEL TO Q-SPACE MAPPING
% ============================================================
%
% For each detector pixel (u,v), compute the scattered wavevector
% k_f and the scattering vector q = k_f - k_i.
%
% This is done once per (theta,phi) since k_i changes with angle.
% The 2theta angle and q_z value for each pixel are stored.
%
% Convention:
%   z-axis: sample surface normal (depth direction)
%   x-axis: along beam projection on surface
%   y-axis: transverse to beam in surface plane
% ============================================================

% Resolve computation grid size — use REAL detector size
% The q-vectors correspond to physical detector pixels
% Module6 will interpolate the volume FFT at these q-vectors
Ny_vol_m2 = size(pipeline.M.x, 1);
Nx_vol_m2 = size(pipeline.M.x, 2);
comp_Npx = detector_Npx;
comp_Npy = detector_Npy;

fprintf('Computation grid    : %d x %d pixels (real detector size)\n', comp_Npx, comp_Npy);
fprintf('Volume grid         : %d x %d pixels (module1 resolution)\n\n', ...
        Nx_vol_m2, Ny_vol_m2);

% Build detector pixel coordinate arrays (physical detector geometry)
beam_center_cx = beam_center_x;
beam_center_cy = beam_center_y;

[det_u, det_v] = meshgrid( ...
    ((1:comp_Npx) - beam_center_cx) * pixel_size_nm, ...
    ((1:comp_Npy) - beam_center_cy) * pixel_size_nm);

% ============================================================
% ============================================================
%   SECTION 4 — RETRIEVE MODULE 1 OUTPUTS
% ============================================================
% ============================================================

Mx      = pipeline.M.x;
My      = pipeline.M.y;
Mz      = pipeline.M.z;
x_vec   = pipeline.M.x_vec;
y_vec   = pipeline.M.y_vec;
z_vec   = pipeline.M.z_vec;   % depth vector in physical units
params  = pipeline.M.params;

Ny_vol  = size(Mx, 1);
Nx_vol  = size(Mx, 2);
Nz_vol  = size(Mx, 3);

% z_vec must be in nm for the phase calculation to be physically correct.
% z_range uses the same unit system as x_range/y_range in module1,
% so the same sample_scale_nm conversion applies.
z_scale_nm = pipeline.M.sample_scale_nm;   % nm per arbitrary unit (same as x/y)
z_vec_nm   = z_vec * z_scale_nm;

fprintf('Magnetization volume: [%d x %d x %d]\n', Ny_vol, Nx_vol, Nz_vol);
fprintf('Depth range         : %.2f to %.2f nm\n', min(z_vec_nm), max(z_vec_nm));
fprintf('\nComputing forward scattering for %d angle pairs...\n\n', N_angles);

% ============================================================
% ============================================================
%   SECTION 5 — ALLOCATE OUTPUT STORAGE
% ============================================================
% ============================================================

I_plus_all   = zeros(comp_Npy, comp_Npx, N_angles);
I_minus_all  = zeros(comp_Npy, comp_Npx, N_angles);
b_plus_all   = zeros(comp_Npy, comp_Npx, N_angles);
b_minus_all  = zeros(comp_Npy, comp_Npx, N_angles);

% Store all q-vectors for Ewald sphere plot
all_qx = [];  all_qy = [];  all_qz = [];

% angle_list built dynamically in the loop — no pre-allocation needed

% ============================================================
% ============================================================
%   SECTION 6 — MAIN LOOP: FORWARD SCATTERING PER ANGLE
% ============================================================
% ============================================================

ang_idx = 0;

for ip = 1:N_phi
    phi_rad   = phi_scan_rad(ip);
    phi_deg_k = phi_angles_deg(ip);

    for it = 1:N_theta
        theta_rad = theta_scan_rad(it);
        theta_deg_k = theta_scan_deg(it);
        ang_idx = ang_idx + 1;

        % --------------------------------------------------------
        % STEP 1 — Incident wavevector k_i (reflection geometry)
        % In reflection geometry the beam comes in at angle theta_i
        % from the surface normal.
        % k_i points INTO the sample:
        %   k_ix =  k * sin(theta) * cos(phi)   (along surface, beam direction)
        %   k_iy =  k * sin(theta) * sin(phi)   (transverse)
        %   k_iz = -k * cos(theta)               (into surface, negative z)
        % --------------------------------------------------------
        k_ix = k_mag * sin(theta_rad) * cos(phi_rad);
        k_iy = k_mag * sin(theta_rad) * sin(phi_rad);
        k_iz = -k_mag * cos(theta_rad);
        k_i     = [k_ix, k_iy, k_iz];
        k_i_hat = k_i / norm(k_i);   % unit vector — XMCD sensitivity direction

        % --------------------------------------------------------
        % STEP 2 — Map detector pixels to scattered wavevectors k_f
        % For a flat detector perpendicular to the nominal exit beam:
        % The exit beam direction at specular reflection is:
        %   k_f_spec = [k_ix, k_iy, +k * cos(theta)]
        % Each pixel offset from center adds a small angular deviation.
        % --------------------------------------------------------
        % Nominal exit direction (specular)
        k_fx_nom = k_ix;
        k_fy_nom = k_iy;
        k_fz_nom = k_mag * cos(theta_rad);   % positive z = out of surface

        % For each pixel, the scattered direction unit vector:
        % Pixel offsets in nm from beam center
        kf_x = k_fx_nom + det_u / det_distance_nm * k_mag;
        kf_y = k_fy_nom + det_v / det_distance_nm * k_mag;
        kf_z_sq = k_mag^2 - kf_x.^2 - kf_y.^2;

        % Physical constraint: kf_z must be real (evanescent pixels set to 0)
        valid = kf_z_sq > 0;
        kf_z = zeros(size(kf_z_sq));
        kf_z(valid) = sqrt(kf_z_sq(valid));

        % Renormalize to |k_f| = k_mag (elastic scattering)
        kf_norm = sqrt(kf_x.^2 + kf_y.^2 + kf_z.^2);
        kf_norm(kf_norm < 1e-12) = 1;
        kf_x = (kf_x ./ kf_norm) .* k_mag;
        kf_y = (kf_y ./ kf_norm) .* k_mag;
        kf_z = (kf_z ./ kf_norm) .* k_mag;

        % --------------------------------------------------------
        % STEP 3 — Scattering vector q = k_f - k_i (per pixel)
        % q encodes which spatial frequency in the sample is probed
        % q_z encodes depth sensitivity: larger q_z -> finer depth res
        % --------------------------------------------------------
        q_x = kf_x - k_ix;
        q_y = kf_y - k_iy;
        q_z = kf_z - k_iz;   % k_iz is negative so q_z = kf_z + |k_iz|

        % 2theta for each pixel (angle between k_i and k_f)
        cos_two_theta = (kf_x.*k_ix + kf_y.*k_iy + kf_z.*k_iz) ./ k_mag^2;
        cos_two_theta = max(min(cos_two_theta, 1), -1);
        two_theta_map = acosd(cos_two_theta);

        % Store q-points for Ewald sphere plot (subsample for speed)
        sub = 16;
        qx_sub = q_x(1:sub:end, 1:sub:end);
        qy_sub = q_y(1:sub:end, 1:sub:end);
        qz_sub = q_z(1:sub:end, 1:sub:end);
        all_qx = [all_qx; qx_sub(:)];
        all_qy = [all_qy; qy_sub(:)];
        all_qz = [all_qz; qz_sub(:)];

        % Store angle info — full pixel-wise q maps for ALL three components
        % These are used by module6 to interpolate FFT at correct q-vectors
        angle_list(ang_idx).theta_deg        = theta_deg_k;
        angle_list(ang_idx).phi_deg          = phi_deg_k;
        angle_list(ang_idx).k_i              = k_i;
        angle_list(ang_idx).k_i_hat          = k_i_hat;
        angle_list(ang_idx).q_center         = [mean(q_x(:)) mean(q_y(:)) mean(q_z(:))];
        angle_list(ang_idx).two_theta_center = two_theta_map(beam_center_y, beam_center_x);
        angle_list(ang_idx).q_x_map          = q_x;   % [Npy x Npx] rad/nm
        angle_list(ang_idx).q_y_map          = q_y;   % [Npy x Npx] rad/nm
        angle_list(ang_idx).q_z_map          = q_z;   % [Npy x Npx] rad/nm
        angle_list(ang_idx).valid            = valid;  % non-evanescent pixels

        % --------------------------------------------------------
        % STEP 4 — Depth-dependent scattering amplitude
        %
        % For each depth layer z_k:
        %   dA(q,z) = F_mag(q,z) * exp(i*q_z(q)*z) * exp(-z/labs)
        %
        % F_mag(q,z) = FFT2[ p · M(x,y,z) ] evaluated at each q-pixel
        % This is the FULL 2D FFT forward model — spatially resolved.
        % Each detector pixel (u,v) gets its own q-vector and phase.
        %
        % Note: q_z varies per pixel (stored in q_z map above).
        % The depth phase exp(i*q_z(u,v)*z) is therefore pixel-wise,
        % which is what allows different depths to be separated.
        % --------------------------------------------------------

        A_charge = zeros(comp_Npy, comp_Npx);
        A_mag    = zeros(comp_Npy, comp_Npx);

        % Volume q-grid axes in rad/nm
        % x_vec is in arbitrary units — convert to nm using sample_scale_nm
        scale_nm = pipeline.M.sample_scale_nm;
        dx_vol_nm = (x_vec(end)-x_vec(1)) / max(Nx_vol-1, 1) * scale_nm;  % nm per pixel
        dy_vol_nm = (y_vec(end)-y_vec(1)) / max(Ny_vol-1, 1) * scale_nm;
        qx_axis_vol = 2*pi * (-Nx_vol/2 : Nx_vol/2-1) / (Nx_vol * dx_vol_nm);  % rad/nm
        qy_axis_vol = 2*pi * (-Ny_vol/2 : Ny_vol/2-1) / (Ny_vol * dy_vol_nm);  % rad/nm

        for kz = 1:Nz_vol
            z_k = z_vec_nm(kz);
            att = exp(-2 * z_k / absorption_length_nm);

            mx_k = Mx(:,:,kz);
            my_k = My(:,:,kz);
            mz_k = Mz(:,:,kz);

            % Magnetic projection along beam: k_i_hat · M
            proj_mag_k = k_i_hat(1)*mx_k + k_i_hat(2)*my_k + k_i_hat(3)*mz_k;

            % Full 2D FFT on volume grid [Ny_vol x Nx_vol]
            F_mag_layer = fftshift(fft2(proj_mag_k));

            % Interpolate F_mag at physical detector q-vectors
            % q_x and q_y are [comp_Npy x comp_Npx] = [512 x 512]
            % F_mag_layer is [140 x 140] on volume q-grid
            F_det = interp2(qx_axis_vol, qy_axis_vol.', F_mag_layer, ...
                            q_x, q_y, 'linear', 0);

            % Pixel-wise depth phase on detector grid [512 x 512]
            phase_depth = exp(1i * q_z * z_k);

            % Now all arrays are [512 x 512] — safe to multiply
            A_charge = A_charge + att * phase_depth;
            A_mag    = A_mag    + att * xmcd_asymmetry * F_det .* phase_depth;
        end

        % --------------------------------------------------------
        % STEP 5 — Left and right circular polarization intensities
        %
        % I+ = |A_charge + A_mag|^2   (left circular)
        % I- = |A_charge - A_mag|^2   (right circular)
        % --------------------------------------------------------
        A_plus  = A_charge + A_mag;
        A_minus = A_charge - A_mag;

        I_plus  = abs(A_plus).^2;
        I_minus = abs(A_minus).^2;

        % Zero out evanescent pixels
        I_plus(~valid)  = 0;
        I_minus(~valid) = 0;

        % --------------------------------------------------------
        % STEP 6 — Add Poisson photon noise
        % --------------------------------------------------------
        if add_poisson_noise
            scale   = photon_flux / max(max(I_plus(:)), max(I_minus(:)));
            if scale > 0 && isfinite(scale)
                I_plus  = double(poissrnd(I_plus  * scale)) / scale;
                I_minus = double(poissrnd(I_minus * scale)) / scale;
            end
        end

        % --------------------------------------------------------
        % STEP 7 — XMCD sum and difference signals
        % --------------------------------------------------------
        b_plus  = 0.5 * (I_plus + I_minus);   % structural (charge)
        b_minus = 0.5 * (I_plus - I_minus);   % magnetic (XMCD)

        % Store
        I_plus_all(:,:,ang_idx)  = I_plus;
        I_minus_all(:,:,ang_idx) = I_minus;
        b_plus_all(:,:,ang_idx)  = b_plus;
        b_minus_all(:,:,ang_idx) = b_minus;

        fprintf('  Angle %3d/%d  |  theta=%5.1f deg  phi=%6.1f deg  |  2theta_center=%.2f deg\n', ...
                ang_idx, N_angles, theta_deg_k, phi_deg_k, ...
                angle_list(ang_idx).two_theta_center);
    end
end

fprintf('\nForward model complete. %d diffraction patterns computed.\n\n', N_angles);

% Amplitude arrays for RAAR / RESIRE input
Amp_plus_all  = sqrt(abs(b_plus_all)  + 1e-12);
Amp_minus_all = sign(b_minus_all) .* sqrt(abs(b_minus_all) + 1e-12);

% ============================================================
% ============================================================
%   SECTION 7 — PACK PIPELINE STRUCT
% ============================================================
% ============================================================

pipeline.meas.I_plus       = I_plus_all;
pipeline.meas.I_minus      = I_minus_all;
pipeline.meas.b_plus       = b_plus_all;
pipeline.meas.b_minus      = b_minus_all;
pipeline.meas.Amp_plus     = Amp_plus_all;
pipeline.meas.Amp_minus    = Amp_minus_all;
pipeline.meas.N_angles     = N_angles;

pipeline.geom.theta_deg    = theta_scan_deg;
pipeline.geom.phi_deg      = phi_angles_deg;
pipeline.geom.angle_list   = angle_list;
pipeline.geom.lambda_nm    = lambda_nm;
pipeline.geom.k_mag        = k_mag;
pipeline.geom.pixel_size_nm = pixel_size_nm;
pipeline.geom.det_distance_nm = det_distance_nm;
pipeline.geom.detector_Npx = detector_Npx;
pipeline.geom.detector_Npy = detector_Npy;
pipeline.geom.comp_Npx     = comp_Npx;
pipeline.geom.comp_Npy     = comp_Npy;
pipeline.geom.beam_center  = [beam_center_x beam_center_y];
pipeline.geom.photon_energy_eV = photon_energy_eV;
pipeline.geom.absorption_length_nm = absorption_length_nm;
pipeline.geom.xmcd_asymmetry      = xmcd_asymmetry;
pipeline.geom.z_scale_nm          = z_scale_nm;   % nm per z_vec unit (same as sample_scale_nm)

pipeline.ewald.qx = all_qx;
pipeline.ewald.qy = all_qy;
pipeline.ewald.qz = all_qz;

% ============================================================
% ============================================================
%   SECTION 8 — FIGURES
% ============================================================
% ============================================================

% Build a red-white-blue colormap manually (MATLAB compatible)
n_cmap = 256;
ramp_down = linspace(1, 0, n_cmap/2)';
ramp_up   = linspace(0, 1, n_cmap/2)';
rwb_cmap  = [ramp_up, ramp_up, ones(n_cmap/2,1); ...
             ones(n_cmap/2,1), ramp_down, ramp_down];

% ---- Figure 1: 3D Ewald Sphere ----
if show_ewald_sphere
figure('Units','normalized','OuterPosition',[0.02 0.30 0.45 0.58],'Color','w');
hold on;

% Draw full Ewald sphere shells for min and max theta
% Each shell is a sphere of radius k_mag centered at -k_i
n_sphere = 60;
[sph_u, sph_v] = meshgrid(linspace(0,2*pi,n_sphere), linspace(0,pi,n_sphere));

for it = [1, N_theta]
    theta_r = theta_scan_rad(it);
    % Center of Ewald sphere in q-space = -k_i
    cx = -k_mag * sin(theta_r);   % phi=0 for display
    cy = 0;
    cz = k_mag * cos(theta_r);

    % Sphere surface points
    sx = k_mag * sin(sph_v) .* cos(sph_u) + cx;
    sy = k_mag * sin(sph_v) .* sin(sph_u) + cy;
    sz = k_mag * cos(sph_v) + cz;

    % Only show the hemisphere that falls in accessible q-space (qz > 0)
    sz(sz < 0) = NaN;

    surf(sx, sy, sz, 'FaceAlpha', 0.08, 'EdgeColor', 'none', ...
         'FaceColor', [0.5 0.7 1.0]);
end

% Draw Ewald circle outlines in qx-qz plane for each theta
colors_theta = jet(N_theta);
for it = 1:N_theta
    theta_r = theta_scan_rad(it);
    t_circ  = linspace(0, 2*pi, 300);
    cx = -k_mag * sin(theta_r);
    cz =  k_mag * cos(theta_r);
    qx_c = k_mag * cos(t_circ) + cx;
    qz_c = k_mag * sin(t_circ) + cz;
    % Only draw accessible arc (qz >= 0)
    qz_c(qz_c < 0) = NaN;
    plot3(qx_c, zeros(size(qx_c)), qz_c, '-', ...
          'Color', colors_theta(it,:), 'LineWidth', 1.5);
end

% Plot measured q-points as small dots colored by phi angle
ang_idx_pts = zeros(numel(all_qx), 1);
pts_per = round(numel(all_qx) / N_angles);
for ai = 1:N_angles
    i1 = (ai-1)*pts_per + 1;
    i2 = min(ai*pts_per, numel(all_qx));
    if i1 <= numel(all_qx) && i1 <= i2
        ang_idx_pts(i1:i2) = ai;
    end
end
scatter3(all_qx, all_qy, all_qz, 3, ang_idx_pts, 'filled');

% Origin = reciprocal lattice origin
scatter3(0, 0, 0, 120, 'r', 'filled');
text(0.05, 0, 0, '\Gamma', 'FontSize', 11, 'Color', 'r', 'FontWeight', 'bold');

% Draw q-space axes
ax_len = k_mag * 1.1;
quiver3(0,0,0, ax_len,0,0, 0, 'k', 'LineWidth', 1.2, 'MaxHeadSize', 0.3);
quiver3(0,0,0, 0,ax_len,0, 0, 'k', 'LineWidth', 1.2, 'MaxHeadSize', 0.3);
quiver3(0,0,0, 0,0,ax_len, 0, 'k', 'LineWidth', 1.2, 'MaxHeadSize', 0.3);
text(ax_len*1.05, 0, 0, 'q_x', 'FontSize', 10, 'FontWeight', 'bold');
text(0, ax_len*1.05, 0, 'q_y', 'FontSize', 10, 'FontWeight', 'bold');
text(0, 0, ax_len*1.05, 'q_z', 'FontSize', 10, 'FontWeight', 'bold');

% Legend entries for theta shells
legend_str = {};
for it = [1, N_theta]
    legend_str{end+1} = sprintf('Ewald shell \\theta=%.0f°', theta_scan_deg(it));
end

colormap(gca, jet(N_angles));
cb = colorbar;
cb.Label.String = 'Angle index (each colour = one theta/phi pair)';
cb.FontSize = 8;

xlabel('q_x  (rad/nm)', 'FontSize', 11);
ylabel('q_y  (rad/nm)', 'FontSize', 11);
zlabel('q_z  (rad/nm)  [depth sensitivity]', 'FontSize', 11);
title({sprintf('Ewald Sphere Coverage  |  E = %.0f eV  |  \\lambda = %.4f nm', ...
       photon_energy_eV, lambda_nm), ...
       sprintf('%d theta steps (%.0f°–%.0f°)  x  %d phi positions  =  %d total patterns', ...
       N_theta, theta_min_deg, theta_max_deg, N_phi, N_angles)}, ...
      'FontSize', 10);

% Annotation: missing wedge label
text(0, 0, -k_mag*0.3, 'missing wedge', 'FontSize', 8, ...
     'Color', [0.5 0.5 0.5], 'HorizontalAlignment', 'center', 'FontAngle', 'italic');

grid on; axis equal;
view(30, 22);

hold off;

end % show_ewald_sphere

% ---- Figure 2: 2theta map and q_z map on detector ----
if show_detector_maps
% Recompute for middle theta, first phi for display
theta_mid = theta_scan_rad(round(N_theta/2));
phi_mid   = phi_scan_rad(1);
k_ix_m = k_mag * sin(theta_mid) * cos(phi_mid);
k_iy_m = k_mag * sin(theta_mid) * sin(phi_mid);
k_iz_m = -k_mag * cos(theta_mid);

kfx_m = k_ix_m + det_u / det_distance_nm * k_mag;
kfy_m = k_iy_m + det_v / det_distance_nm * k_mag;
kfz_sq_m = k_mag^2 - kfx_m.^2 - kfy_m.^2;
valid_m = kfz_sq_m > 0;
kfz_m = zeros(size(kfz_sq_m));
kfz_m(valid_m) = sqrt(kfz_sq_m(valid_m));
kfn = sqrt(kfx_m.^2 + kfy_m.^2 + kfz_m.^2); kfn(kfn<1e-12)=1;
kfx_m=(kfx_m./kfn).*k_mag; kfy_m=(kfy_m./kfn).*k_mag; kfz_m=(kfz_m./kfn).*k_mag;

cos2th_m = (kfx_m.*k_ix_m + kfy_m.*k_iy_m + kfz_m.*k_iz_m)./k_mag^2;
two_theta_disp = acosd(max(min(cos2th_m,1),-1));
qz_disp = kfz_m - k_iz_m;

figure('Units','normalized','OuterPosition',[0.52 0.35 0.44 0.55],'Color','w');

subplot(1,2,1);
imagesc(rad2deg(linspace(-atan(detector_Npx/2*pixel_size_nm/det_distance_nm), ...
                          atan(detector_Npx/2*pixel_size_nm/det_distance_nm), detector_Npx)), ...
        rad2deg(linspace(-atan(detector_Npy/2*pixel_size_nm/det_distance_nm), ...
                          atan(detector_Npy/2*pixel_size_nm/det_distance_nm), detector_Npy)), ...
        two_theta_disp);
axis image; colormap(gca,'hot'); colorbar;
xlabel('Detector angle x (deg)','FontSize',9);
ylabel('Detector angle y (deg)','FontSize',9);
title(sprintf('2\\theta map  |  \\theta_{inc}=%.1f deg', rad2deg(theta_mid)),'FontSize',10);

subplot(1,2,2);
imagesc(qz_disp);
axis image; colormap(gca,'cool'); colorbar;
xlabel('Detector pixel x','FontSize',9);
ylabel('Detector pixel y','FontSize',9);
title('q_z map (rad/nm)  — depth sensitivity per pixel','FontSize',10);

sgtitle(sprintf('Detector Geometry  |  E=%.0f eV  |  det dist=%.0f mm  |  pixel=%.0f um', ...
        photon_energy_eV, det_distance_mm, pixel_size_um), 'FontSize', 11);

end % show_detector_maps

% ---- Figure 3: Example I+, I-, b+, b- at middle angle ----
if show_example_patterns
mid_ang = round(N_angles/2);
figure('Units','normalized','OuterPosition',[0.02 0.02 0.55 0.30],'Color','w');

subplot(1,4,1);
imagesc(log10(I_plus_all(:,:,mid_ang)+1));
axis image; colormap(gca,'turbo'); colorbar;
title(sprintf('I+ (left circ)\n\\theta=%.1f\\phi=%.1f', ...
      angle_list(mid_ang).theta_deg, angle_list(mid_ang).phi_deg),'FontSize',9);

subplot(1,4,2);
imagesc(log10(I_minus_all(:,:,mid_ang)+1));
axis image; colormap(gca,'turbo'); colorbar;
title('I- (right circ)','FontSize',9);

subplot(1,4,3);
imagesc(log10(abs(b_plus_all(:,:,mid_ang))+1));
axis image; colormap(gca,'gray'); colorbar;
title('b+ = structural signal','FontSize',9);

subplot(1,4,4);
imagesc(b_minus_all(:,:,mid_ang));
axis image; colormap(gca, rwb_cmap); colorbar;
title('b- = XMCD magnetic signal','FontSize',9);

sgtitle(sprintf('Example Patterns  |  Angle %d/%d  |  \\theta=%.1f deg  \\phi=%.1f deg', ...
        mid_ang, N_angles, angle_list(mid_ang).theta_deg, angle_list(mid_ang).phi_deg), ...
        'FontSize',10);

end % show_example_patterns

% ---- Figure 4: 3D reciprocal space coverage map ----
if show_qspace_coverage
figure('Units','normalized','OuterPosition',[0.60 0.02 0.38 0.30],'Color','w');
hold on;

% Show coverage as scatter colored by angle index
ang_color = zeros(numel(all_qx),1);
pts_per_ang = numel(all_qx) / N_angles;
for ai = 1:N_angles
    i1 = round((ai-1)*pts_per_ang)+1;
    i2 = min(round(ai*pts_per_ang), numel(all_qx));
    if i1 <= i2
        ang_color(i1:i2) = ai;
    end
end
scatter3(all_qx, all_qy, all_qz, 1, ang_color, 'filled');

xlabel('q_x','FontSize',9); ylabel('q_y','FontSize',9); zlabel('q_z','FontSize',9);
title(sprintf('Reciprocal Space Coverage  |  %d angle pairs\nColour = angle index  |  Gaps = missing wedge', ...
      N_angles),'FontSize',9);
colormap(gca,'hsv'); colorbar;
grid on; view(20,30); axis equal;

hold off;

end % show_qspace_coverage

% ============================================================
% ============================================================
%   SECTION 9 — PRINT GEOMETRY SUMMARY TO CONSOLE
% ============================================================
% ============================================================

fprintf('============================================================\n');
fprintf('GEOMETRY SUMMARY\n');
fprintf('============================================================\n');
fprintf('Scattering geometry : REFLECTION / DIFFRACTION\n');
fprintf('Polarization        : Left/Right circular (XMCD)\n');
fprintf('Photon energy       : %.1f eV  (lambda = %.4f nm)\n', ...
        photon_energy_eV, lambda_nm);
fprintf('Wavevector |k|      : %.4f rad/nm\n', k_mag);
fprintf('Detector distance   : %.1f mm\n', det_distance_mm);
fprintf('Pixel size          : %.1f um  (%.1f nm)\n', pixel_size_um, pixel_size_nm);
fprintf('Detector size       : %d x %d pixels  (%.1f x %.1f mm)\n', ...
        detector_Npx, detector_Npy, ...
        detector_Npx*pixel_size_um/1e3, detector_Npy*pixel_size_um/1e3);
fprintf('Theta range         : %.1f to %.1f deg\n', theta_min_deg, theta_max_deg);
fprintf('Phi positions       : '); fprintf('%.1f  ', phi_angles_deg); fprintf('deg\n');
fprintf('Total measurements  : %d\n', N_angles);
fprintf('Absorption length   : %.1f nm\n', absorption_length_nm);
fprintf('XMCD asymmetry      : %.2f\n', xmcd_asymmetry);
fprintf('\nq-space range covered:\n');
fprintf('  qx : [%.4f  %.4f] rad/nm\n', min(all_qx), max(all_qx));
fprintf('  qy : [%.4f  %.4f] rad/nm\n', min(all_qy), max(all_qy));
fprintf('  qz : [%.4f  %.4f] rad/nm\n', min(all_qz), max(all_qz));
fprintf('\nDepth resolution estimate:\n');
dqz = max(all_qz) - min(all_qz);
if dqz > 0
    dz_res = 2*pi / dqz;
    fprintf('  delta_qz = %.4f rad/nm  ->  depth resolution ~ %.2f nm\n', dqz, dz_res);
end
fprintf('  Absorption limit : %.1f nm (soft limit on reconstructable depth)\n', ...
        absorption_length_nm);
fprintf('\nDepth ambiguity note:\n');
fprintf('  Layers at same (x,y) but different z contribute to same\n');
fprintf('  detector pixel. These are separated by varying theta,\n');
fprintf('  which shifts the Ewald sphere and changes q_z per pixel.\n');
fprintf('  Missing wedge (theta < %.1f deg) leaves a gap in q_z\n', theta_min_deg);
fprintf('  that limits depth resolution. Wider theta range = better.\n');
fprintf('============================================================\n\n');

fprintf('--- Module 2 complete ---\n');
fprintf('pipeline.meas  : I+, I-, b+, b-  [%d x %d x %d angles]\n', ...
        detector_Npy, detector_Npx, N_angles);
fprintf('pipeline.geom  : angles, q-vectors, detector geometry\n');
fprintf('pipeline.ewald : 3D q-space coverage\n');
fprintf('Pass pipeline to Module 3 (sensing matrix).\n\n');

% ============================================================
% ============================================================
%   APPROXIMATIONS IN THIS MODULE — READ BEFORE USING
% ============================================================
%
% 1. FULL 2D FFT FORWARD MODEL (implemented):
%    Each layer's magnetic projection is Fourier transformed
%    via fft2 and the result resampled to the detector grid.
%    The q_z depth phase is applied pixel-wise using q_z_map.
%    This correctly captures spatial structure within each layer.
%
% 2. FFT GRID MISMATCH:
%    Volume grid (Ny x Nx) resampled to detector (Npy x Npx)
%    using imresize. A proper simulation would evaluate the FFT
%    at the exact (qx,qy) of each detector pixel.
%    Acceptable for current testing — fix in Stage 1 roadmap.
%
% 3. BORN APPROXIMATION:
%    Single scattering only. Multiple scattering between layers
%    is neglected. Valid when absorption is strong (soft X-rays)
%    and layers are thin.
%
% 4. STRUCTURAL FACTOR:
%    Charge scattering amplitude set to uniform (=1 per layer).
%    Acceptable since b- isolates the magnetic signal.
%
% 5. FORM FACTOR:
%    Magnetic form factor f_mag(q) assumed constant = xmcd_asymmetry.
%    Add resonant falloff after core code is performing (Stage 1).
%
% ============================================================
% ============================================================
%   BEAMLINE REFERENCE — ALS COSMIC 7.0.1.1
% ============================================================
%
% Facility   : Advanced Light Source (ALS)
%              Lawrence Berkeley National Laboratory
%              1 Cyclotron Road, Berkeley, CA 94720
%
% Beamline   : 7.0.1.1  COSMIC Scattering
%              Contact: Sujoy Roy  sroy@lbl.gov  510-486-7438
%
% Energy     : 250 – 2500 eV  (soft X-ray, EPU undulator)
%              Source: ALS beamline directory, als.lbl.gov/beamlines/7-0-1-1
%
% Detector   : VeryFastCCD
%              Pixel size  : 48 um
%              Frame rate  : 5-10 kHz
%              Readout noise: ~20 electrons
%              Full well   : >4e5 electrons/pixel
%              QE at 285 eV: >85%
%              Source: COSMIC imaging paper, ResearchGate 326943759
%
% Technique  : Resonant coherent X-ray scattering
%              XMCD contrast (left/right circular polarization)
%              Reflection/diffraction geometry
%              Source: ALS COSMIC Halbach Award article, 2024
%
% Magnetic edges commonly used at this beamline:
%              Fe L3 : 707 eV  (lambda = 1.754 nm)
%              Co L3 : 779 eV  (lambda = 1.592 nm)
%              Ni L3 : 853 eV  (lambda = 1.454 nm)
%              Mn L3 : 641 eV  (lambda = 1.934 nm)
%
% NOTE: ALS is currently undergoing the ALS-U upgrade.
%       Beamline 7.0.1.1 will migrate to FLEXON at 10.0.1.
%       Verify operational status before scheduling beamtime.
%       All parameters above are USER CONTROLLED and should be
%       verified against your experimental logbook.
%
% ============================================================