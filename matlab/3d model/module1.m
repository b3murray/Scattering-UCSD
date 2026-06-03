% ============================================================
% ============================================================
%          MODULE 1 — 3D MAGNETIC VECTOR FIELD GENERATOR
% ============================================================
%
% Generates a true 3D magnetization volume:
%   Mx(Nx, Ny, Nz),  My(Nx, Ny, Nz),  Mz(Nx, Ny, Nz)
%
% Domain walls run through the bulk in z.
% Each z-layer can be independently tilted.
% Bloch/Neel assignment supports split, checkerboard, and random tiles.
%
% OUTPUT VARIABLES (passed to Module 2):
%   Mx, My, Mz   — 3D magnetization arrays  [Ny x Nx x Nz]
%   x, y, z_vec  — coordinate vectors
%   X, Y         — 2D meshgrid (for reference)
%   domain_params — struct of all key parameters
%
% FIGURES:
%   Figure 1 — 3D quiver (subsampled), color coded by region + walls
%   Figure 2 — Slice inspector: XY / XZ / YZ centre slices
%   Figure 3 — Domain wall isosurface (layman overview)
%
% ============================================================

% Note: workspace clearing is handled by master_pipeline.m
% Do not add clear/clc here — it will wipe pipeline and algo_mode

% ============================================================
% ============================================================
%              SECTION 1 — GRID AND PHYSICAL SIZE
% ============================================================
% ============================================================

x_range = [-10 10];     % physical X extent (in units below)
y_range = [-10 10];     % physical Y extent
z_range = [0   6];      % physical Z extent (depth into bulk)

% Physical scale — what does one unit in x_range/y_range correspond to?
% This sets the spatial frequency content of the volume to match the detector.
%
% The detector q range at ALS COSMIC (150mm distance, 48um pixels, 512px):
%   qx_max_det ≈ 0.29 rad/nm
%
% For the volume FFT to cover the same q range:
%   qx_max_vol = pi / dx_vol_nm  must equal  qx_max_det
%   dx_vol_nm = pi / qx_max_det = pi / 0.29 ≈ 10.8 nm
%   sample_scale_nm = dx_vol_nm / dx_units = 10.8 / (20/139) ≈ 75 nm
%
% This means the total field of view is 75 * 20 = 1500 nm = 1.5 um
% which is physically reasonable for magnetic domain structures.
%
% If you change det_distance_mm or pixel_size_um in module2,
% recalculate: sample_scale_nm = pi/qx_max_det / (x_range_width/Nx)
sample_scale_nm = 75.0;   % nm per x_range/y_range unit

Nx = 140;               % grid points along X
Ny = 140;               % grid points along Y
Nz = 3;                 % number of z layers

% ============================================================
% ============================================================
%              SECTION 2 — DOMAIN GEOMETRY
% ============================================================
% ============================================================

domain_width = 3;       % width of each domain stripe (physical units)

% Direction stripes repeat:
%   'x' -> vertical stripes (vary left-to-right)
%   'y' -> horizontal stripes (vary bottom-to-top)
stripe_axis  = 'x';

% ============================================================
% ============================================================
%              SECTION 3 — DOMAIN WALL CONTROLS
% ============================================================
% ============================================================

% Sharpness of the wall transition
%   smaller -> very sharp wall
%   larger  -> smooth / thick wall
wall_smoothness = 0.007;

% Mathematical profile for wall shape:
%   'tanh' -> smooth, physically realistic (recommended)
%   'atan' -> alternative profile
profile_type = 'tanh';

% Physical width of the wall region (for visualization mask only)
wall_visual_thick = 0.35;

% Optional: multiply wall_smoothness to make walls visually thicker
use_wall_thick_multiplier = true;
wall_smooth_multiplier    = 3.0;

% ============================================================
% ============================================================
%              SECTION 4 — SPIN TEXTURE
% ============================================================
% ============================================================

% Angular resolution — controls how many distinct spin orientations appear
%   pi/4  -> 4 steps (coarse)
%   pi/8  -> 8 steps
%   0.05  -> ~63 steps (nearly continuous)
epsilon   = pi/4;

% Rotation direction through the wall:
%   +1 -> counterclockwise
%   -1 -> clockwise
chirality = +1;

% ============================================================
%   BULK SPIN ORIENTATION CONTROLS
% ============================================================
%
% These set where the spins point IN THE BULK (between walls).
% The wall rotation (epsilon, chirality) is applied ON TOP of this.
%
% theta_bulk_deg — polar angle from the +z axis
%   0   -> spin UP   (all spins pointing out of plane, +z)
%   180 -> spin DOWN (all spins pointing into plane, -z)
%   90  -> spins lying flat in-plane
%   45  -> spins canted 45 degrees from vertical
%
theta_bulk_deg = 0;

% phi_bulk_deg — azimuthal angle in the xy-plane
%   only has effect when theta_bulk_deg is not 0 or 180
%   0   -> in-plane component points toward +x
%   90  -> in-plane component points toward +y
%   45  -> diagonal
%
phi_bulk_deg = 0;

% domain_contrast — controls what the SECOND domain does
%   'alternating' -> domains alternate: theta_bulk and theta_bulk+180
%                    this is the classic up/down domain pattern
%   'uniform'     -> ALL domains point the same direction (ferromagnet)
%                    walls still rotate but bulk is uniform
%   'orthogonal'  -> second domain is rotated 90 deg from first
%                    useful for vortex-like or flux-closure patterns
%
domain_contrast = 'alternating';

% ============================================================
%   BLOCH / NEEL ASSIGNMENT MODE
% ============================================================
%
%   'split'  -> simple half-plane division (vertical or horizontal)
%   'tiles'  -> grid of rectangular regions (checker or random)
%
assignment_mode = 'split';

% --- Split options (used when assignment_mode = 'split') ---
%   'vertical'   -> left = Neel, right = Bloch
%   'horizontal' -> bottom = Neel, top = Bloch
split_type     = 'vertical';
split_location = 0;             % x (or y) coordinate of the dividing line

% --- Tile options (used when assignment_mode = 'tiles') ---
tile_Nx = 6;                    % number of tile columns
tile_Ny = 6;                    % number of tile rows
%   'checker' -> alternating checkerboard
%   'random'  -> random assignment (controlled by seed below)
tile_pattern     = 'checker';
tile_random_seed = 1;

% ============================================================
% ============================================================
%              SECTION 5 — Z-LAYER / DEPTH CONTROLS
% ============================================================
% ============================================================

% --- Z layer positions ---
% 'linear'  -> evenly spaced from z_range(1) to z_range(2)
% 'custom'  -> use z_layers array below (must have exactly Nz values)
z_spacing_mode = 'linear';
z_layers_custom = [0 3 6];      % only used if z_spacing_mode = 'custom'

% ============================================================
%   PER-LAYER TILT CONTROLS
% ============================================================
%
% Each z-layer plane can be independently tilted in X and Y.
% This simulates real bulk materials where layers are not perfectly flat.
%
% use_per_layer_tilt = false -> use the same global tilt for all layers
% use_per_layer_tilt = true  -> use the arrays below (must have Nz values)
%
% Global tilt (used for all layers when use_per_layer_tilt = false,
%              also used as fallback if arrays are too short):
global_tilt_ax = 0.05;          % X-tilt applied to all layers
global_tilt_ay = 0.02;          % Y-tilt applied to all layers

% Per-layer tilt (only used when use_per_layer_tilt = true)
% Add or remove values to match Nz — one entry per layer.
% If you set Nz = 5, provide 5 values in each array.
use_per_layer_tilt  = false;
layer_tilt_ax = [0.08, -0.05, 0.03];   % X-tilt per layer
layer_tilt_ay = [0.04,  0.10, 0.00];   % Y-tilt per layer

% ============================================================
%   INTER-LAYER COHERENCE
% ============================================================
%
% Controls how similar adjacent z-layers are to each other.
%
% z_coherence_mode:
%   'copy'        -> identical at every z (perfect coherence)
%   'advect'      -> domains drift laterally with z
%   'morph'       -> domain parameters change slowly with z
%   'hybrid'      -> advect + morph + optional rare events
%   'independent' -> fully independent per layer (stress test)
%
z_coherence_mode  = 'hybrid';
z_corr_strength   = 0.85;       % [0..1]  1 = highly coherent, 0 = independent
z_advect_sigma    = 0.25;       % lateral drift step size per layer (advect/hybrid)

% Parameter drift per layer (morph/hybrid)
z_width_jitter_frac   = 0.06;   % fractional jitter of domain_width
z_wall_jitter_frac    = 0.08;   % fractional jitter of wall_smoothness
z_split_jitter_abs    = 0.15;   % absolute jitter of split_location

% Chirality behavior across z
%   'locked'      -> same chirality every layer
%   'random_walk' -> occasionally flips with probability below
z_chirality_mode      = 'locked';
z_chirality_flip_prob = 0.05;

% Random seeding across layers
%   'layer_offset' -> seed = z_base_seed + layer index
%   'locked'       -> same seed every layer
z_seed_mode  = 'layer_offset';
z_base_seed  = 123;

% Rare events (hybrid mode only)
% Localized patches of perturbed magnetization — stress-tests reconstruction
z_event_rate       = 0.10;      % probability per layer of an event occurring
z_event_patch_frac = 0.18;      % patch size as fraction of field of view
z_event_strength   = 0.55;      % perturbation strength [0..1]

% ============================================================
% ============================================================
%              SECTION 6 — VISUALIZATION CONTROLS
% ============================================================
% ============================================================

% ---- Figure toggles — set false to skip ----
% Use the exist-guard pattern so a caller (e.g. Generate_training_data)
% can set these to false BEFORE running this module and have it respected.
if ~exist('show_fig1_3d_volume',   'var'), show_fig1_3d_volume   = true; end
if ~exist('show_fig2_single_layer','var'), show_fig2_single_layer = true; end

% Arrow density — controls spacing between arrows
%   0.35 matches the original codes (recommended starting point)
%   increase toward 0.5 for denser fill, decrease for sparser
quiver_density_3d  = 0.35;      % Figure 1 — 3D volume
quiver_density_2d  = 0.35;      % Figure 2 — single layer detail

% Which layer to show in Figure 2 (1 = first, Nz = deepest)
fig2_layer = 1;

% Arrow scale — set to 0 to let MATLAB auto-scale (no overlap)
% Use a small positive number like 0.4-0.6 for consistent sizing
% DO NOT set above 1.0 — arrows will overlap into adjacent cells
arrow_scale = 0.5;

% Arrow line thickness
arrow_width = 1.5;

% Colors — matching original code conventions
color_neel  = [0.00 0.35 0.85]; % blue   — Neel domain arrows
color_bloch = [0.85 0.10 0.10]; % red    — Bloch domain arrows
color_wall  = [0.95 0.80 0.00]; % yellow — domain wall arrows
color_wall_scatter = [0.95 0.80 0.0];

wall_marker_size = 12;
wall_alpha       = 0.85;

% Figure sizes — moderate, side by side, easy to rotate
% [left bottom width height] in normalized screen units
fig1_pos = [0.03 0.25 0.43 0.50];   % 3D volume — left
fig2_pos = [0.53 0.25 0.43 0.50];   % single layer detail — right

% ============================================================
% ============================================================
%   GENERATOR OVERRIDE — applied if gen_params struct exists
%   Allows generate_training_data.m to randomize parameters
%   without editing module1 defaults. Each field is optional.
% ============================================================
% ============================================================
if exist('gen_params', 'var') && isstruct(gen_params)
    if isfield(gen_params,'Nx'),              Nx = gen_params.Nx;                           end
    if isfield(gen_params,'Ny'),              Ny = gen_params.Ny;                           end
    if isfield(gen_params,'Nz'),              Nz = gen_params.Nz;                           end
    if isfield(gen_params,'z_range'),         z_range = gen_params.z_range;                 end
    if isfield(gen_params,'sample_scale_nm'), sample_scale_nm = gen_params.sample_scale_nm; end
    if isfield(gen_params,'domain_width'),    domain_width = gen_params.domain_width;       end
    if isfield(gen_params,'stripe_axis'),     stripe_axis = gen_params.stripe_axis;         end
    if isfield(gen_params,'wall_smoothness'), wall_smoothness = gen_params.wall_smoothness; end
    if isfield(gen_params,'theta_bulk_deg'),  theta_bulk_deg = gen_params.theta_bulk_deg;   end
    if isfield(gen_params,'phi_bulk_deg'),    phi_bulk_deg = gen_params.phi_bulk_deg;       end
    if isfield(gen_params,'domain_contrast'), domain_contrast = gen_params.domain_contrast; end
    if isfield(gen_params,'chirality'),       chirality = gen_params.chirality;             end
    if isfield(gen_params,'split_type'),      split_type = gen_params.split_type;           end
    if isfield(gen_params,'split_location'),  split_location = gen_params.split_location;   end
    if isfield(gen_params,'epsilon'),         epsilon = gen_params.epsilon;                 end
end

% ============================================================
% ============================================================
%              BUILD COORDINATE GRIDS
% ============================================================
% ============================================================

x = linspace(x_range(1), x_range(2), Nx);
y = linspace(y_range(1), y_range(2), Ny);

% Build z layer positions
if strcmpi(z_spacing_mode,'custom') && numel(z_layers_custom) == Nz
    z_vec = z_layers_custom(:).';
else
    z_vec = linspace(z_range(1), z_range(2), Nz);
end

[X, Y] = meshgrid(x, y);   % [Ny x Nx]

% Allocate 3D output arrays
Mx = zeros(Ny, Nx, Nz);
My = zeros(Ny, Nx, Nz);
Mz = zeros(Ny, Nx, Nz);

% Wall mask 3D (for visualization)
Wall_mask = false(Ny, Nx, Nz);

% Store per-layer region masks for correct color coding in figures
Mask_neel  = false(Ny, Nx, Nz);
Mask_bloch = false(Ny, Nx, Nz);

% ============================================================
% ============================================================
%              PRECOMPUTE BASE PROFILE (layer 1 reference)
% ============================================================
% ============================================================

if lower(stripe_axis) == 'x'
    S_base = X; Smin_base = min(x);
else
    S_base = Y; Smin_base = min(y);
end

% ============================================================
% ============================================================
%              COMPUTE INTER-LAYER ADVECTION DRIFT
% ============================================================
% ============================================================

dx_drift = zeros(1, Nz);
dy_drift = zeros(1, Nz);

rng(z_base_seed);   % reproducible

for kk = 2:Nz
    dx_drift(kk) = dx_drift(kk-1) + z_advect_sigma * randn;
    dy_drift(kk) = dy_drift(kk-1) + z_advect_sigma * randn;
end

alpha_coh = max(0, min(1, z_corr_strength));

% ============================================================
% ============================================================
%              MAIN LOOP — BUILD 3D FIELD LAYER BY LAYER
% ============================================================
% ============================================================

base_dw   = domain_width;
base_ws   = wall_smoothness;
base_sl   = split_location;
base_chir = chirality;

Nlayers = max(1, round(pi / abs(epsilon)));

for kk = 1:Nz

    % --- Seed ---
    if strcmpi(z_seed_mode,'layer_offset')
        rng(z_base_seed + kk);
    end

    % --- Per-layer domain parameters (morph/hybrid) ---
    if strcmpi(z_coherence_mode,'copy')
        dw_k  = base_dw;
        ws_k  = base_ws;
        sl_k  = base_sl;
        chir_k = base_chir;
        Xk = X; Yk = Y;
    else
        dw_k  = base_dw  * (1 + (1-alpha_coh) * z_width_jitter_frac * randn);
        ws_k  = base_ws  * (1 + (1-alpha_coh) * z_wall_jitter_frac  * randn);
        sl_k  = base_sl  + (1-alpha_coh) * z_split_jitter_abs * randn;
        chir_k = base_chir;

        % Chirality random walk
        if strcmpi(z_chirality_mode,'random_walk') && rand < z_chirality_flip_prob
            chir_k = -base_chir;
            base_chir = chir_k;
        end

        % Lateral advection
        Xk = X; Yk = Y;
        if any(strcmpi(z_coherence_mode, {'advect','hybrid'}))
            Xk = X - dx_drift(kk);
            Yk = Y - dy_drift(kk);
        end
        if strcmpi(z_coherence_mode,'independent')
            dw_k  = base_dw  * (0.7 + 0.6*rand);
            ws_k  = base_ws  * (0.5 + rand);
            sl_k  = base_sl  + 4*(rand - 0.5);
        end
    end

    % --- Stripe coordinate ---
    if lower(stripe_axis) == 'x'
        Sk = Xk; Smin_k = min(x);
    else
        Sk = Yk; Smin_k = min(y);
    end

    domain_index_k = floor((Sk - Smin_k) / max(dw_k, 1e-9));
    s_mod_k        = mod((Sk - Smin_k), max(dw_k,1e-9)) - max(dw_k,1e-9)/2;

    % --- Compute bulk spin angles from user controls ---
    theta_bulk_rad = deg2rad(theta_bulk_deg);
    phi_bulk_rad   = deg2rad(phi_bulk_deg);

    % Base angle for domain 1 (even index)
    theta_dom1 = theta_bulk_rad;

    % Base angle for domain 2 (odd index) — depends on domain_contrast
    switch lower(domain_contrast)
        case 'uniform'
            theta_dom2 = theta_bulk_rad;       % same as domain 1
        case 'orthogonal'
            theta_dom2 = theta_bulk_rad + pi/2; % 90 deg offset
        otherwise % 'alternating'
            theta_dom2 = theta_bulk_rad + pi;   % flip to opposite pole
    end

    % Build theta_base_k: even domains get theta_dom1, odd get theta_dom2
    is_odd_domain = mod(domain_index_k, 2) == 1;
    theta_base_k  = theta_dom1 * ones(size(domain_index_k));
    theta_base_k(is_odd_domain) = theta_dom2;

    % --- Wall profile ---
    ws_eff_k = ws_k;
    if use_wall_thick_multiplier
        ws_eff_k = ws_k * wall_smooth_multiplier;
    end

    switch lower(profile_type)
        case 'atan', fk = (2/pi) * atan(s_mod_k / max(ws_eff_k,1e-6));
        otherwise,   fk = tanh(s_mod_k / max(ws_eff_k,1e-6));
    end

    % --- Wall mask for this layer ---
    wall_mask_k = abs(s_mod_k) < wall_visual_thick;
    Wall_mask(:,:,kk) = wall_mask_k;

    % --- Bloch / Neel assignment ---
    mask_bloch_k = false(size(X));

    switch lower(assignment_mode)
        case 'tiles'
            if strcmpi(tile_pattern,'random')
                rng(tile_random_seed);
                tile_assign = rand(tile_Ny, tile_Nx) > 0.5;
            else
                [ii,jj] = ndgrid(1:tile_Ny, 1:tile_Nx);
                tile_assign = mod(ii+jj,2) == 0;
            end
            tx = min(tile_Nx, max(1, floor((Xk-x_range(1))/diff(x_range)*tile_Nx)+1));
            ty = min(tile_Ny, max(1, floor((Yk-y_range(1))/diff(y_range)*tile_Ny)+1));
            for iy = 1:tile_Ny
                for ix = 1:tile_Nx
                    mask_bloch_k = mask_bloch_k | (tx==ix & ty==iy & tile_assign(iy,ix));
                end
            end
        otherwise
            if strcmpi(split_type,'horizontal')
                mask_bloch_k = (Yk >= sl_k);
            else
                mask_bloch_k = (Xk >= sl_k);
            end
    end
    mask_neel_k = ~mask_bloch_k;

    % --- Build spin field for this layer ---
    mx_k = zeros(Ny, Nx);
    my_k = zeros(Ny, Nx);
    mz_k = zeros(Ny, Nx);

    for k = 1:Nlayers
        theta_kz = theta_base_k + chir_k * k * epsilon * fk;

        % sin(theta_kz) gives the in-plane magnitude
        % phi_bulk_rad splits that between x and y components
        in_plane = sin(theta_kz);
        out_plane = cos(theta_kz);

        % Neel: rotates in the plane defined by phi_bulk and z
        mx_k(mask_neel_k) = mx_k(mask_neel_k) + in_plane(mask_neel_k) * cos(phi_bulk_rad);
        my_k(mask_neel_k) = my_k(mask_neel_k) + in_plane(mask_neel_k) * sin(phi_bulk_rad);
        mz_k(mask_neel_k) = mz_k(mask_neel_k) + out_plane(mask_neel_k);

        % Bloch: rotates in the plane perpendicular to phi_bulk
        % perpendicular azimuth = phi_bulk + pi/2
        phi_bloch = phi_bulk_rad + pi/2;
        mx_k(mask_bloch_k) = mx_k(mask_bloch_k) + in_plane(mask_bloch_k) * cos(phi_bloch);
        my_k(mask_bloch_k) = my_k(mask_bloch_k) + in_plane(mask_bloch_k) * sin(phi_bloch);
        mz_k(mask_bloch_k) = mz_k(mask_bloch_k) + out_plane(mask_bloch_k);
    end

    % --- Normalize to unit vectors ---
    mag_k = sqrt(mx_k.^2 + my_k.^2 + mz_k.^2);
    mag_k(mag_k < 1e-12) = 1;
    mx_k = mx_k ./ mag_k;
    my_k = my_k ./ mag_k;
    mz_k = mz_k ./ mag_k;

    % --- Rare event perturbation (hybrid) ---
    if strcmpi(z_coherence_mode,'hybrid') && rand < z_event_rate
        px = x_range(1) + rand*diff(x_range);
        py = y_range(1) + rand*diff(y_range);
        rx = z_event_patch_frac * diff(x_range);
        ry = z_event_patch_frac * diff(y_range);
        patch = (abs(X-px) <= rx/2) & (abs(Y-py) <= ry/2);

        ax=randn; ay=randn; az=randn;
        an = sqrt(ax^2+ay^2+az^2)+1e-12;
        ax=ax/an; ay=ay/an; az=az/an;

        mx_k(patch) = (1-z_event_strength)*mx_k(patch) + z_event_strength*ax;
        my_k(patch) = (1-z_event_strength)*my_k(patch) + z_event_strength*ay;
        mz_k(patch) = (1-z_event_strength)*mz_k(patch) + z_event_strength*az;

        nn = sqrt(mx_k(patch).^2+my_k(patch).^2+mz_k(patch).^2)+1e-12;
        mx_k(patch) = mx_k(patch)./nn;
        my_k(patch) = my_k(patch)./nn;
        mz_k(patch) = mz_k(patch)./nn;
    end

    % --- Store in 3D volume ---
    Mx(:,:,kk) = mx_k;
    My(:,:,kk) = my_k;
    Mz(:,:,kk) = mz_k;
    Mask_neel(:,:,kk)  = mask_neel_k;
    Mask_bloch(:,:,kk) = mask_bloch_k;

end

fprintf('3D field built: [%d x %d x %d]  (%d z-layers)\n', Ny, Nx, Nz, Nz);

% ============================================================
% ============================================================
%   COMPUTE PER-LAYER Z SURFACE WITH TILT
% ============================================================
% Tilt is applied to visualization only — does NOT affect Mx,My,Mz values.
% ============================================================

% Build per-layer tilt arrays (pad with global value if too short)
ax_list = global_tilt_ax * ones(1, Nz);
ay_list = global_tilt_ay * ones(1, Nz);

if use_per_layer_tilt
    for kk = 1:Nz
        if kk <= numel(layer_tilt_ax), ax_list(kk) = layer_tilt_ax(kk); end
        if kk <= numel(layer_tilt_ay), ay_list(kk) = layer_tilt_ay(kk); end
    end
end

% ============================================================
% ============================================================
%   FIGURE 1 — 3D QUIVER VOLUME
%   Color: Blue=Neel  Red=Bloch  Yellow=Wall
% ============================================================
% ============================================================

if show_fig1_3d_volume

step3 = max(1, round(1/quiver_density_3d));

figure('Units','normalized','OuterPosition',fig1_pos,'Color','w');
hold on;

% Faint reference grid at bottom layer
[Xp,Yp] = meshgrid(linspace(x_range(1),x_range(2),20), ...
                    linspace(y_range(1),y_range(2),20));
surf(Xp, Yp, z_vec(1)*ones(size(Xp)), ...
    'FaceAlpha',0.04,'EdgeColor',[0.75 0.75 0.75],'EdgeAlpha',0.35);

% Vertical corner lines — depth perception
corners = [x_range(1) y_range(1); x_range(2) y_range(1);
           x_range(2) y_range(2); x_range(1) y_range(2)];
for c = 1:4
    plot3([corners(c,1) corners(c,1)],[corners(c,2) corners(c,2)], ...
          [z_vec(1) z_vec(end)],'Color',[0.7 0.7 0.7],'LineWidth',0.8);
end

for kk = 1:Nz

    ax_k = ax_list(kk);
    ay_k = ay_list(kk);

    mx_k = Mx(:,:,kk);
    my_k = My(:,:,kk);
    mz_k = Mz(:,:,kk);

    % Use stored per-layer masks — correct for advection/morph
    mn_k = Mask_neel(:,:,kk);
    mb_k = Mask_bloch(:,:,kk);
    wm_k = Wall_mask(:,:,kk);

    % Subsample grid
    Xs3 = X(1:step3:end, 1:step3:end);
    Ys3 = Y(1:step3:end, 1:step3:end);
    Zs3 = z_vec(kk) + ax_k*Xs3 + ay_k*Ys3;

    mxs = mx_k(1:step3:end,1:step3:end);
    mys = my_k(1:step3:end,1:step3:end);
    mzs = mz_k(1:step3:end,1:step3:end);

    idx_n = mn_k(1:step3:end,1:step3:end) & ~wm_k(1:step3:end,1:step3:end);
    idx_b = mb_k(1:step3:end,1:step3:end) & ~wm_k(1:step3:end,1:step3:end);
    idx_w = wm_k(1:step3:end,1:step3:end);

    % Neel — blue
    if any(idx_n(:))
        quiver3(Xs3(idx_n), Ys3(idx_n), Zs3(idx_n), ...
                mxs(idx_n), mys(idx_n), mzs(idx_n), ...
                arrow_scale, 'Color', color_neel, 'LineWidth', arrow_width);
    end
    % Bloch — red
    if any(idx_b(:))
        quiver3(Xs3(idx_b), Ys3(idx_b), Zs3(idx_b), ...
                mxs(idx_b), mys(idx_b), mzs(idx_b), ...
                arrow_scale, 'Color', color_bloch, 'LineWidth', arrow_width);
    end
    % Wall — yellow (slightly thicker)
    if any(idx_w(:))
        quiver3(Xs3(idx_w), Ys3(idx_w), Zs3(idx_w), ...
                mxs(idx_w), mys(idx_w), mzs(idx_w), ...
                arrow_scale, 'Color', color_wall, 'LineWidth', arrow_width+0.5);
    end

    % Wall scatter dots — reinforce wall position
    Xwk = X(wm_k); Ywk = Y(wm_k);
    Zwk = z_vec(kk) + ax_k*Xwk + ay_k*Ywk;
    scatter3(Xwk, Ywk, Zwk, wall_marker_size, color_wall_scatter, ...
             'filled','MarkerFaceAlpha',wall_alpha);
end

xlabel('X','FontSize',11); ylabel('Y','FontSize',11);
zlabel('Z  (depth into bulk)','FontSize',11);
title(sprintf('3D Magnetic Volume  |  %d layers  |  coherence: %s', ...
      Nz, z_coherence_mode),'FontSize',12);
legend({'Néel (blue)','Bloch (red)','Wall (yellow)'}, ...
       'Location','northeast','FontSize',9);
grid on; axis equal; view(35,28);
rotate3d on;
hold off;

end % show_fig1_3d_volume

% ============================================================
% ============================================================
%   FIGURE 2 — SINGLE LAYER DETAIL VIEW
% ============================================================
% ============================================================

if show_fig2_single_layer

% Clamp layer index to valid range
fig2_layer = max(1, min(Nz, fig2_layer));

mx_d  = Mx(:,:,fig2_layer);
my_d  = My(:,:,fig2_layer);
mz_d  = Mz(:,:,fig2_layer);
mn_d  = Mask_neel(:,:,fig2_layer);
mb_d  = Mask_bloch(:,:,fig2_layer);
wm_d  = Wall_mask(:,:,fig2_layer);

ax_d  = ax_list(fig2_layer);
ay_d  = ay_list(fig2_layer);

step2 = max(1, round(1/quiver_density_2d));

figure('Units','normalized','OuterPosition',fig2_pos,'Color','w');
hold on;

% Faint background plane
[Xp2,Yp2] = meshgrid(linspace(x_range(1),x_range(2),24), ...
                      linspace(y_range(1),y_range(2),24));
Zp2 = z_vec(fig2_layer) + ax_d*Xp2 + ay_d*Yp2;
surf(Xp2, Yp2, Zp2, 'FaceAlpha',0.05,'EdgeColor',[0.75 0.75 0.75]);

% Wall scatter dots — prominent
Xwd = X(wm_d); Ywd = Y(wm_d);
Zwd = z_vec(fig2_layer) + ax_d*Xwd + ay_d*Ywd;
scatter3(Xwd, Ywd, Zwd, wall_marker_size*1.5, color_wall_scatter, ...
         'filled','MarkerFaceAlpha',wall_alpha);

% Subsampled grid
Xs2 = X(1:step2:end,1:step2:end);
Ys2 = Y(1:step2:end,1:step2:end);
Zs2 = z_vec(fig2_layer) + ax_d*Xs2 + ay_d*Ys2;

mxs2 = mx_d(1:step2:end,1:step2:end);
mys2 = my_d(1:step2:end,1:step2:end);
mzs2 = mz_d(1:step2:end,1:step2:end);

idx_n2 = mn_d(1:step2:end,1:step2:end) & ~wm_d(1:step2:end,1:step2:end);
idx_b2 = mb_d(1:step2:end,1:step2:end) & ~wm_d(1:step2:end,1:step2:end);
idx_w2 = wm_d(1:step2:end,1:step2:end);

% Neel — blue
if any(idx_n2(:))
    quiver3(Xs2(idx_n2), Ys2(idx_n2), Zs2(idx_n2), ...
            mxs2(idx_n2), mys2(idx_n2), mzs2(idx_n2), ...
            arrow_scale, 'Color', color_neel, 'LineWidth', arrow_width);
end
% Bloch — red
if any(idx_b2(:))
    quiver3(Xs2(idx_b2), Ys2(idx_b2), Zs2(idx_b2), ...
            mxs2(idx_b2), mys2(idx_b2), mzs2(idx_b2), ...
            arrow_scale, 'Color', color_bloch, 'LineWidth', arrow_width);
end
% Wall — yellow
if any(idx_w2(:))
    quiver3(Xs2(idx_w2), Ys2(idx_w2), Zs2(idx_w2), ...
            mxs2(idx_w2), mys2(idx_w2), mzs2(idx_w2), ...
            arrow_scale, 'Color', color_wall, 'LineWidth', arrow_width+0.5);
end

xlabel('X','FontSize',11); ylabel('Y','FontSize',11);
zlabel('Z','FontSize',11);
title(sprintf('Single Layer Detail  |  Layer %d / %d  |  z = %.2f', ...
      fig2_layer, Nz, z_vec(fig2_layer)),'FontSize',12);
legend({'Néel (blue)','Bloch (red)','Wall (yellow)'},'Location','northeast','FontSize',9);
grid on; axis equal; view(35,28);
rotate3d on;
hold off;

end % show_fig2_single_layer

% ============================================================
% ============================================================
%   PACKAGE OUTPUT FOR MODULE 2
% ============================================================
% ============================================================

domain_params.x_range         = x_range;
domain_params.y_range         = y_range;
domain_params.z_range         = z_range;
domain_params.Nx              = Nx;
domain_params.Ny              = Ny;
domain_params.Nz              = Nz;
domain_params.domain_width    = domain_width;
domain_params.epsilon         = epsilon;
domain_params.chirality       = chirality;
domain_params.assignment_mode = assignment_mode;
domain_params.z_coherence_mode = z_coherence_mode;

% ============================================================
%   PACK PIPELINE STRUCT — passed to Module 2 and beyond
% ============================================================
pipeline.M.x        = Mx;
pipeline.M.y        = My;
pipeline.M.z        = Mz;
pipeline.M.x_vec    = x;
pipeline.M.y_vec    = y;
pipeline.M.z_vec    = z_vec;
pipeline.M.params   = domain_params;
pipeline.M.Wall_mask  = Wall_mask;
pipeline.M.Mask_neel  = Mask_neel;
pipeline.M.Mask_bloch = Mask_bloch;
pipeline.M.sample_scale_nm = sample_scale_nm;  % nm per x_range unit — for q-axis

fprintf('\n--- Module 1 complete ---\n');
fprintf('Output arrays: Mx, My, Mz  [%d x %d x %d]\n', Ny, Nx, Nz);
fprintf('Coordinate vectors: x (%d pts), y (%d pts), z_vec (%d pts)\n', Nx, Ny, Nz);
fprintf('Pass these to Module 2 (forward scattering model).\n');