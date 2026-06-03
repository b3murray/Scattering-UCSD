% ============================================================
% ============================================================
%   MODULE 3 — SENSING MATRIX
%   Angular Bank | Polarization Sensitivity | Rank Diagnostics
% ============================================================
%
% PURPOSE:
%   Builds the sensing matrix Pmat from the XMCD measurement
%   geometry. In XMCD reflection geometry the sensitivity vector
%   for each channel is the incident beam direction k_i_hat —
%   the detector is sensitive to the spin component ALONG the beam.
%
%   For each (theta, phi) measurement:
%       p_r = k_i_hat_r = [-sin(theta)*cos(phi), 
%                           -sin(theta)*sin(phi),
%                           -cos(theta)]
%
%   The sensing matrix Pmat is [N_angles x 3].
%   For full vector reconstruction we need rank(Pmat) = 3.
%   If rank < 3, one or more spin components are invisible
%   and cannot be reconstructed no matter how many iterations
%   RESIRE runs.
%
%   If rank < 3 or condition number is too high, this module
%   suggests specific additional phi angles that would fix it.
%
% INPUTS (from pipeline struct):
%   pipeline.geom.angle_list    — all (theta,phi) pairs + k_i
%   pipeline.geom.theta_deg     — theta scan array
%   pipeline.geom.phi_deg       — phi positions array
%   pipeline.meas.N_angles      — total number of channels
%
% OUTPUTS (added to pipeline struct):
%   pipeline.sensing.Pmat       — [N_angles x 3] sensing matrix
%   pipeline.sensing.Pinv       — pseudoinverse [3 x N_angles]
%   pipeline.sensing.rank       — rank of Pmat
%   pipeline.sensing.cond       — condition number of Pmat
%   pipeline.sensing.sv         — singular values
%   pipeline.sensing.p_vecs     — [N_angles x 3] unit sensitivity vectors
%   pipeline.sensing.status     — 'ok' | 'warning' | 'critical'
%
% FIGURES:
%   Figure 1 — Singular value spectrum + condition number
%   Figure 2 — Angular coverage map (sensitivity directions on sphere)
%   Figure 3 — Per-component sensitivity across all channels
%
% ============================================================

% ============================================================
%   CHECK MODULES 1 AND 2 HAVE BEEN RUN
% ============================================================
if ~exist('pipeline','var') || ~isfield(pipeline,'geom')
    error('Modules 1 and 2 must be run first.\nRun module1.m then module2.m before module3.m');
end

fprintf('\n============================================================\n');
fprintf('MODULE 3 — SENSING MATRIX\n');
fprintf('XMCD geometry: sensitivity = incident beam direction\n');
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 1 — USER CONTROLS
% ============================================================
% ============================================================

% Condition number threshold — above this the LS seed will be
% unreliable due to noise amplification
cond_warning  = 50;     % warn if cond(Pmat) > this
cond_critical = 200;    % critical if cond(Pmat) > this

% Minimum acceptable singular value
% Below this a spin component is essentially unmeasured
sv_min_thresh = 0.1;    % relative to largest singular value

% Candidate phi angles to test when suggesting fixes
% These are the angles we test adding to improve coverage
phi_candidates_deg = 0:15:345;

% ---- Figure toggles ----
% exist-guard so a caller can set these false before running.
if ~exist('show_sv_spectrum',    'var'), show_sv_spectrum     = true; end
if ~exist('show_sphere_coverage','var'), show_sphere_coverage = true; end
if ~exist('show_per_component',  'var'), show_per_component   = true; end

% ============================================================
% ============================================================
%   SECTION 2 — BUILD SENSING MATRIX
% ============================================================
% ============================================================

N_angles   = pipeline.meas.N_angles;
angle_list = pipeline.geom.angle_list;

% Pmat: each row is the sensitivity vector for one channel
% In XMCD: p_r = k_i_hat (incident beam direction unit vector)
Pmat   = zeros(N_angles, 3);
p_vecs = zeros(N_angles, 3);

for r = 1:N_angles
    k_i   = angle_list(r).k_i;           % [kx ky kz]
    k_hat = k_i / norm(k_i);             % unit vector
    Pmat(r,:)   = k_hat;
    p_vecs(r,:) = k_hat;
end

fprintf('Sensing matrix built: [%d channels x 3 spin components]\n', N_angles);
fprintf('Sensitivity model: XMCD — p_r = incident beam direction k_i_hat\n\n');

% ============================================================
% ============================================================
%   SECTION 3 — RANK AND CONDITION NUMBER DIAGNOSTICS
% ============================================================
% ============================================================

% SVD of Pmat
[U_sv, S_sv, V_sv] = svd(Pmat, 'econ');
sv = diag(S_sv);   % singular values, descending

pmat_rank = rank(Pmat);
cond_num  = sv(1) / max(sv(end), 1e-15);
cond_PtP  = cond(Pmat' * Pmat);

% Relative singular values
sv_rel = sv / sv(1);

% Pseudoinverse
if cond_num > cond_critical
    % Truncated SVD pseudoinverse — more stable when ill-conditioned
    tol = sv(1) * sv_min_thresh;
    sv_inv = zeros(size(sv));
    sv_inv(sv > tol) = 1 ./ sv(sv > tol);
    Pinv = V_sv * diag(sv_inv) * U_sv';
    fprintf('WARNING: Using truncated SVD pseudoinverse (cond=%.1f > %.0f)\n\n', ...
            cond_num, cond_critical);
else
    Pinv = pinv(Pmat);
end

% Determine status
if pmat_rank < 3
    status = 'critical';
elseif cond_num > cond_critical
    status = 'critical';
elseif cond_num > cond_warning || sv_rel(end) < sv_min_thresh
    status = 'warning';
else
    status = 'ok';
end

% ============================================================
%   PRINT DIAGNOSTICS
% ============================================================

fprintf('============================================================\n');
fprintf('SENSING MATRIX DIAGNOSTICS\n');
fprintf('============================================================\n');
fprintf('Rank of Pmat          : %d / 3\n', pmat_rank);
fprintf('Condition number      : %.2f\n', cond_num);
fprintf('cond(Pmat^T Pmat)     : %.2f\n', cond_PtP);
fprintf('Singular values       : ');
fprintf('%.4f  ', sv); fprintf('\n');
fprintf('Relative sv           : ');
fprintf('%.4f  ', sv_rel); fprintf('\n');
fprintf('\nSpin component visibility (from V matrix of SVD):\n');

% The right singular vectors tell us which spin directions are well-measured
component_labels = {'Mx (x-spin)', 'My (y-spin)', 'Mz (z-spin)'};
for c = 1:3
    % How much each singular value contributes to this component
    contrib = sum(abs(V_sv(c,:)) .* sv(:)') / sum(sv);
    fprintf('  %-14s : visibility = %.3f', component_labels{c}, contrib);
    if contrib < 0.15
        fprintf('  <<< POORLY MEASURED');
    elseif contrib < 0.25
        fprintf('  << marginal');
    end
    fprintf('\n');
end

fprintf('\nStatus: ');
switch status
    case 'ok'
        fprintf('OK — sensing geometry is well-conditioned\n');
    case 'warning'
        fprintf('WARNING — sensing geometry is marginal\n');
        fprintf('  Consider adding more phi angles (see suggestions below)\n');
    case 'critical'
        fprintf('CRITICAL — reconstruction will fail or be unreliable\n');
        fprintf('  Additional phi angles are REQUIRED (see suggestions below)\n');
end
fprintf('============================================================\n\n');

% ============================================================
% ============================================================
%   SECTION 4 — SUGGEST FIX IF NEEDED
% ============================================================
% ============================================================

if strcmp(status, 'warning') || strcmp(status, 'critical')

    fprintf('============================================================\n');
    fprintf('PHI ANGLE SUGGESTIONS\n');
    fprintf('Testing candidate phi angles to improve conditioning...\n');
    fprintf('============================================================\n');

    current_phi = pipeline.geom.phi_deg;
    current_theta = pipeline.geom.theta_deg;

    best_improvement = 0;
    best_phi_single  = [];
    best_cond_single = cond_num;

    % Test each candidate phi angle added to the existing set
    improvement_table = zeros(numel(phi_candidates_deg), 3);  % phi, new_cond, new_rank

    for ci = 1:numel(phi_candidates_deg)
        phi_test = phi_candidates_deg(ci);

        % Skip if already in current set
        if any(abs(current_phi - phi_test) < 1)
            improvement_table(ci,:) = [phi_test, cond_num, pmat_rank];
            continue;
        end

        % Build augmented Pmat with this phi added at all current thetas
        Pmat_aug = Pmat;
        for it = 1:numel(current_theta)
            theta_r = deg2rad(current_theta(it));
            phi_r   = deg2rad(phi_test);
            p_new   = [-sin(theta_r)*cos(phi_r), ...
                       -sin(theta_r)*sin(phi_r), ...
                       -cos(theta_r)];
            p_new   = p_new / norm(p_new);
            Pmat_aug = [Pmat_aug; p_new];
        end

        sv_aug   = svd(Pmat_aug);
        cond_aug = sv_aug(1) / max(sv_aug(end), 1e-15);
        rank_aug = rank(Pmat_aug);

        improvement_table(ci,:) = [phi_test, cond_aug, rank_aug];

        if cond_aug < best_cond_single
            best_cond_single = cond_aug;
            best_phi_single  = phi_test;
            best_improvement = cond_num - cond_aug;
        end
    end

    % Sort by condition number improvement
    [sorted_cond, sort_idx] = sort(improvement_table(:,2));
    sorted_phi  = improvement_table(sort_idx, 1);
    sorted_rank = improvement_table(sort_idx, 3);

    fprintf('\nTop 5 single phi angles to add (sorted by improvement):\n');
    fprintf('  %-12s  %-18s  %-8s\n', 'phi (deg)', 'new cond number', 'new rank');
    fprintf('  %s\n', repmat('-',1,42));
    shown = 0;
    for si = 1:numel(sorted_phi)
        phi_s = sorted_phi(si);
        if any(abs(current_phi - phi_s) < 1), continue; end
        fprintf('  %-12.1f  %-18.3f  %-8d\n', phi_s, sorted_cond(si), sorted_rank(si));
        shown = shown + 1;
        if shown >= 5, break; end
    end

    % Also test pairs of phi angles for critical case
    if strcmp(status,'critical') && pmat_rank < 3
        fprintf('\nRank is %d — testing pairs of phi angles...\n', pmat_rank);
        best_cond_pair = cond_num;
        best_pair = [];

        test_phis = phi_candidates_deg(~ismember(phi_candidates_deg, current_phi));
        for ci = 1:min(numel(test_phis),12)
            for cj = ci+1:min(numel(test_phis),12)
                phi1 = test_phis(ci);
                phi2 = test_phis(cj);
                Pmat_aug = Pmat;
                for it = 1:numel(current_theta)
                    theta_r = deg2rad(current_theta(it));
                    for phi_t = [phi1 phi2]
                        phi_r = deg2rad(phi_t);
                        p_new = [-sin(theta_r)*cos(phi_r), ...
                                 -sin(theta_r)*sin(phi_r), ...
                                 -cos(theta_r)];
                        p_new = p_new / norm(p_new);
                        Pmat_aug = [Pmat_aug; p_new];
                    end
                end
                sv_aug = svd(Pmat_aug);
                cond_aug = sv_aug(1) / max(sv_aug(end), 1e-15);
                if cond_aug < best_cond_pair
                    best_cond_pair = cond_aug;
                    best_pair = [phi1 phi2];
                end
            end
        end

        if ~isempty(best_pair)
            fprintf('\nBest pair of phi angles to add: [%.0f deg, %.0f deg]\n', ...
                    best_pair(1), best_pair(2));
            fprintf('  New condition number: %.3f\n', best_cond_pair);
        end
    end

    fprintf('\nTo add a phi angle: add it to phi_angles_deg in module2.m\n');
    fprintf('then rerun module2.m and module3.m.\n');
    fprintf('============================================================\n\n');

else
    fprintf('Sensing geometry is well-conditioned — no changes needed.\n\n');
end

% ============================================================
% ============================================================
%   SECTION 5 — PACK PIPELINE STRUCT
% ============================================================
% ============================================================

pipeline.sensing.Pmat    = Pmat;
pipeline.sensing.Pinv    = Pinv;
pipeline.sensing.rank    = pmat_rank;
pipeline.sensing.cond    = cond_num;
pipeline.sensing.cond_PtP = cond_PtP;
pipeline.sensing.sv      = sv;
pipeline.sensing.sv_rel  = sv_rel;
pipeline.sensing.p_vecs  = p_vecs;
pipeline.sensing.status  = status;
pipeline.sensing.U_sv    = U_sv;
pipeline.sensing.V_sv    = V_sv;

% ============================================================
% ============================================================
%   SECTION 6 — FIGURES
% ============================================================
% ============================================================

% ---- Figure 1: Singular value spectrum ----
if show_sv_spectrum
figure('Units','normalized','OuterPosition',[0.02 0.30 0.43 0.50],'Color','w');

subplot(1,2,1);
bar(sv, 'FaceColor',[0.2 0.4 0.8], 'EdgeColor','none');
hold on;
yline(sv(1)*sv_min_thresh, 'r--', 'LineWidth', 1.5);
text(0.6, sv(1)*sv_min_thresh*1.1, sprintf('min threshold (%.0f%% of max)', sv_min_thresh*100), ...
     'Color','r','FontSize',8);
xlabel('Singular value index','FontSize',10);
ylabel('Magnitude','FontSize',10);
title('Singular Values of Pmat','FontSize',11);
grid on; hold off;

subplot(1,2,2);
c_color = [0.2 0.7 0.2];
if cond_num > cond_warning,  c_color = [0.9 0.6 0.0]; end
if cond_num > cond_critical, c_color = [0.85 0.1 0.1]; end

bar([cond_num cond_warning cond_critical], ...
    'FaceColor','flat', ...
    'CData', [c_color; 0.9 0.6 0.0; 0.85 0.1 0.1], ...
    'EdgeColor','none');
set(gca,'XTickLabel',{'Current','Warning\nthreshold','Critical\nthreshold'});
ylabel('Condition number','FontSize',10);
title(sprintf('Condition Number: %.1f  [%s]', cond_num, upper(status)),'FontSize',11);
grid on;

sgtitle(sprintf('Sensing Matrix Health  |  Rank = %d/3  |  %d channels', ...
        pmat_rank, N_angles), 'FontSize', 12);
end % show_sv_spectrum

% ---- Figure 2: Sensitivity directions on unit sphere ----
if show_sphere_coverage
figure('Units','normalized','OuterPosition',[0.52 0.30 0.43 0.50],'Color','w');
hold on;

[sph_x, sph_y, sph_z] = sphere(30);
surf(sph_x, sph_y, sph_z, 'FaceAlpha', 0.06, 'EdgeColor', [0.8 0.8 0.8], ...
     'EdgeAlpha', 0.3, 'FaceColor', [0.85 0.90 1.0]);

colors_phi = hsv(N_phi);
phi_list_unique = pipeline.geom.phi_deg;

for r = 1:N_angles
    p = p_vecs(r,:);
    % Find which phi this belongs to
    phi_r = angle_list(r).phi_deg;
    phi_idx = find(abs(phi_list_unique - phi_r) < 0.5, 1);
    if isempty(phi_idx), phi_idx = 1; end
    scatter3(p(1), p(2), p(3), 60, colors_phi(phi_idx,:), 'filled', ...
             'MarkerEdgeColor', 'none');
end

% Draw coordinate axes
ax_l = 1.3;
quiver3(0,0,0,ax_l,0,0,0,'r','LineWidth',1.5,'MaxHeadSize',0.3);
quiver3(0,0,0,0,ax_l,0,0,'g','LineWidth',1.5,'MaxHeadSize',0.3);
quiver3(0,0,0,0,0,ax_l,0,'b','LineWidth',1.5,'MaxHeadSize',0.3);
text(ax_l,0,0,'x','FontSize',11,'FontWeight','bold','Color','r');
text(0,ax_l,0,'y','FontSize',11,'FontWeight','bold','Color',[0 0.6 0]);
text(0,0,ax_l,'z (depth)','FontSize',11,'FontWeight','bold','Color','b');

% Legend for phi values
legend_h = zeros(N_phi,1);
for ip = 1:N_phi
    legend_h(ip) = scatter3(NaN,NaN,NaN, 60, colors_phi(ip,:), 'filled');
end
legend(legend_h, arrayfun(@(p) sprintf('\\phi = %.0f°', p), ...
       phi_list_unique, 'UniformOutput', false), ...
       'Location','eastoutside','FontSize',8);

xlabel('k_x','FontSize',10); ylabel('k_y','FontSize',10); zlabel('k_z','FontSize',10);
title({'Sensitivity Directions on Unit Sphere', ...
       'Each point = one (theta,phi) channel  |  Colour = phi angle'}, 'FontSize',10);
grid on; axis equal; view(35,25);
rotate3d on;
hold off;
end % show_sphere_coverage

% ---- Figure 3: Per-component sensitivity across channels ----
if show_per_component
figure('Units','normalized','OuterPosition',[0.15 0.02 0.65 0.25],'Color','w');

theta_per_channel = arrayfun(@(s) s.theta_deg, angle_list);
phi_per_channel   = arrayfun(@(s) s.phi_deg,   angle_list);

subplot(1,3,1);
scatter(1:N_angles, abs(p_vecs(:,1)), 30, theta_per_channel, 'filled');
colormap(gca, jet); colorbar;
xlabel('Channel index','FontSize',9); ylabel('|p_x|  sensitivity to Mx','FontSize',9);
title('Mx sensitivity per channel','FontSize',10); grid on; ylim([0 1]);

subplot(1,3,2);
scatter(1:N_angles, abs(p_vecs(:,2)), 30, theta_per_channel, 'filled');
colormap(gca, jet); colorbar;
xlabel('Channel index','FontSize',9); ylabel('|p_y|  sensitivity to My','FontSize',9);
title('My sensitivity per channel','FontSize',10); grid on; ylim([0 1]);

subplot(1,3,3);
scatter(1:N_angles, abs(p_vecs(:,3)), 30, theta_per_channel, 'filled');
colormap(gca, jet); colorbar;
xlabel('Channel index','FontSize',9); ylabel('|p_z|  sensitivity to Mz','FontSize',9);
title('Mz sensitivity per channel','FontSize',10); grid on; ylim([0 1]);

sgtitle('Per-Component Spin Sensitivity  |  Colour = theta angle  |  Value = |p_component|', ...
        'FontSize',10);
end % show_per_component

% ============================================================

fprintf('--- Module 3 complete ---\n');
fprintf('pipeline.sensing populated.\n');
fprintf('  Pmat  : [%d x 3]\n', N_angles);
fprintf('  Rank  : %d\n', pmat_rank);
fprintf('  Cond  : %.2f\n', cond_num);
fprintf('  Status: %s\n', upper(status));
fprintf('Pass pipeline to algorithm slot (algo_raar.m / algo_cnn.m).\n\n');

% ============================================================
% ============================================================
%   BEAMLINE REFERENCE — ALS COSMIC 7.0.1.1
% ============================================================
%
% Facility   : Advanced Light Source (ALS)
%              Lawrence Berkeley National Laboratory
% Beamline   : 7.0.1.1  COSMIC Scattering
%
% XMCD sensitivity model:
%   In XMCD the absorption difference (I+ - I-) is proportional
%   to the projection of the magnetic moment onto the photon
%   propagation direction (incident beam k_i_hat).
%   Reference: van der Laan & Figueroa, Coord. Chem. Rev. 2014
%              "X-ray magnetic circular dichroism — a versatile
%               tool to study magnetism"
%
%   For reflection geometry at ALS COSMIC:
%   The incident beam direction rotates with (theta, phi),
%   sampling different projections of the spin vector.
%   Full 3D reconstruction requires rank(Pmat) = 3, meaning
%   the beam must probe all three spatial directions.
%
%   With only vertical tilt (theta sweep at fixed phi), the
%   beam stays in one plane and rank(Pmat) = 2 — My or Mx
%   (depending on phi) is invisible. This is why multiple
%   phi angles are essential.
%
% All parameters are USER CONTROLLED above.
% ============================================================