% ============================================================
% ============================================================
%   MODULE 8 — ERROR ANALYSIS AND FINAL FIGURES
%   Compares reconstruction to true field across all z-layers
% ============================================================
%
% PURPOSE:
%   Comprehensive error analysis comparing pipeline.recon
%   against pipeline.M (true field from module1).
%   Produces publication-quality summary figures.
%
% INPUTS:
%   pipeline.recon.Mx/My/Mz   — reconstructed field
%   pipeline.M.x/y/z          — true field
%   pipeline.recon.err_hist   — amplitude residual history
%   pipeline.recon.ang_hist   — angular error history
%   pipeline.recon.l2_hist    — L2 error history
%   pipeline.seed.method      — which algorithm was used
%
% OUTPUTS (printed to console):
%   Per-layer and global error metrics
%
% FIGURES:
%   Figure 1 — Convergence summary
%   Figure 2 — True vs reconstructed 3D quiver (all layers)
%   Figure 3 — Angular error maps per z-layer
%   Figure 4 — Component-wise percent difference
%   Figure 5 — Summary table printed as figure
%
% ============================================================

if ~exist('pipeline','var') || ~isfield(pipeline,'recon')
    error('Modules 1-6 must be run first.');
end

fprintf('\n============================================================\n');
fprintf('MODULE 8 — ERROR ANALYSIS\n');
fprintf('============================================================\n\n');

% ============================================================
%   FIGURE TOGGLES
% ============================================================
show_convergence   = true;   % Fig 1: convergence curves
show_3d_comparison = true;   % Fig 2: 3D quiver true vs recon
show_angular_maps  = true;   % Fig 3: angular error maps
show_component_err = true;   % Fig 4: component percent difference
show_summary_table = true;   % Fig 5: printed summary as figure

% ============================================================
%   RETRIEVE DATA
% ============================================================

Mx_rec  = pipeline.recon.Mx;
My_rec  = pipeline.recon.My;
Mz_rec  = pipeline.recon.Mz;

Mx_true = pipeline.M.x;
My_true = pipeline.M.y;
Mz_true = pipeline.M.z;

z_vec   = pipeline.M.z_vec;
x_vec   = pipeline.M.x_vec;
y_vec   = pipeline.M.y_vec;

[Ny_vol, Nx_vol, Nz_vol] = size(Mx_true);
epsv = 1e-12;

err_hist = pipeline.recon.err_hist;
ang_hist = pipeline.recon.ang_hist;
l2_hist  = pipeline.recon.l2_hist;
seed_method = pipeline.recon.seed_method;
recon_method = pipeline.recon.method;

% ============================================================
%   COMPUTE ERRORS PER LAYER AND GLOBALLY
% ============================================================

fprintf('Per-layer error analysis:\n');
fprintf('%-10s  %-14s  %-14s  %-14s\n', 'Layer', 'L2 error (%)', 'Mean angle', 'Median angle');
fprintf('%s\n', repmat('-',1,56));

layer_l2   = zeros(1, Nz_vol);
layer_ang_mean = zeros(1, Nz_vol);
layer_ang_med  = zeros(1, Nz_vol);
ang_maps   = zeros(Ny_vol, Nx_vol, Nz_vol);

for kz = 1:Nz_vol
    mxt = Mx_true(:,:,kz); myt = My_true(:,:,kz); mzt = Mz_true(:,:,kz);
    mxr = Mx_rec(:,:,kz);  myr = My_rec(:,:,kz);  mzr = Mz_rec(:,:,kz);

    % L2
    l2 = sqrt(norm(mxr(:)-mxt(:))^2 + norm(myr(:)-myt(:))^2 + norm(mzr(:)-mzt(:))^2) / ...
         (sqrt(norm(mxt(:))^2 + norm(myt(:))^2 + norm(mzt(:))^2) + epsv);
    layer_l2(kz) = 100 * l2;

    % Angular error
    dotp  = mxt.*mxr + myt.*myr + mzt.*mzr;
    normT = sqrt(mxt.^2+myt.^2+mzt.^2) + epsv;
    normR = sqrt(mxr.^2+myr.^2+mzr.^2) + epsv;
    ang_k = acosd(max(min(dotp./(normT.*normR), 1), -1));
    ang_maps(:,:,kz)    = ang_k;
    layer_ang_mean(kz)  = mean(ang_k(:), 'omitnan');
    layer_ang_med(kz)   = median(ang_k(:), 'omitnan');

    fprintf('z-layer %d   z=%.2f   %-14.2f %-14.2f %-14.2f\n', ...
            kz, z_vec(kz), layer_l2(kz), layer_ang_mean(kz), layer_ang_med(kz));
end

% Global errors
rel_L2_global = sqrt(norm(Mx_rec(:)-Mx_true(:))^2 + ...
                     norm(My_rec(:)-My_true(:))^2 + ...
                     norm(Mz_rec(:)-Mz_true(:))^2) / ...
                (sqrt(norm(Mx_true(:))^2+norm(My_true(:))^2+norm(Mz_true(:))^2)+epsv);
all_ang = ang_maps(:);
mean_ang_global   = mean(all_ang, 'omitnan');
median_ang_global = median(all_ang, 'omitnan');

% Per-component errors
% Percent error is only meaningful where the true component is significantly
% nonzero (e.g. Mx/My are near zero everywhere except domain walls when
% domains are z-polarized). Use a threshold of 0.1 (10% of unit norm)
% and report RMSE alongside masked percent error.
comp_thresh = 0.1;

mask_x = abs(Mx_true) > comp_thresh;
mask_y = abs(My_true) > comp_thresh;
mask_z = abs(Mz_true) > comp_thresh;

% RMSE per component — always valid, no division by true value
rmse_x = sqrt(mean((Mx_rec(:)-Mx_true(:)).^2));
rmse_y = sqrt(mean((My_rec(:)-My_true(:)).^2));
rmse_z = sqrt(mean((Mz_rec(:)-Mz_true(:)).^2));

% Masked percent error — only where true component is large enough
pct_x_vals = 100*abs(Mx_rec(mask_x)-Mx_true(mask_x))./(abs(Mx_true(mask_x))+epsv);
pct_y_vals = 100*abs(My_rec(mask_y)-My_true(mask_y))./(abs(My_true(mask_y))+epsv);
pct_z_vals = 100*abs(Mz_rec(mask_z)-Mz_true(mask_z))./(abs(Mz_true(mask_z))+epsv);

pct_x_mean = mean(pct_x_vals(:));  if isempty(pct_x_vals), pct_x_mean = NaN; end
pct_y_mean = mean(pct_y_vals(:));  if isempty(pct_y_vals), pct_y_mean = NaN; end
pct_z_mean = mean(pct_z_vals(:));  if isempty(pct_z_vals), pct_z_mean = NaN; end

n_x = sum(mask_x(:)); n_y = sum(mask_y(:)); n_z = sum(mask_z(:));

fprintf('%s\n', repmat('-',1,56));
fprintf('GLOBAL:     %-14.2f %-14.2f %-14.2f\n', ...
        100*rel_L2_global, mean_ang_global, median_ang_global);

fprintf('\n============================================================\n');
fprintf('RECONSTRUCTION SUMMARY\n');
fprintf('============================================================\n');
fprintf('Algorithm seed     : %s\n', seed_method);
fprintf('Refinement method  : %s\n', recon_method);
fprintf('Volume size        : [%d x %d x %d]\n', Ny_vol, Nx_vol, Nz_vol);
fprintf('------------------------------------------------------------\n');
fprintf('Global L2 error    : %.2f %%\n', 100*rel_L2_global);
fprintf('Mean angular error : %.2f deg\n', mean_ang_global);
fprintf('Median ang. error  : %.2f deg\n', median_ang_global);
fprintf('------------------------------------------------------------\n');
fprintf('Per-component RMSE (always valid):\n');
fprintf('  Mx RMSE          : %.4f\n', rmse_x);
fprintf('  My RMSE          : %.4f\n', rmse_y);
fprintf('  Mz RMSE          : %.4f\n', rmse_z);
fprintf('Per-component percent error (masked: |M_true|>%.2f, %d/%d/%d px):\n', ...
        comp_thresh, n_x, n_y, n_z);
if isnan(pct_x_mean)
    fprintf('  Mx pct error     : N/A (no pixels above threshold)\n');
else
    fprintf('  Mx pct error     : %.2f %%\n', pct_x_mean);
end
if isnan(pct_y_mean)
    fprintf('  My pct error     : N/A (no pixels above threshold)\n');
else
    fprintf('  My pct error     : %.2f %%\n', pct_y_mean);
end
if isnan(pct_z_mean)
    fprintf('  Mz pct error     : N/A (no pixels above threshold)\n');
else
    fprintf('  Mz pct error     : %.2f %%\n', pct_z_mean);
end
fprintf('Final amp residual : %.4e\n', err_hist(end));
fprintf('============================================================\n\n');

% ============================================================
%   FIGURE 1 — CONVERGENCE
% ============================================================
if show_convergence && ~isempty(err_hist)
    figure('Units','normalized','OuterPosition',[0.02 0.52 0.43 0.42],'Color','w');

    subplot(2,1,1);
    semilogy(err_hist, 'b-', 'LineWidth', 2);
    xlabel('Iteration','FontSize',10);
    ylabel('Amplitude residual','FontSize',10);
    title(sprintf('Convergence — %s → %s', seed_method, recon_method),'FontSize',11);
    grid on;

    subplot(2,1,2);
    plot(ang_hist, 'r-', 'LineWidth', 2); hold on;
    plot(l2_hist,  'b-', 'LineWidth', 2);
    yline(90,'k--','LineWidth',1);
    legend('Mean angular error (deg)', 'L2 error (%)', ...
           'Random baseline','Location','best','FontSize',8);
    xlabel('Iteration','FontSize',10);
    ylabel('Error','FontSize',10);
    title('Error vs Iteration','FontSize',11);
    grid on; hold off;
end

% ============================================================
%   FIGURE 2 — TRUE VS RECONSTRUCTED (ALL Z-LAYERS)
% ============================================================
if show_3d_comparison
    step_q = max(1, round(Ny_vol/16));
    [Xg, Yg] = meshgrid(x_vec, y_vec);
    Xs = Xg(1:step_q:end,1:step_q:end);
    Ys = Yg(1:step_q:end,1:step_q:end);
    Zs = zeros(size(Xs));

    figure('Units','normalized','OuterPosition',[0.02 0.02 0.95 0.46],'Color','w');

    for kz = 1:Nz_vol
        mxt = Mx_true(:,:,kz); myt = My_true(:,:,kz); mzt = Mz_true(:,:,kz);
        mxr = Mx_rec(:,:,kz);  myr = My_rec(:,:,kz);  mzr = Mz_rec(:,:,kz);

        % True — top row
        subplot(2, Nz_vol, kz); hold on;
        quiver3(Xs,Ys,Zs, mxt(1:step_q:end,1:step_q:end), ...
                           myt(1:step_q:end,1:step_q:end), ...
                           mzt(1:step_q:end,1:step_q:end), 0.6,'b','LineWidth',1.2);
        grid on; axis equal; view(35,25);
        xlabel('X'); ylabel('Y'); zlabel('Mz');
        title(sprintf('TRUE  z=%d  (%.1f nm)', kz, z_vec(kz)),'FontSize',9);
        hold off;

        % Reconstructed — bottom row
        subplot(2, Nz_vol, kz+Nz_vol); hold on;
        quiver3(Xs,Ys,Zs, mxr(1:step_q:end,1:step_q:end), ...
                           myr(1:step_q:end,1:step_q:end), ...
                           mzr(1:step_q:end,1:step_q:end), 0.6,'r','LineWidth',1.2);
        grid on; axis equal; view(35,25);
        xlabel('X'); ylabel('Y'); zlabel('Mz');
        title(sprintf('RECON z=%d  ang=%.1f°', kz, layer_ang_mean(kz)),'FontSize',9);
        hold off;
    end

    sgtitle(sprintf('True (blue) vs Reconstructed (red)  |  %s  |  L2=%.1f%%  ang=%.1f°', ...
            seed_method, 100*rel_L2_global, mean_ang_global), 'FontSize',11);
end

% ============================================================
%   FIGURE 3 — ANGULAR ERROR MAPS
% ============================================================
if show_angular_maps
    figure('Units','normalized','OuterPosition',[0.52 0.52 0.45 0.42],'Color','w');

    for kz = 1:Nz_vol
        subplot(1, Nz_vol, kz);
        imagesc(x_vec, y_vec, ang_maps(:,:,kz));
        axis image; colorbar;
        colormap(gca,'hot'); clim([0 90]);
        title(sprintf('z=%d  mean=%.1f°  med=%.1f°', ...
              kz, layer_ang_mean(kz), layer_ang_med(kz)),'FontSize',9);
        xlabel('X (units)'); ylabel('Y (units)');
    end
    sgtitle('Angular Error (deg)  |  0°=perfect  90°=random  |  lower=better','FontSize',10);
end

% ============================================================
%   FIGURE 4 — COMPONENT RMSE MAPS (middle layer)
% ============================================================
if show_component_err
    kz_mid = round(Nz_vol/2);
    figure('Units','normalized','OuterPosition',[0.52 0.02 0.45 0.46],'Color','w');

    comps    = {'M_x','M_y','M_z'};
    err_maps = {abs(Mx_rec(:,:,kz_mid)-Mx_true(:,:,kz_mid)), ...
                abs(My_rec(:,:,kz_mid)-My_true(:,:,kz_mid)), ...
                abs(Mz_rec(:,:,kz_mid)-Mz_true(:,:,kz_mid))};

    for c = 1:3
        subplot(1,3,c);
        imagesc(x_vec, y_vec, err_maps{c});
        axis image; colorbar;
        colormap(gca,'turbo');
        title(sprintf('%s  rmse=%.4f', comps{c}, sqrt(mean(err_maps{c}(:).^2))),'FontSize',10);
        xlabel('X'); ylabel('Y');
    end
    sgtitle(sprintf('Component Absolute Error  |  z-layer %d/%d  |  seed: %s', ...
            kz_mid, Nz_vol, seed_method),'FontSize',10);
end

% ============================================================
%   FIGURE 5 — SUMMARY TABLE
% ============================================================
if show_summary_table
    figure('Units','normalized','OuterPosition',[0.30 0.42 0.40 0.18],'Color','w');
    axis off;

    col_hdr = {'Metric','Value'};
    rows = {
        'Seed method',        seed_method;
        'Recon method',       recon_method;
        'Volume size',        sprintf('%dx%dx%d', Ny_vol, Nx_vol, Nz_vol);
        'Global L2 error',    sprintf('%.2f %%', 100*rel_L2_global);
        'Mean angular error', sprintf('%.2f deg', mean_ang_global);
        'Median ang. error',  sprintf('%.2f deg', median_ang_global);
        'Mx RMSE',            sprintf('%.4f', rmse_x);
        'My RMSE',            sprintf('%.4f', rmse_y);
        'Mz RMSE',            sprintf('%.4f', rmse_z);
        'Mx pct error (masked)', sprintf('%.2f %%  (%d px)', pct_x_mean, n_x);
        'My pct error (masked)', sprintf('%.2f %%  (%d px)', pct_y_mean, n_y);
        'Mz pct error (masked)', sprintf('%.2f %%  (%d px)', pct_z_mean, n_z);
        'Final amp residual', sprintf('%.3e', err_hist(end));
    };

    t = uitable('Data', rows, 'ColumnName', col_hdr, ...
                'Units','normalized','Position',[0 0 1 1], ...
                'ColumnWidth',{220,180}, ...
                'FontSize', 10);
end

% ============================================================

fprintf('--- Module 8 complete ---\n');
fprintf('All error metrics computed and figures generated.\n\n');