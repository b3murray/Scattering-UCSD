% ============================================================
% ============================================================
%   ALGORITHM SLOT — TRUE FIELD SEED (TESTING ONLY)
%   Seeds RESIRE-V with the true field + small noise
%   PURPOSE: Verify RESIRE-V converges correctly before
%            the full RAAR/CNN pipeline is fixed.
%   DO NOT USE FOR REAL RECONSTRUCTION.
% ============================================================
%
% This module bypasses phase retrieval entirely and seeds
% RESIRE-V with the ground truth magnetization plus
% controllable Gaussian noise. If RESIRE-V works correctly
% it should converge to near-zero error from this seed.
%
% If RESIRE-V fails even from a near-perfect seed, the
% problem is in the RESIRE-V gradient/step logic, not RAAR.
% If RESIRE-V succeeds, the problem is upstream in the
% forward model or RAAR, and we can fix those next.
%
% INPUTS:
%   pipeline.M.x/y/z    — true magnetization [Ny x Nx x Nz]
%
% OUTPUTS:
%   pipeline.seed.Mx/My/Mz  — noisy true field
%   pipeline.seed.method    — 'true_field_test'
%
% ============================================================

if ~exist('pipeline','var') || ~isfield(pipeline,'M')
    error('Module 1 must be run first.');
end

fprintf('\n============================================================\n');
fprintf('ALGORITHM SLOT — TRUE FIELD SEED (RESIRE-V TEST MODE)\n');
fprintf('WARNING: Using ground truth as seed — testing only\n');
fprintf('============================================================\n\n');

% ============================================================
%   CONTROLS
% ============================================================

% Noise level added to true field
% 0.0  = perfect seed — use this first to verify RESIRE-V works
% 0.05 = 5% noise (realistic best-case seed)
% 0.20 = 20% noise (stress test)
noise_level = 0.0;

% Random seed for reproducibility
rng(42);

% ============================================================
%   BUILD NOISY TRUE FIELD SEED
% ============================================================

Mx_true = pipeline.M.x;
My_true = pipeline.M.y;
Mz_true = pipeline.M.z;

[Ny_vol, Nx_vol, Nz_vol] = size(Mx_true);

% Add Gaussian noise scaled to noise_level
noise_x = noise_level * randn(size(Mx_true));
noise_y = noise_level * randn(size(My_true));
noise_z = noise_level * randn(size(Mz_true));

Mx_seed = Mx_true + noise_x;
My_seed = My_true + noise_y;
Mz_seed = Mz_true + noise_z;

% Renormalize to unit vectors (spins must stay on unit sphere)
mag_seed = sqrt(Mx_seed.^2 + My_seed.^2 + Mz_seed.^2);
mag_seed(mag_seed < 1e-12) = 1;
Mx_seed = Mx_seed ./ mag_seed;
My_seed = My_seed ./ mag_seed;
Mz_seed = Mz_seed ./ mag_seed;

% ============================================================
%   MEASURE SEED QUALITY
% ============================================================

epsv = 1e-12;
rel_L2 = sqrt(norm(Mx_seed(:)-Mx_true(:))^2 + ...
              norm(My_seed(:)-My_true(:))^2 + ...
              norm(Mz_seed(:)-Mz_true(:))^2) / ...
         (sqrt(norm(Mx_true(:))^2 + norm(My_true(:))^2 + norm(Mz_true(:))^2) + epsv);

dotp   = Mx_true.*Mx_seed + My_true.*My_seed + Mz_true.*Mz_seed;
normT  = sqrt(Mx_true.^2 + My_true.^2 + Mz_true.^2) + epsv;
normS  = sqrt(Mx_seed.^2 + My_seed.^2 + Mz_seed.^2) + epsv;
cosang = max(min(dotp./(normT.*normS), 1), -1);
ang_err = acosd(cosang);

fprintf('Seed quality (before RESIRE-V):\n');
fprintf('  Noise level        : %.0f%%\n', noise_level*100);
fprintf('  Relative L2 error  : %.2f%%\n', 100*rel_L2);
fprintf('  Mean angular error : %.2f deg\n', mean(ang_err(:), 'omitnan'));
fprintf('  Median ang. error  : %.2f deg\n', median(ang_err(:), 'omitnan'));
fprintf('\nExpected RESIRE-V outcome:\n');
if noise_level < 0.1
    fprintf('  Should converge to <5%% error if RESIRE-V is correct.\n');
elseif noise_level < 0.3
    fprintf('  Should converge to <20%% error if RESIRE-V is correct.\n');
else
    fprintf('  Challenging seed — convergence will be slower.\n');
end
fprintf('\n');

% ============================================================
%   PACK PIPELINE STRUCT
% ============================================================

pipeline.seed.Mx           = Mx_seed;
pipeline.seed.My           = My_seed;
pipeline.seed.Mz           = Mz_seed;
pipeline.seed.method       = 'true_field_test';
pipeline.seed.noise_level  = noise_level;
pipeline.seed.rel_L2       = rel_L2;
pipeline.seed.mean_ang_err = mean(ang_err(:), 'omitnan');
pipeline.seed.raar_err     = [];   % not applicable
pipeline.seed.ls_residual  = [];   % not applicable

fprintf('--- module4_truefield complete ---\n');
fprintf('Seed: true field + %.0f%% noise  [%d x %d x %d]\n', ...
        noise_level*100, Ny_vol, Nx_vol, Nz_vol);
fprintf('Pass pipeline to module6.m (RESIRE-V).\n\n');