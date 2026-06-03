% ============================================================
% ============================================================
%   MODULE 5 — CNN SEED
%   Convolutional Neural Network Direct Prediction
% ============================================================
%
% PURPOSE:
%   Uses a trained CNN to predict [Mx, My, Mz] directly from
%   the stack of XMCD diffraction intensity patterns, bypassing
%   RAAR phase retrieval entirely.
%
%   The CNN takes the N_angles intensity patterns as input
%   channels and outputs a 3-channel [Mx, My, Mz] prediction.
%   This prediction is then used as the seed for RESIRE-V.
%
% STATUS: NOT YET IMPLEMENTED
%   This module is a stub. It will be implemented after:
%     1. Module 2 forward model is fixed (full FFT, not spatial mean)
%     2. RESIRE-V (module6) is confirmed working with truefield seed
%     3. Training data can be generated using the corrected forward model
%
% PLANNED ARCHITECTURE:
%   - U-Net encoder-decoder (same as Code 1/5 drafts)
%   - Input:  [Ny x Nx x N_angles] — stacked intensity patterns
%   - Output: [Ny x Nx x 3]        — predicted Mx, My, Mz
%   - Loss:   cosine loss (minimizes angular error directly)
%   - Train:  5000+ samples from module1+module2 synthetic data
%   - Target: validation cosine loss < 0.1 before use as seed
%
% WHEN IMPLEMENTED THIS MODULE WILL:
%   1. Load trained network from file
%   2. Format pipeline.meas.b_minus as CNN input tensor
%   3. Run forward pass to get M prediction
%   4. Enforce unit norm on predicted spins
%   5. Populate pipeline.seed exactly like module4 does
%
% INPUTS (when implemented):
%   pipeline.meas.b_minus       — magnetic signal [Npy x Npx x Nangles]
%   pipeline.meas.Amp_minus     — amplitudes
%   cnn_model_path              — path to saved .mat network file
%
% OUTPUTS (when implemented):
%   pipeline.seed.Mx/My/Mz     — CNN predicted field
%   pipeline.seed.method       — 'cnn'
%
% ============================================================

if ~exist('pipeline','var') || ~isfield(pipeline,'meas')
    error('Modules 1-3 must be run first.');
end

fprintf('\n============================================================\n');
fprintf('MODULE 5 — CNN SEED\n');
fprintf('============================================================\n\n');

fprintf('STATUS: Not yet implemented.\n\n');
fprintf('Module 5 will be written after:\n');
fprintf('  1. Module 2 forward model is corrected (full 2D FFT)\n');
fprintf('  2. Module 6 RESIRE-V is confirmed working\n');
fprintf('  3. CNN training data can be generated correctly\n\n');
fprintf('To test the pipeline now, use algo_mode = ''truefield''\n');
fprintf('in master_pipeline.m\n\n');

error('Module 5 (CNN) not yet implemented. See notes above.');