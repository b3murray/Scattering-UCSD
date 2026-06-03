% ============================================================
% ============================================================
%   MASTER PIPELINE — COHERENT MAGNETIC SCATTERING
%   3D Spin Vector Reconstruction from XMCD Diffraction
% ============================================================
%
% USAGE:
%   Run this script to execute the full pipeline end to end.
%   Or run individual modules manually in sequence.
%
% MODULE ORDER:
%   1. module1.m   — generate true 3D spin field
%   2. module2.m   — simulate XMCD diffraction patterns
%   3. module3.m   — sensing matrix + rank diagnostics
%   4. module4.m   — algorithm slot (RAAR default)
%   5. module5.m   — CNN seed (optional, after core works)
%   6. module6.m   — RESIRE-V refinement
%   7. module8.m   — error analysis and figures
%
% ALGORITHM SLOT (module4) — set algo_mode below:
%   'raar'      -> module4.m   — RAAR phase retrieval + LS seed
%   'cnn'       -> module5_cnn.m — CNN direct prediction
%   'raar_cnn'  -> module4 then module5_cnn
%   'ls_direct' -> module4_ls.m — direct LS, no phase retrieval
%   'random'    -> module4_rand.m — random baseline
%
% ============================================================

% ============================================================
%   ALGORITHM SELECTION — SET THIS BEFORE RUNNING
% ============================================================
%   'raar'      -> RAAR phase retrieval + least squares seed
%   'cnn'       -> CNN direct prediction (requires trained network)
%   'raar_cnn'  -> RAAR seed refined by CNN
%   'ls_direct' -> direct LS from amplitudes, no phase retrieval
%   'random'    -> random initialization (baseline test)
% ============================================================
algo_mode = 'raar';  % SET THIS — use 'raar' for real reconstruction

% ============================================================
%   PATH SETUP
%   Add the folder containing all module files to MATLAB path.
%   This file (Master.m) must be in the same folder as the modules.
%   If not, change pipeline_dir to the correct path.
% ============================================================
pipeline_dir = fileparts(mfilename('fullpath'));
if isempty(pipeline_dir)
    pipeline_dir = pwd;   % fallback: use current directory
end
addpath(pipeline_dir);

clc; close all;

% ============================================================
%   RUN MODULES IN ORDER
% ============================================================

fprintf('============================================================\n');
fprintf('MASTER PIPELINE — COHERENT MAGNETIC SCATTERING\n');
fprintf('Algorithm mode: %s\n', upper(algo_mode));
fprintf('============================================================\n\n');

t_total = tic;

% --- Module 1: Vector field ---
fprintf('[1/6] Running Module 1 — Vector Field Generator...\n');
run('module1.m');
fprintf('[1/6] Module 1 done.\n\n');

% --- Module 2: Forward model ---
fprintf('[2/6] Running Module 2 — Forward Scattering Model...\n');
run('module2.m');
fprintf('[2/6] Module 2 done.\n\n');

% --- Module 3: Sensing matrix ---
fprintf('[3/6] Running Module 3 — Sensing Matrix...\n');
run('module3.m');
fprintf('[3/6] Module 3 done.\n\n');

% --- Algorithm slot ---
fprintf('[4/6] Running Algorithm Slot — %s...\n', upper(algo_mode));
switch lower(algo_mode)
    case 'raar'
        run('module4.m');
    case 'truefield'
        run('module4_truefield.m');  % TEST ONLY — seeds with true field + noise
    case 'cnn'
        run('module5.m');
    case 'raar_cnn'
        run('module4.m');
        run('module5_cnn.m');
    case 'ls_direct'
        run('module4_ls.m');
    case 'random'
        run('module4_rand.m');
    otherwise
        error('Unknown algo_mode: %s', algo_mode);
end
fprintf('[4/6] Algorithm slot done.\n\n');

% --- Module 6: RESIRE-V ---
fprintf('[5/6] Running Module 6 — RESIRE-V Refinement...\n');
run('module6.m');
fprintf('[5/6] Module 6 done.\n\n');

% --- Module 8: Error analysis ---
fprintf('[6/6] Running Module 8 — Error Analysis...\n');
run('module8.m');
fprintf('[6/6] Module 8 done.\n\n');

fprintf('============================================================\n');
fprintf('Pipeline complete.  Total time: %.1f seconds\n', toc(t_total));
fprintf('============================================================\n');