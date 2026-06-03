# Physics-Informed Machine Learning for Magnetic Domain Reconstruction from XMCD Scattering

**ECE 228 — Machine Learning for Physical Applications | UC San Diego**  
**Author:** Bill Murray

---

## Overview

This repository contains all code and results for a project investigating machine learning approaches to reconstructing 3D magnetic spin vector fields from simulated X-ray Magnetic Circular Dichroism (XMCD) coherent scattering patterns.

The core problem is a **phase retrieval inverse problem**: given only the amplitude of XMCD diffraction patterns measured at multiple angles, recover the full 3D magnetization field M(x,y,z) = (Mx, My, Mz). This is fundamentally ill-posed — phase information is lost in the measurement process — making it one of the harder inverse problems in magnetic imaging.

The project progresses through four stages, from simple single-vector recovery to a full hybrid classical+ML pipeline:

---

## Repository Structure

```
Scattering-UCSD/
│
├── matlab/
│   ├── 3d model/               # Full 3D XMCD reconstruction pipeline
│   │   ├── Master.m            # Pipeline orchestrator — run this
│   │   ├── module1.m           # 3D magnetic vector field generator (Bloch/Néel domains)
│   │   ├── module2.m           # Forward scattering model (Ewald sphere, XMCD, Beer-Lambert)
│   │   ├── module3.m           # Sensing matrix + rank diagnostics
│   │   ├── module4.m           # RAAR phase retrieval
│   │   ├── module4_ls.m        # Direct least-squares inversion (no phase retrieval)
│   │   ├── module5.m           # CNN seed module
│   │   ├── module6.m           # RESIRE-V gradient descent refinement (Adam optimizer)
│   │   ├── module8.m           # Error analysis and figures
│   │   └── Results/            # Output figures and results PDF
│   │
│   └── Single Vector/          # Single spin vector recovery experiments
│       ├── onevector.m         # Noiseless least-squares recovery
│       ├── onevectornoise.m    # Tikhonov regularized least-squares
│       ├── onevecresire.m      # RESIRE-V iterative recovery (single vector)
│       └── Results/            # Output figures and results PDF
│
├── python/
│   ├── Benchmark Ml testing/          # Architecture benchmark: 5 models head-to-head
│   │   ├── train_all_inverse_scattering_models_v2_angle_conditioned_updated.py
│   │   └── Results/                   # Per-model metrics, figures, trained weights (.pt)
│   │       ├── metrics_summary.json   # All test MSE results
│   │       ├── cnn/
│   │       ├── Unet/
│   │       ├── transformer/
│   │       ├── FNO/
│   │       └── Neural/
│   │
│   ├── MagCNN / Physics-Informed U-Net/   # 16M-param attention U-Net experiments
│   │   ├── MagCNN_Colab_fixed.ipynb       # Physics-informed loss U-Net (Colab, T4 GPU)
│   │   └── MagCNN_Results_Summary.pdf
│   │
│   └── MagXMCD_v9_HybridPipeline/         # Hybrid classical+CNN pipeline (main result)
│       ├── MagXMCD_v9.ipynb               # Full hybrid pipeline notebook
│       └── results/                       # All 7 pipeline reconstruction results
│           ├── run_config.json
│           ├── reconstructions.npz
│           ├── training_histories.npz
│           ├── testrun_CNN_alone.png
│           ├── testrun_RAAR_alone.png
│           ├── testrun_GENFIRE_alone.png
│           ├── testrun_RESIRE_alone.png
│           ├── testrun_RAAR_to_CNN.png
│           ├── testrun_GENFIRE_to_CNN.png
│           └── testrun_RESIRE_to_CNN.png
```

---

## Development Timeline

### Stage 1 — Single Vector Recovery (`matlab/Single Vector/`)
**Question:** Can we recover a single spin vector from scalar XMCD projections?

Direct least-squares (`onevector.m`, `onevectornoise.m`) recovers a 3-component spin vector **perfectly** from a set of angle-sampled dot-product measurements. RESIRE-V iterative reconstruction (`onevecresire.m`) completely diverges (~10²⁰% error), establishing that the iterative approach requires careful calibration even at the simplest possible scale.

### Stage 2 — ML Architecture Benchmark (`python/Benchmark Ml testing/`)
**Question:** Which ML architecture best learns the 2D inverse scattering map?

Five architectures (CNN, U-Net, Transformer, FNO, Neural ODE) are trained on 2,000 angle-conditioned synthetic samples for 30 epochs. **U-Net wins** (test MSE = 0.202). Critical finding: **no architecture recovers Mx or My** — all five produce identical MSE ~0.125 for in-plane components, confirming a fundamental sensing geometry limitation rather than a model capacity issue.

| Model | Test MSE | Params | Train Time |
|-------|----------|--------|------------|
| **U-Net** | **0.202** | 2.0M | 1.5 hr |
| FNO | 0.258 | 7.1M | 1.0 hr |
| Transformer | 0.262 | 985K | 0.5 hr |
| Neural ODE | 0.282 | 176K | 22.9 hr (CPU) |
| CNN | 0.321 | 171K | 2.0 hr |

### Stage 3 — Physics-Informed Attention U-Net (`python/MagCNN/`)
**Question:** Can a larger, physics-regularized U-Net solve the 2D problem?

A 16M-parameter attention U-Net with GELU activations, residual blocks, and physics-informed losses (unit norm constraint + total variation) is trained on 4,000 synthetic samples on a Colab T4 GPU. **Both runs fail** — angular error flatlines at ~87° (random baseline = 90°). Run v1 (pure MSE) oscillates around random; Run v2 (physics loss) diverges monotonically. Confirms the failure is due to the ill-posed phase retrieval problem, not model capacity.

### Stage 4 — Hybrid Classical+CNN Pipeline (`python/MagXMCD_v9_HybridPipeline/`) ← **Main Result**
**Question:** Can a CNN refine the output of classical phase retrieval algorithms?

Three classical phase retrieval algorithms (RAAR, GENFIRE, RESIRE) are tested standalone and as preprocessors feeding into a CNN, giving 7 total pipelines:

| Pipeline | Type | Mz Accuracy | MSE |
|----------|------|-------------|-----|
| CNN alone | Pure ML | 52.2% | 0.981 |
| RAAR alone | Classical | 74.1% | 0.513 |
| GENFIRE alone | Classical | 74.1% | 0.513 |
| RESIRE alone | Classical | 74.2% | 0.513 |
| RAAR→CNN | Hybrid | 74.9% | 0.436 |
| **GENFIRE→CNN** | **Hybrid** | **75.0%** | **0.512** |
| RESIRE→CNN | Hybrid | 74.3% | 0.521 |

**Key finding:** Classical phase retrieval does the heavy lifting (~74% accuracy). CNN refinement adds ~0.7–0.9%. CNN alone barely beats random (52.2% vs 50%). The answer to the inverse problem is: **phase retrieval first, CNN refinement second**.

---

## Requirements

### MATLAB Pipeline
- MATLAB R2024b or later
- No additional toolboxes required

### Python / Jupyter
```
tensorflow >= 2.20
numpy < 2.0
scipy
matplotlib
torch (for benchmark script)
jupyter
```

Install Python dependencies:
```bash
pip install tensorflow "numpy<2" scipy matplotlib torch jupyter
```

---

## How to Run

### 3D MATLAB Pipeline
1. Open MATLAB and navigate to `matlab/3d model/`
2. Set `algo_mode` at the top of `Master.m`:
   - `'raar'` — RAAR phase retrieval + LS seed
   - `'ls_direct'` — direct least-squares (recommended starting point)
   - `'truefield'` — oracle test with true field seed
3. Run `Master.m`

Full pipeline takes ~8–9 minutes (RAAR: 86s, RESIRE-V: 418s).

### ML Benchmark (5 architectures)
```bash
cd "python/Benchmark Ml testing"
python train_all_inverse_scattering_models_v2_angle_conditioned_updated.py
```
Results saved to a timestamped folder in your working directory.

### Hybrid Pipeline (MagXMCD v9)
1. Open `python/MagXMCD_v9_HybridPipeline/MagXMCD_v9.ipynb` in Jupyter or Google Colab
2. Set runtime to **T4 GPU** in Colab (recommended)
3. Run all cells — training takes ~15–20 minutes on T4

### Physics-Informed U-Net (MagCNN)
1. Upload `python/MagCNN/MagCNN_Colab_fixed.ipynb` to Google Colab
2. Set runtime to **T4 GPU**
3. Run all cells

---

## Key Results Summary

| Experiment | Best Result | Method |
|------------|-------------|--------|
| Single vector recovery | **Perfect (0% error)** | Direct least-squares |
| 3D MATLAB pipeline | **47.8° mean angular error** | LS direct → RESIRE-V Adam |
| ML architecture benchmark | **MSE = 0.202** | U-Net (angle-conditioned) |
| Physics-informed U-Net | **87° angular error (failed)** | 16M-param attention U-Net |
| Hybrid pipeline | **75.0% Mz domain accuracy** | GENFIRE→CNN |

---

## Beamline Parameters (Simulation)

All simulations use parameters matched to the ALS COSMIC beamline:

| Parameter | Value |
|-----------|-------|
| Photon energy | 707.0 eV (Fe L-edge) |
| Wavelength | 1.7537 nm |
| Detector | 512 × 512 px, 48 µm pixel |
| Sample-detector distance | 150 mm |
| Theta range | 15°–60° (8 steps) |
| Phi positions | 0°, 45°, 90°, 135° |
| Total angle pairs | 32 |
| Absorption length | 30 nm |

---

## Citation

If you use this code, please cite:

> Murray, B. (2026). *Physics-Informed Machine Learning for Magnetic Domain Reconstruction from XMCD Scattering*. ECE 228 Final Project, UC San Diego.
