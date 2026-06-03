
# ============================================================
# train_all_inverse_scattering_models_v2_angle_conditioned.py
# ============================================================
#
# Version B:
#   Input to each model:
#       1. Scattering image I(qx,qy): [B, 1, H, W]
#       2. Geometry / angle metadata: [B, metadata_dim]
#
#   Output:
#       Magnetic vector field M(x,y): [B, 3, H, W]
#
# Models trained back-to-back:
#   1. CNN
#   2. U-Net
#   3. Transformer / self-attention
#   4. FNO
#   5. Neural ODE-style model
#
# Optional:
#   6. Conditional diffusion model
#
# Everything saves to:
#   ~/Downloads/InverseScatteringResults_YYYYMMDD_HHMMSS/
#
# ============================================================

import os
import csv
import json
import math
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# CONFIG
# ============================================================

class Config:
    # ----------------------------
    # Dataset
    # ----------------------------
    num_samples = 2000
    nx = 128
    ny = 128

    # ----------------------------
    # Training
    # ----------------------------
    batch_size = 8
    epochs = 30
    lr = 1e-3
    train_fraction = 0.8
    val_fraction = 0.1
    seed = 123

    # ----------------------------
    # Model switches
    # ----------------------------
    train_cnn = True
    train_unet = True
    train_transformer = True
    train_fno = True
    train_neural_ode = True

    # Diffusion is included but off by default.
    # Turn this on after the first five models work.
    train_diffusion = False
    diffusion_epochs = 10
    diffusion_steps = 100

    # ----------------------------
    # Metadata conditioning
    # ----------------------------
    # Metadata vector:
    #   [theta_i, phi_i, theta_s, phi_s, delta_theta, delta_phi,
    #    sensitivity_x, sensitivity_y, sensitivity_z]
    metadata_dim = 9

    # ----------------------------
    # Output folder
    # ----------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path.home() / "Downloads" / f"InverseScatteringResults_{timestamp}"

    # ----------------------------
    # Device
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# SYNTHETIC MAGNETIC VECTOR FIELD GENERATOR
# ============================================================

def generate_vector_field(
    nx=128,
    ny=128,
    domain_width=16,
    wall_smoothness=2.0,
    angle_noise=0.15,
    curvature_strength=0.10,
):
    """
    Creates a 2D magnetic vector field.

    Output:
        M: [3, ny, nx]
    """

    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    stripe_coord = np.arange(nx)[None, :].astype(np.float32)

    curve = curvature_strength * nx * np.sin(
        2.0 * np.pi * Y * np.random.uniform(0.5, 2.0)
    )

    stripe_coord_2d = stripe_coord + curve

    domain_index = np.floor(stripe_coord_2d / domain_width)
    sign = (-1.0) ** domain_index

    wall_phase = (stripe_coord_2d % domain_width) - domain_width / 2.0
    wall_profile = np.tanh(wall_phase / wall_smoothness)

    # Mostly up/down magnetic domains with smooth walls.
    theta = np.where(sign > 0, 0.0, np.pi)
    theta = theta + 0.5 * np.pi * (1.0 - wall_profile)

    # Random in-plane azimuth plus noise.
    phi0 = np.random.uniform(0.0, 2.0 * np.pi)
    phi = phi0 + angle_noise * np.random.randn(ny, nx)

    Mx = np.sin(theta) * np.cos(phi)
    My = np.sin(theta) * np.sin(phi)
    Mz = np.cos(theta)

    mag = np.sqrt(Mx**2 + My**2 + Mz**2)
    mag[mag < 1e-12] = 1.0

    Mx = Mx / mag
    My = My / mag
    Mz = Mz / mag

    M = np.stack([Mx, My, Mz], axis=0)

    return M.astype(np.float32)


# ============================================================
# ANGLE / GEOMETRY UTILITIES
# ============================================================

def unit_vector_from_angles(theta_deg, phi_deg):
    """
    theta: polar angle from z-axis, degrees
    phi: azimuthal angle in x-y plane, degrees
    """

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float32,
    )


def normalize_metadata(raw_metadata):
    """
    Normalizes angle metadata to roughly order-one values.

    Input raw metadata:
        [theta_i, phi_i, theta_s, phi_s, delta_theta, delta_phi,
         sx, sy, sz]

    Angles are divided by 90 degrees.
    Sensitivity components are already between -1 and 1.
    """

    meta = raw_metadata.copy().astype(np.float32)

    meta[0:6] = meta[0:6] / 90.0

    return meta


# ============================================================
# ANGLE-CONDITIONED SYNTHETIC SCATTERING MODEL
# ============================================================

def compute_angle_scattering_image(
    M,
    incident_theta_deg,
    incident_phi_deg,
    scatter_theta_deg,
    scatter_phi_deg,
):
    """
    Simplified angle-dependent magnetic scattering.

    The current toy amplitude is:

        rho_m(x,y) = s · M(x,y)

    where

        s = normalize(k_s x k_i)

    and

        I(qx,qy) = |FFT[rho_m]|^2.

    This is not a full XRMS model, but it introduces incident/scattered
    geometry and component-dependent magnetic sensitivity.
    """

    k_i = unit_vector_from_angles(incident_theta_deg, incident_phi_deg)
    k_s = unit_vector_from_angles(scatter_theta_deg, scatter_phi_deg)

    q_vec = k_s - k_i
    q_hat = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    sensitivity = np.cross(k_s, k_i)

    if np.linalg.norm(sensitivity) < 1e-8:
        sensitivity = q_hat

    sensitivity = sensitivity / (np.linalg.norm(sensitivity) + 1e-12)

    Mx = M[0]
    My = M[1]
    Mz = M[2]

    rho_m = (
        sensitivity[0] * Mx
        + sensitivity[1] * My
        + sensitivity[2] * Mz
    )

    A_q = np.fft.fftshift(np.fft.fft2(rho_m))

    I_q = np.abs(A_q) ** 2
    I_q = np.log1p(I_q)

    I_q = I_q - I_q.mean()
    I_q = I_q / (I_q.std() + 1e-8)

    raw_metadata = np.array(
        [
            incident_theta_deg,
            incident_phi_deg,
            scatter_theta_deg,
            scatter_phi_deg,
            scatter_theta_deg - incident_theta_deg,
            scatter_phi_deg - incident_phi_deg,
            sensitivity[0],
            sensitivity[1],
            sensitivity[2],
        ],
        dtype=np.float32,
    )

    metadata = normalize_metadata(raw_metadata)

    return (
        I_q[None, :, :].astype(np.float32),
        metadata.astype(np.float32),
        raw_metadata.astype(np.float32),
        sensitivity.astype(np.float32),
    )


def generate_dataset(num_samples=2000, nx=128, ny=128):
    """
    Generates:
        X:            [N, 1, H, W]
        Y:            [N, 3, H, W]
        metadata:     [N, 9]
        raw_metadata: [N, 9]
        sensitivities:[N, 3]
    """

    X_list = []
    Y_list = []
    metadata_list = []
    raw_metadata_list = []
    sensitivity_list = []

    for n in range(num_samples):
        domain_width = np.random.randint(8, 28)
        wall_smoothness = np.random.uniform(1.0, 5.0)
        angle_noise = np.random.uniform(0.02, 0.25)
        curvature_strength = np.random.uniform(0.0, 0.18)

        M = generate_vector_field(
            nx=nx,
            ny=ny,
            domain_width=domain_width,
            wall_smoothness=wall_smoothness,
            angle_noise=angle_noise,
            curvature_strength=curvature_strength,
        )

        # Wider angular coverage is important because a single scattering
        # image measures only one geometry-dependent projection of M.
        incident_theta = np.random.uniform(5.0, 75.0)
        incident_phi = np.random.uniform(-180.0, 180.0)

        scatter_theta = np.random.uniform(5.0, 85.0)
        scatter_phi = np.random.uniform(-180.0, 180.0)

        I, metadata, raw_metadata, sensitivity = compute_angle_scattering_image(
            M,
            incident_theta,
            incident_phi,
            scatter_theta,
            scatter_phi,
        )

        X_list.append(I)
        Y_list.append(M)
        metadata_list.append(metadata)
        raw_metadata_list.append(raw_metadata)
        sensitivity_list.append(sensitivity)

        if (n + 1) % 100 == 0:
            print(f"Generated {n + 1}/{num_samples}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    metadata = np.stack(metadata_list, axis=0).astype(np.float32)
    raw_metadata = np.stack(raw_metadata_list, axis=0).astype(np.float32)
    sensitivities = np.stack(sensitivity_list, axis=0).astype(np.float32)

    return X, Y, metadata, raw_metadata, sensitivities


# ============================================================
# DATASET CLASS
# ============================================================

class ScatteringDataset(Dataset):
    def __init__(self, X, Y, metadata):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.metadata[idx], self.Y[idx]


# ============================================================
# CONDITIONING BLOCKS
# ============================================================

class MetadataToFeatureMap(nn.Module):
    """
    Converts metadata vector into a spatial feature map.

    metadata: [B, metadata_dim]
    output:   [B, out_channels, H, W]
    """

    def __init__(self, metadata_dim, out_channels):
        super().__init__()

        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, metadata, H, W):
        B = metadata.shape[0]
        z = self.net(metadata)
        z = z.view(B, self.out_channels, 1, 1)
        z = z.expand(B, self.out_channels, H, W)

        return z


class MetadataBias(nn.Module):
    """
    Converts metadata into a channel-wise bias.
    """

    def __init__(self, metadata_dim, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(metadata_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, x, metadata):
        B, C, H, W = x.shape
        bias = self.net(metadata).view(B, C, 1, 1)
        return x + bias


# ============================================================
# COMMON CONV BLOCK
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# MODEL 1 — ANGLE-CONDITIONED CNN
# ============================================================

class ConditionedCNN(nn.Module):
    def __init__(self, metadata_dim=9, cond_channels=8):
        super().__init__()

        self.meta_map = MetadataToFeatureMap(metadata_dim, cond_channels)

        self.net = nn.Sequential(
            ConvBlock(1 + cond_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, metadata):
        B, C, H, W = x.shape
        m = self.meta_map(metadata, H, W)
        x = torch.cat([x, m], dim=1)
        return self.net(x)


# ============================================================
# MODEL 2 — ANGLE-CONDITIONED U-NET
# ============================================================

class ConditionedUNet(nn.Module):
    def __init__(self, metadata_dim=9, cond_channels=8):
        super().__init__()

        self.meta_map = MetadataToFeatureMap(metadata_dim, cond_channels)

        self.enc1 = ConvBlock(1 + cond_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.meta_bottleneck = MetadataBias(metadata_dim, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, metadata):
        B, C, H, W = x.shape

        meta = self.meta_map(metadata, H, W)
        x = torch.cat([x, meta], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))
        b = self.meta_bottleneck(b, metadata)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# MODEL 3 — ANGLE-CONDITIONED PATCH TRANSFORMER
# ============================================================

class ConditionedPatchTransformer(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        metadata_dim=9,
        d_model=128,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, d_model)

        self.meta_embed = nn.Sequential(
            nn.Linear(metadata_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Instead of directly decoding every patch with one linear layer,
        # reshape tokens into a low-resolution feature map and use a
        # convolutional decoder. This reduces block artifacts at patch edges.
        self.feature_proj = nn.Linear(d_model, d_model)

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, metadata):
        B, C, H, W = x.shape
        p = self.patch_size

        patches = F.unfold(x, kernel_size=p, stride=p)
        patches = patches.transpose(1, 2)

        tokens = self.patch_embed(patches)

        meta_token = self.meta_embed(metadata).unsqueeze(1)
        tokens = tokens + self.pos + meta_token

        z = self.encoder(tokens)
        z = self.feature_proj(z)

        # [B, num_patches, d_model] -> [B, d_model, H/p, W/p]
        z = z.transpose(1, 2)
        z = z.reshape(
            B,
            -1,
            self.num_patches_per_side,
            self.num_patches_per_side,
        )

        out = self.conv_decoder(z)

        # For patch_size=8 and img_size=128, three stride-2 upsamplings
        # return exactly 128 x 128.
        if out.shape[-2:] != (H, W):
            out = F.interpolate(
                out,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return out


# ============================================================
# MODEL 4 — ANGLE-CONDITIONED FNO
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x=32, modes_y=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (in_channels * out_channels)

        self.weights = nn.Parameter(
            scale * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(self, input_fft, weights):
        return torch.einsum("bixy,ioxy->boxy", input_fft, weights)

    def forward(self, x):
        B, C, H, W = x.shape

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mx = min(self.modes_x, H)
        my = min(self.modes_y, W // 2 + 1)

        out_ft[:, :, :mx, :my] = self.compl_mul2d(
            x_ft[:, :, :mx, :my],
            self.weights[:, :, :mx, :my],
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))

        return x


class ConditionedFNO2d(nn.Module):
    def __init__(
        self,
        metadata_dim=9,
        cond_channels=8,
        width=48,
        modes_x=32,
        modes_y=32,
    ):
        super().__init__()

        self.meta_map = MetadataToFeatureMap(metadata_dim, cond_channels)

        self.lift = nn.Conv2d(1 + cond_channels, width, kernel_size=1)

        self.spec1 = SpectralConv2d(width, width, modes_x, modes_y)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)

        self.spec2 = SpectralConv2d(width, width, modes_x, modes_y)
        self.w2 = nn.Conv2d(width, width, kernel_size=1)

        self.spec3 = SpectralConv2d(width, width, modes_x, modes_y)
        self.w3 = nn.Conv2d(width, width, kernel_size=1)

        self.meta_bias = MetadataBias(metadata_dim, width)

        self.proj = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x, metadata):
        B, C, H, W = x.shape

        meta = self.meta_map(metadata, H, W)
        x = torch.cat([x, meta], dim=1)

        x = self.lift(x)

        x = F.gelu(self.spec1(x) + self.w1(x))
        x = F.gelu(self.spec2(x) + self.w2(x))
        x = F.gelu(self.spec3(x) + self.w3(x))

        x = self.meta_bias(x, metadata)

        return self.proj(x)


# ============================================================
# MODEL 5 — ANGLE-CONDITIONED NEURAL ODE STYLE MODEL
# ============================================================

class ODEFunc(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.Tanh(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.Tanh(),
        )

    def forward(self, h):
        return self.net(h)


class ConditionedNeuralODEStyleNet(nn.Module):
    """
    No torchdiffeq dependency.

    Uses fixed RK4 steps:
        dh/dt = f(h)
    """

    def __init__(
        self,
        metadata_dim=9,
        cond_channels=8,
        channels=64,
        steps=6,
    ):
        super().__init__()

        self.meta_map = MetadataToFeatureMap(metadata_dim, cond_channels)

        self.encoder = nn.Sequential(
            ConvBlock(1 + cond_channels, 32),
            ConvBlock(32, channels),
        )

        self.meta_bias = MetadataBias(metadata_dim, channels)

        self.f = ODEFunc(channels)
        self.steps = steps

        self.decoder = nn.Sequential(
            ConvBlock(channels, 32),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh(),
        )

    def rk4_step(self, h, dt):
        k1 = self.f(h)
        k2 = self.f(h + 0.5 * dt * k1)
        k3 = self.f(h + 0.5 * dt * k2)
        k4 = self.f(h + dt * k3)

        return h + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def forward(self, x, metadata):
        B, C, H, W = x.shape

        meta = self.meta_map(metadata, H, W)
        x = torch.cat([x, meta], dim=1)

        h = self.encoder(x)
        h = self.meta_bias(h, metadata)

        dt = 1.0 / self.steps

        for _ in range(self.steps):
            h = self.rk4_step(h, dt)

        return self.decoder(h)


# ============================================================
# OPTIONAL MODEL 6 — ANGLE-CONDITIONED DIFFUSION MODEL
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.dim // 2

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device).float()
            / max(half - 1, 1)
        )

        args = t[:, None].float() * freqs[None, :]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))

        return self.net(emb)


class ConditionedDiffusionUNet(nn.Module):
    def __init__(self, metadata_dim=9, base=32, tdim=128):
        super().__init__()

        self.time = TimeEmbedding(tdim)

        self.meta = nn.Sequential(
            nn.Linear(metadata_dim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = ConvBlock(4, base)

        self.pool = nn.MaxPool2d(2)

        self.down1 = ConvBlock(base, base * 2)
        self.down2 = ConvBlock(base * 2, base * 4)

        self.mid = ConvBlock(base * 4, base * 4)

        self.t_proj_mid = nn.Linear(tdim, base * 4)
        self.t_proj_d2 = nn.Linear(tdim, base * 2)
        self.t_proj_d1 = nn.Linear(tdim, base)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 3, kernel_size=1)

    def add_embedding(self, x, emb, proj):
        B, C, H, W = x.shape
        b = proj(emb).view(B, C, 1, 1)
        return x + b

    def forward(self, x_t, cond_image, metadata, t):
        emb = self.time(t) + self.meta(metadata)

        x = torch.cat([x_t, cond_image], dim=1)

        e1 = self.in_conv(x)
        e2 = self.down1(self.pool(e1))
        e3 = self.down2(self.pool(e2))

        b = self.mid(e3)
        b = self.add_embedding(b, emb, self.t_proj_mid)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.add_embedding(d2, emb, self.t_proj_d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.add_embedding(d1, emb, self.t_proj_d1)

        return self.out(d1)


class DiffusionSchedule:
    def __init__(self, T=100, device="cpu"):
        self.T = T
        self.device = device

        beta = torch.linspace(1e-4, 0.02, T, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar

    def q_sample(self, x0, t, noise):
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


# ============================================================
# METRICS
# ============================================================

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_metrics(pred, true):
    with torch.no_grad():
        err = pred - true

        mse_mx = torch.mean(err[:, 0] ** 2).item()
        mse_my = torch.mean(err[:, 1] ** 2).item()
        mse_mz = torch.mean(err[:, 2] ** 2).item()
        mse_total = torch.mean(err ** 2).item()

        err_norm = torch.sqrt(torch.sum(err ** 2, dim=1))
        mean_error_norm = torch.mean(err_norm).item()

    return {
        "mse_mx": mse_mx,
        "mse_my": mse_my,
        "mse_mz": mse_mz,
        "mse_total": mse_total,
        "mean_error_norm": mean_error_norm,
    }


# ============================================================
# FIGURE UTILITIES
# ============================================================

def save_loss_curve(train_losses, val_losses, path, title):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_dataset_preview(X, Y, path):
    I = X[0, 0]
    Mx = Y[0, 0]
    My = Y[0, 1]
    Mz = Y[0, 2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].imshow(I)
    axes[0].set_title("Scattering Pattern I(q)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(Mx, vmin=-1, vmax=1)
    axes[1].set_title("True Mx")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(My, vmin=-1, vmax=1)
    axes[2].set_title("True My")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(Mz, vmin=-1, vmax=1)
    axes[3].set_title("True Mz")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_angle_histograms(raw_metadata, path):
    """
    raw_metadata columns:
        0 theta_i
        1 phi_i
        2 theta_s
        3 phi_s
        4 delta_theta
        5 delta_phi
        6 sx
        7 sy
        8 sz
    """

    labels = [
        r"$\theta_i$",
        r"$\phi_i$",
        r"$\theta_s$",
        r"$\phi_s$",
        r"$\Delta\theta$",
        r"$\Delta\phi$",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    for i in range(6):
        axes[i].hist(raw_metadata[:, i], bins=40)
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("degrees")
        axes[i].set_ylabel("count")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_sensitivity_histograms(sensitivities, path):
    labels = [r"$s_x$", r"$s_y$", r"$s_z$"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for i in range(3):
        axes[i].hist(sensitivities[:, i], bins=40)
        axes[i].set_title(labels[i])
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_truth_and_scattering(x, y, path_truth, path_scattering):
    I = x[0]
    Mx = y[0]
    My = y[1]
    Mz = y[2]

    plt.figure(figsize=(5, 5))
    plt.imshow(I)
    plt.title("Test Scattering Pattern I(q)")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path_scattering, dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    comps = [Mx, My, Mz]
    names = ["Mx", "My", "Mz"]

    for ax, comp, name in zip(axes, comps, names):
        im = ax.imshow(comp, vmin=-1, vmax=1)
        ax.set_title(f"True {name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(path_truth, dpi=200)
    plt.close()


def save_reconstruction_figure(x, y_true, y_pred, path, title):
    I = x[0]
    names = ["Mx", "My", "Mz"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row in range(3):
        true_comp = y_true[row]
        pred_comp = y_pred[row]
        err_comp = pred_comp - true_comp

        im0 = axes[row, 0].imshow(I)
        axes[row, 0].set_title("Input I(q)")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(true_comp, vmin=-1, vmax=1)
        axes[row, 1].set_title(f"True {names[row]}")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(pred_comp, vmin=-1, vmax=1)
        axes[row, 2].set_title(f"Pred {names[row]}")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        im3 = axes[row, 3].imshow(err_comp)
        axes[row, 3].set_title(f"Error {names[row]}")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        for col in range(4):
            axes[row, col].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_model_barplot(metrics, path):
    names = [m["model"] for m in metrics]
    mse = [m["mse_total"] for m in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(names, mse)
    plt.ylabel("Total Test MSE")
    plt.title("Model Comparison: Reconstruction Error")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_training_time_barplot(metrics, path):
    names = [m["model"] for m in metrics]
    times = [m["train_time_sec"] for m in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(names, times)
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_parameter_count_barplot(metrics, path):
    names = [m["model"] for m in metrics]
    params = [m["num_parameters"] for m in metrics]

    plt.figure(figsize=(10, 5))
    plt.bar(names, params)
    plt.ylabel("Trainable Parameters")
    plt.title("Model Size Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_all_models_side_by_side(shared_example, predictions, path):
    """
    Saves one comparison figure for all vector components.

    Rows:
        Mx, My, Mz

    Columns:
        input scattering pattern, true field, model predictions
    """

    x = shared_example["x"]
    y = shared_example["y"]

    model_names = list(predictions.keys())
    comp_names = ["Mx", "My", "Mz"]

    ncols = len(model_names) + 2

    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))

    for row in range(3):
        axes[row, 0].imshow(x[0])
        axes[row, 0].set_title("Input I(q)")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(y[row], vmin=-1, vmax=1)
        axes[row, 1].set_title(f"True {comp_names[row]}")
        axes[row, 1].axis("off")

        for i, name in enumerate(model_names):
            axes[row, i + 2].imshow(predictions[name][row], vmin=-1, vmax=1)
            axes[row, i + 2].set_title(f"{name}\nPred {comp_names[row]}")
            axes[row, i + 2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# TRAINING STANDARD MODELS
# ============================================================

def train_standard_model(model, model_name, train_loader, val_loader, cfg):
    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    model_dir = cfg.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_path = model_dir / "best_model.pt"

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        train_sum = 0.0
        train_count = 0

        for x, metadata, y in train_loader:
            x = x.to(cfg.device)
            metadata = metadata.to(cfg.device)
            y = y.to(cfg.device)

            optimizer.zero_grad()

            pred = model(x, metadata)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            train_sum += loss.item() * x.shape[0]
            train_count += x.shape[0]

        train_loss = train_sum / train_count

        model.eval()

        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x, metadata, y in val_loader:
                x = x.to(cfg.device)
                metadata = metadata.to(cfg.device)
                y = y.to(cfg.device)

                pred = model(x, metadata)
                loss = loss_fn(pred, y)

                val_sum += loss.item() * x.shape[0]
                val_count += x.shape[0]

        val_loss = val_sum / val_count

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train = {train_loss:.6f} | val = {val_loss:.6f}"
        )

    elapsed = time.time() - start_time

    model.load_state_dict(torch.load(best_path, map_location=cfg.device))

    save_loss_curve(
        train_losses,
        val_losses,
        model_dir / "loss_curve.png",
        model_name,
    )

    return model, {
        "model": model_name,
        "best_val_loss": best_val,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "train_time_sec": elapsed,
        "num_parameters": count_trainable_parameters(model),
    }


def save_reconstruction_metadata_json(record, model_name, label, path):
    """
    Saves the scalar information associated with a reconstruction example.
    """

    data = {
        "model": model_name,
        "case": label,
        "global_index": int(record["global_index"]),
        "mse": float(record["mse"]),
    }

    if "metadata" in record:
        data["metadata_normalized"] = [float(v) for v in record["metadata"]]

    if "raw_metadata" in record:
        data["metadata_raw"] = [float(v) for v in record["raw_metadata"]]

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


@torch.no_grad()
def evaluate_standard_model(
    model,
    model_name,
    test_loader,
    cfg,
    shared_batch=None,
):
    model.eval()

    model_dir = cfg.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    all_batch_metrics = []
    sample_records = []

    global_index = 0

    for x, metadata, y in test_loader:
        x = x.to(cfg.device)
        metadata = metadata.to(cfg.device)
        y = y.to(cfg.device)

        pred = model(x, metadata)

        batch_metrics = mse_metrics(pred, y)
        all_batch_metrics.append(batch_metrics)

        # Per-sample MSE for best/worst/median reconstructions.
        per_sample = torch.mean((pred - y) ** 2, dim=(1, 2, 3))

        for i in range(x.shape[0]):
            sample_records.append(
                {
                    "global_index": global_index,
                    "mse": per_sample[i].item(),
                    "x": x[i].detach().cpu().numpy(),
                    "y": y[i].detach().cpu().numpy(),
                    "pred": pred[i].detach().cpu().numpy(),
                    "metadata": metadata[i].detach().cpu().numpy(),
                }
            )
            global_index += 1

    avg = {}
    for key in all_batch_metrics[0].keys():
        avg[key] = float(np.mean([m[key] for m in all_batch_metrics]))

    # Save best / median / worst.
    sample_records = sorted(sample_records, key=lambda r: r["mse"])

    best = sample_records[0]
    median = sample_records[len(sample_records) // 2]
    worst = sample_records[-1]

    save_reconstruction_figure(
        best["x"],
        best["y"],
        best["pred"],
        model_dir / "best_reconstruction.png",
        f"{model_name} Best Reconstruction | MSE={best['mse']:.6f}",
    )

    save_reconstruction_figure(
        median["x"],
        median["y"],
        median["pred"],
        model_dir / "median_reconstruction.png",
        f"{model_name} Median Reconstruction | MSE={median['mse']:.6f}",
    )

    save_reconstruction_figure(
        worst["x"],
        worst["y"],
        worst["pred"],
        model_dir / "worst_reconstruction.png",
        f"{model_name} Worst Reconstruction | MSE={worst['mse']:.6f}",
    )

    save_reconstruction_metadata_json(
        best,
        model_name,
        "best",
        model_dir / "best_reconstruction_metadata.json",
    )

    save_reconstruction_metadata_json(
        median,
        model_name,
        "median",
        model_dir / "median_reconstruction_metadata.json",
    )

    save_reconstruction_metadata_json(
        worst,
        model_name,
        "worst",
        model_dir / "worst_reconstruction_metadata.json",
    )

    # Prediction on the shared example for side-by-side model comparison.
    shared_pred = None
    if shared_batch is not None:
        x0 = shared_batch["x"].to(cfg.device)
        m0 = shared_batch["metadata"].to(cfg.device)

        p0 = model(x0, m0)

        shared_pred = p0[0].detach().cpu().numpy()

    return avg, shared_pred


# ============================================================
# TRAINING OPTIONAL DIFFUSION MODEL
# ============================================================

def train_diffusion_model(model, model_name, train_loader, val_loader, cfg):
    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)

    model_dir = cfg.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(cfg.device)
    schedule = DiffusionSchedule(T=cfg.diffusion_steps, device=cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_path = model_dir / "best_model.pt"

    start_time = time.time()

    for epoch in range(1, cfg.diffusion_epochs + 1):
        model.train()

        train_sum = 0.0
        train_count = 0

        for x, metadata, y0 in train_loader:
            x = x.to(cfg.device)
            metadata = metadata.to(cfg.device)
            y0 = y0.to(cfg.device)

            B = y0.shape[0]

            t = torch.randint(0, schedule.T, (B,), device=cfg.device)
            noise = torch.randn_like(y0)
            y_t = schedule.q_sample(y0, t, noise)

            optimizer.zero_grad()
            noise_pred = model(y_t, x, metadata, t)

            loss = loss_fn(noise_pred, noise)
            loss.backward()
            optimizer.step()

            train_sum += loss.item() * B
            train_count += B

        train_loss = train_sum / train_count

        model.eval()

        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x, metadata, y0 in val_loader:
                x = x.to(cfg.device)
                metadata = metadata.to(cfg.device)
                y0 = y0.to(cfg.device)

                B = y0.shape[0]

                t = torch.randint(0, schedule.T, (B,), device=cfg.device)
                noise = torch.randn_like(y0)
                y_t = schedule.q_sample(y0, t, noise)

                noise_pred = model(y_t, x, metadata, t)
                loss = loss_fn(noise_pred, noise)

                val_sum += loss.item() * B
                val_count += B

        val_loss = val_sum / val_count

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:03d}/{cfg.diffusion_epochs} | "
            f"train = {train_loss:.6f} | val = {val_loss:.6f}"
        )

    elapsed = time.time() - start_time

    model.load_state_dict(torch.load(best_path, map_location=cfg.device))

    save_loss_curve(
        train_losses,
        val_losses,
        model_dir / "loss_curve.png",
        model_name,
    )

    return model, schedule, {
        "model": model_name,
        "best_val_loss": best_val,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "train_time_sec": elapsed,
        "num_parameters": count_trainable_parameters(model),
    }


@torch.no_grad()
def sample_diffusion(model, schedule, cond_image, metadata):
    model.eval()

    B, _, H, W = cond_image.shape

    y = torch.randn(B, 3, H, W, device=cond_image.device)

    for step in reversed(range(schedule.T)):
        t = torch.full((B,), step, device=cond_image.device, dtype=torch.long)

        beta_t = schedule.beta[t].view(-1, 1, 1, 1)
        alpha_t = schedule.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = schedule.alpha_bar[t].view(-1, 1, 1, 1)

        noise_pred = model(y, cond_image, metadata, t)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            y - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )

        if step > 0:
            z = torch.randn_like(y)
            y = mean + torch.sqrt(beta_t) * z
        else:
            y = mean

    return torch.tanh(y)


@torch.no_grad()
def evaluate_diffusion_model(
    model,
    schedule,
    model_name,
    test_loader,
    cfg,
    shared_batch=None,
):
    model.eval()

    model_dir = cfg.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    all_batch_metrics = []
    sample_records = []
    global_index = 0

    for x, metadata, y in test_loader:
        x = x.to(cfg.device)
        metadata = metadata.to(cfg.device)
        y = y.to(cfg.device)

        pred = sample_diffusion(model, schedule, x, metadata)

        batch_metrics = mse_metrics(pred, y)
        all_batch_metrics.append(batch_metrics)

        per_sample = torch.mean((pred - y) ** 2, dim=(1, 2, 3))

        for i in range(x.shape[0]):
            sample_records.append(
                {
                    "global_index": global_index,
                    "mse": per_sample[i].item(),
                    "x": x[i].detach().cpu().numpy(),
                    "y": y[i].detach().cpu().numpy(),
                    "pred": pred[i].detach().cpu().numpy(),
                    "metadata": metadata[i].detach().cpu().numpy(),
                }
            )
            global_index += 1

    avg = {}
    for key in all_batch_metrics[0].keys():
        avg[key] = float(np.mean([m[key] for m in all_batch_metrics]))

    sample_records = sorted(sample_records, key=lambda r: r["mse"])

    best = sample_records[0]
    median = sample_records[len(sample_records) // 2]
    worst = sample_records[-1]

    save_reconstruction_figure(
        best["x"],
        best["y"],
        best["pred"],
        model_dir / "best_reconstruction.png",
        f"{model_name} Best Reconstruction | MSE={best['mse']:.6f}",
    )

    save_reconstruction_figure(
        median["x"],
        median["y"],
        median["pred"],
        model_dir / "median_reconstruction.png",
        f"{model_name} Median Reconstruction | MSE={median['mse']:.6f}",
    )

    save_reconstruction_figure(
        worst["x"],
        worst["y"],
        worst["pred"],
        model_dir / "worst_reconstruction.png",
        f"{model_name} Worst Reconstruction | MSE={worst['mse']:.6f}",
    )

    save_reconstruction_metadata_json(
        best,
        model_name,
        "best",
        model_dir / "best_reconstruction_metadata.json",
    )

    save_reconstruction_metadata_json(
        median,
        model_name,
        "median",
        model_dir / "median_reconstruction_metadata.json",
    )

    save_reconstruction_metadata_json(
        worst,
        model_name,
        "worst",
        model_dir / "worst_reconstruction_metadata.json",
    )

    shared_pred = None

    if shared_batch is not None:
        x0 = shared_batch["x"].to(cfg.device)
        m0 = shared_batch["metadata"].to(cfg.device)

        p0 = sample_diffusion(model, schedule, x0, m0)

        shared_pred = p0[0].detach().cpu().numpy()

    return avg, shared_pred


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics_csv(metrics, path):
    keys = [
        "model",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "mse_mx",
        "mse_my",
        "mse_mz",
        "mse_total",
        "mean_error_norm",
        "train_time_sec",
        "num_parameters",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in metrics:
            writer.writerow(row)


def save_metrics_json(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_config_json(cfg, path):
    data = {
        "num_samples": cfg.num_samples,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "train_fraction": cfg.train_fraction,
        "val_fraction": cfg.val_fraction,
        "seed": cfg.seed,
        "metadata_dim": cfg.metadata_dim,
        "train_cnn": cfg.train_cnn,
        "train_unet": cfg.train_unet,
        "train_transformer": cfg.train_transformer,
        "train_fno": cfg.train_fno,
        "train_neural_ode": cfg.train_neural_ode,
        "train_diffusion": cfg.train_diffusion,
        "diffusion_epochs": cfg.diffusion_epochs,
        "diffusion_steps": cfg.diffusion_steps,
        "device": cfg.device,
        "out_dir": str(cfg.out_dir),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CFG.seed)

    CFG.out_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", CFG.device)
    print("Saving everything to:")
    print(CFG.out_dir)

    save_config_json(CFG, CFG.out_dir / "run_config.json")

    print("\nGenerating angle-conditioned synthetic dataset...")

    X, Y, metadata, raw_metadata, sensitivities = generate_dataset(
        num_samples=CFG.num_samples,
        nx=CFG.nx,
        ny=CFG.ny,
    )

    print("\nDataset shapes:")
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("metadata:", metadata.shape)
    print("raw_metadata:", raw_metadata.shape)
    print("sensitivities:", sensitivities.shape)

    np.savez_compressed(
        CFG.out_dir / "angle_conditioned_scattering_dataset_v2.npz",
        X_data=X,
        Y_data=Y,
        metadata=metadata,
        raw_metadata=raw_metadata,
        sensitivities=sensitivities,
    )

    np.save(CFG.out_dir / "metadata.npy", metadata)
    np.save(CFG.out_dir / "raw_metadata.npy", raw_metadata)
    np.save(CFG.out_dir / "sensitivities.npy", sensitivities)

    save_dataset_preview(
        X,
        Y,
        CFG.out_dir / "dataset_preview.png",
    )

    save_angle_histograms(
        raw_metadata,
        CFG.out_dir / "angle_histograms.png",
    )

    save_sensitivity_histograms(
        sensitivities,
        CFG.out_dir / "sensitivity_histograms.png",
    )

    dataset = ScatteringDataset(X, Y, metadata)

    n_total = len(dataset)
    n_train = int(CFG.train_fraction * n_total)
    n_val = int(CFG.val_fraction * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=CFG.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CFG.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=CFG.batch_size,
        shuffle=False,
    )

    print("\nSplit:")
    print("Train:", n_train)
    print("Val:", n_val)
    print("Test:", n_test)

    # Shared test example used for all-model side-by-side figure.
    first_test_x, first_test_meta, first_test_y = test_set[0]

    shared_batch = {
        "x": first_test_x.unsqueeze(0),
        "metadata": first_test_meta.unsqueeze(0),
        "y": first_test_y.unsqueeze(0),
    }

    save_truth_and_scattering(
        first_test_x.numpy(),
        first_test_y.numpy(),
        CFG.out_dir / "test_field_truth.png",
        CFG.out_dir / "test_scattering_pattern.png",
    )

    models = []

    if CFG.train_cnn:
        models.append(
            (
                "cnn_angle_conditioned",
                ConditionedCNN(metadata_dim=CFG.metadata_dim),
            )
        )

    if CFG.train_unet:
        models.append(
            (
                "unet_angle_conditioned",
                ConditionedUNet(metadata_dim=CFG.metadata_dim),
            )
        )

    if CFG.train_transformer:
        models.append(
            (
                "transformer_self_attention_angle_conditioned",
                ConditionedPatchTransformer(
                    img_size=CFG.nx,
                    metadata_dim=CFG.metadata_dim,
                ),
            )
        )

    if CFG.train_fno:
        models.append(
            (
                "fno_angle_conditioned",
                ConditionedFNO2d(metadata_dim=CFG.metadata_dim),
            )
        )

    if CFG.train_neural_ode:
        models.append(
            (
                "neural_ode_angle_conditioned",
                ConditionedNeuralODEStyleNet(metadata_dim=CFG.metadata_dim),
            )
        )

    all_metrics = []
    shared_predictions = {}

    for model_name, model in models:
        trained_model, train_info = train_standard_model(
            model,
            model_name,
            train_loader,
            val_loader,
            CFG,
        )

        test_info, shared_pred = evaluate_standard_model(
            trained_model,
            model_name,
            test_loader,
            CFG,
            shared_batch=shared_batch,
        )

        row = {}
        row.update(train_info)
        row.update(test_info)

        all_metrics.append(row)

        if shared_pred is not None:
            shared_predictions[model_name] = shared_pred

        print("\nTest metrics:")
        for k, v in test_info.items():
            print(f"{k}: {v:.6f}")

    if CFG.train_diffusion:
        model_name = "conditional_diffusion_angle_conditioned"

        diffusion_model = ConditionedDiffusionUNet(metadata_dim=CFG.metadata_dim)

        trained_diffusion, schedule, train_info = train_diffusion_model(
            diffusion_model,
            model_name,
            train_loader,
            val_loader,
            CFG,
        )

        test_info, shared_pred = evaluate_diffusion_model(
            trained_diffusion,
            schedule,
            model_name,
            test_loader,
            CFG,
            shared_batch=shared_batch,
        )

        row = {}
        row.update(train_info)
        row.update(test_info)

        all_metrics.append(row)

        if shared_pred is not None:
            shared_predictions[model_name] = shared_pred

    save_metrics_csv(
        all_metrics,
        CFG.out_dir / "metrics_summary.csv",
    )

    save_metrics_json(
        all_metrics,
        CFG.out_dir / "metrics_summary.json",
    )

    save_model_barplot(
        all_metrics,
        CFG.out_dir / "model_comparison_mse.png",
    )

    save_training_time_barplot(
        all_metrics,
        CFG.out_dir / "training_time_comparison.png",
    )

    save_parameter_count_barplot(
        all_metrics,
        CFG.out_dir / "model_size_comparison.png",
    )

    if len(shared_predictions) > 0:
        shared_example_np = {
            "x": first_test_x.numpy(),
            "y": first_test_y.numpy(),
        }

        save_all_models_side_by_side(
            shared_example_np,
            shared_predictions,
            CFG.out_dir / "all_models_comparison_all_components.png",
        )

    print("\nDone.")
    print("Everything saved to:")
    print(CFG.out_dir.resolve())


if __name__ == "__main__":
    main()
