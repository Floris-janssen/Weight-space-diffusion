# 3D Weight-Space Diffusion of FINER Implicit Neural Representations

A production-ready pipeline for learning, quantizing, and generating 3D Signed Distance Functions (SDFs) directly in neural-network weight space. 

This repository replaces standard ReLU/WIRE networks with **FINER** (Flexible Spectral-bias Tuning) and utilizes **Multiplicative MAML-LoRA** to solve weight-space permutation symmetries. The extracted low-rank adapter vectors are compressed via a VQ-VAE and generated using a 1D Latent Diffusion Transformer (DiT), before being extracted into watertight meshes via FlexiCubes.

## Pipeline Overview

| Stage | Script | Description |
| :--- | :--- | :--- |
| **0. Preprocess** | `scripts/preprocess_mesh.py` | Voxelizes raw meshes to watertight solids, normalizes, and extracts SDF point clouds. |
| **1. Meta-Prior** | `scripts/train_maml_base.py` | Meta-learns a shared FINER-SDF base weight via curvature-guided MAML. |
| **2. Adapters** | `scripts/fit_adapters.py` | Freezes the base prior and extracts Rank-16 multiplicative LoRA adapters per shape. |
| **3. VQ-VAE** | `scripts/train_vqvae.py` | Normalizes and quantizes the 65.6k-D adapters into discrete 128-D codebook tokens. |
| **4. Latent DiT** | `scripts/train_dit.py` | Trains a 1D Diffusion Transformer to denoise the discrete latent tokens. |
| **5. Generation** | `scripts/generate_meshes.py` | Samples novel latents, decodes weights, and extracts 3D meshes using FlexiCubes. |

## Installation

Requires Python 3.10+ and a CUDA-capable GPU (tested on NVIDIA T4).

```bash
git clone [https://github.com/Floris-janssen/Weight-space-diffusion.git)
cd Weight-space-diffusion
pip install -r requirements.txt