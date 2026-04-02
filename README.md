# Deep Image Restoration from Lossy Compression

## Overview

This project focuses on the restoration of images degraded by lossy compression techniques using deep learning models. The goal is to reconstruct high-quality images from compressed inputs by reducing visual artifacts such as blurring, blocking, and ringing.

The work explores both convolutional and transformer-based architectures, comparing their effectiveness across different compression domains.

---

## Problem Statement

Lossy image compression reduces storage and transmission costs at the expense of visual quality. Common artifacts include:

- Blurring: loss of fine details  
- Blocking: discontinuities between image blocks  
- Ringing: halo artifacts near edges  

The core research question addressed in this project is:

Can deep learning models effectively reconstruct high-quality images from compressed inputs?

---

## Experimental Pipeline

Original Image → Lossy Compression → Restoration Model → Reconstructed Image

---

## Dataset

The project uses the BSD500 (Berkeley Segmentation Dataset):

- 500 high-quality natural images  
- Diverse content: landscapes, objects, textures, faces  
- Standard benchmark in computer vision  

### Preprocessing

- Images resized to 256 × 256  
- Each image generates:
  - Original (target)
  - Fourier-compressed version
  - Wavelet-compressed version

---

## Compression Techniques

### Fourier Compression

- Transform to frequency domain  
- Retain only low-frequency components (25%)  
- Result: blurred images with loss of fine detail  

### Wavelet Compression

- Multi-scale decomposition (Haar wavelet)  
- Discard high-frequency components  
- Result: sharper structure but localized blocking artifacts  

---

## Models

### 1. Convolutional Autoencoder (U-Net Style)

A custom-designed encoder-decoder architecture with skip connections.

Key features:

- 4 encoder + 4 decoder stages  
- Strided convolutions for downsampling  
- Transposed convolutions for upsampling  
- Skip connections  
- Bottleneck latent representation  

Model size: ~2.3M parameters

---

### 2. Restormer (Transformer-Based)

Adapted from:

Zamir et al., Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)

Key features:

- Channel-wise self-attention  
- Transformer blocks (MDTA + GDFN)  
- Residual learning: output = net(x) + x  

Model size: ~26M parameters

---

## Results

Evaluation metrics: **PSNR** (↑ higher is better), **SSIM** (↑ closer to 1 is better), **LPIPS** (↓ lower is better).

| Experiment | PSNR ↑ | ΔPSNR | SSIM ↑ | ΔSSIM | LPIPS ↓ | ΔLPIPS |
|---|---|---|---|---|---|---|
| Baseline (compressed) | 25.60 | — | 0.788 | — | 0.281 | — |
| AE — Both | 26.62 | +1.02 | 0.785 | −0.003 | 0.136 | −0.145 |
| AE — Fourier only | 24.22 | −1.38 | 0.731 | −0.057 | 0.194 | −0.087 |
| AE — Wavelet only | 26.28 | +0.68 | 0.783 | −0.006 | 0.227 | −0.054 |
| **Restormer — Both** | **28.20** | **+2.60** | **0.809** | **+0.021** | **0.142** | **−0.139** |
| Restormer — Fourier only | 25.51 | −0.09 | 0.752 | −0.037 | 0.169 | −0.112 |
| Restormer — Wavelet only | 27.57 | +1.97 | 0.810 | +0.021 | 0.238 | −0.043 |

---

## Acknowledgments

- Restormer adapted from original implementation  
- BSD500 dataset  

---

## Author

Enrico Favale  
University of Ferrara  
