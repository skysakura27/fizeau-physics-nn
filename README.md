# Fizeau Physics-Informed Neural Network

A PyTorch implementation of a **Physics-Informed Neural Network (PINN)** for phase retrieval from **Fizeau interferometry** data.

## 🎯 Overview

This project combines deep learning with optical physics to reconstruct wavefront aberrations from interferometric measurements. The network architecture integrates:

- **Algorithm Unrolling**: Unfolds iterative phase retrieval into learnable layers
- **Zernike Polynomials**: Aberration parameterization using orthonormal basis
- **Wavelet Preprocessing**: 2D DWT for multi-scale feature extraction
- **Residual Denoising**: Skip connections for robust phase reconstruction
- **Physics Constraints**: Airy diffraction model integrated into loss function

## 📁 Repository Structure
