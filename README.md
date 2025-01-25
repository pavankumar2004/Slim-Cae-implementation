# Compression Autoencoder

This project implements a **Compression Autoencoder** using PyTorch to compress and reconstruct images. The autoencoder reduces image size using a convolutional encoder-decoder architecture while maintaining reconstruction quality. 

## Features
- **Image Compression**: Encodes images into a compact latent representation.
- **Quantization**: Simulates compression during training with optional noise addition.
- **GPU Support**: Leverages CUDA for efficient training and inference.
- **Customizable Architecture**: Adjustable latent dimensions and image sizes for resource-limited environments.
- **Training and Testing Pipelines**: Includes easy-to-use functions for model training, testing, and evaluation.

---

## Requirements

### Dependencies
- Python 3.8+
- PyTorch 1.10+
- torchvision
- Pillow
- matplotlib

Install the dependencies using:
```bash
pip install torch torchvision pillow matplotlib
