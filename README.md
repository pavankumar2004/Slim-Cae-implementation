# Slimmable Compressive Autoencoders (SlimCAEs)

**Slimmable Compressive Autoencoders (SlimCAEs)** is a neural image compression model that dynamically adjusts the network width to achieve variable compression rates. The model utilizes slimmable Generalized Divisive Normalization (GDN) layers and a λ-scheduling training algorithm to optimize memory usage, computation, and latency while maintaining high compression performance.

## Features

- **Dynamic Network Width**: Adjusts the network's width to achieve different compression rates.
- **Slimmable GDN Layers**: Uses slimmable GDN layers for flexible image compression.
- **λ-Scheduling Algorithm**: Optimizes the model's training for better performance with varying compression levels.
- **Optimized Memory & Computation**: Reduces computational costs and memory usage while preserving compression quality.

## Prerequisites

- Python 3.x
- PyTorch (depending on your implementation)
- CUDA-enabled GPU for training (optional but recommended)

