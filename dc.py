import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Slimmable Autoencoder Model
class SlimCAE(nn.Module):
    def __init__(self, widths):
        super(SlimCAE, self).__init__()
        # Define encoders and decoders with different widths (slimmable layers)
        self.encoders = nn.ModuleList([self.build_encoder(width) for width in widths])
        self.decoders = nn.ModuleList([self.build_decoder(width) for width in widths])
    
    def build_encoder(self, width):
        return nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    
    def build_decoder(self, width):
        return nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(width, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, width_idx):
        encoded = self.encoders[width_idx](x)
        decoded = self.decoders[width_idx](encoded)
        return decoded


# Loss Functions: Rate (R) and Distortion (D)
def distortion_loss(x, x_hat):
    return nn.MSELoss()(x_hat, x)

def rate_loss(encoded):
    # A proxy for rate. Assume encoded feature should be sparse (minimize magnitude).
    return torch.mean(torch.abs(encoded))

# Training SlimCAE with lambda scheduling
def train_slimcae(slimcae, dataloader, num_epochs, lambdas, widths, optimizers):
    slimcae.train()
    for epoch in range(num_epochs):
        for x, _ in dataloader:
            x = x.cuda()  # Assuming CUDA availability
            for i, width in enumerate(widths):
                optimizer = optimizers[i]
                optimizer.zero_grad()

                # Forward pass with a specific subCAE
                x_hat = slimcae(x, width_idx=i)
                encoded = slimcae.encoders[i](x)

                # Rate-Distortion loss
                D = distortion_loss(x, x_hat)
                R = rate_loss(encoded)
                loss = D + lambdas[i] * R

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Lambda Scheduling Algorithm
def lambda_scheduling(slimcae, dataloader_train, dataloader_val, widths, lambdas, κ=1.1, T=2000, M=5):
    # Initialize optimizers for each width
    optimizers = [optim.Adam(slimcae.parameters(), lr=1e-4) for _ in widths]

    # Naive training with the largest lambda
    train_slimcae(slimcae, dataloader_train, num_epochs=5, lambdas=lambdas, widths=widths, optimizers=optimizers)

    # Begin λ-scheduling
    for i in range(len(widths) - 1, 0, -1):
        for _ in range(M):
            # Adjust lambda for the current subCAE
            lambdas[i] *= κ

            # Train with the updated λ
            train_slimcae(slimcae, dataloader_train, num_epochs=5, lambdas=lambdas, widths=widths, optimizers=optimizers)

            # Evaluate Rate and Distortion on validation set
            R_i, D_i = calculate_rate_distortion(slimcae, dataloader_val, width_idx=i)
            R_next, D_next = calculate_rate_distortion(slimcae, dataloader_val, width_idx=i + 1)

            slope = (D_next - D_i) / (R_next - R_i)
            if slope >= 0:  # If slope is no longer improving, break
                break

    return slimcae

# Rate-Distortion Calculation
def calculate_rate_distortion(slimcae, dataloader, width_idx):
    slimcae.eval()
    total_rate = 0
    total_distortion = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.cuda()
            x_hat = slimcae(x, width_idx)
            encoded = slimcae.encoders[width_idx](x)

            # Calculate rate and distortion
            total_distortion += distortion_loss(x, x_hat).item()
            total_rate += rate_loss(encoded).item()

    # Average over the dataset
    avg_rate = total_rate / len(dataloader)
    avg_distortion = total_distortion / len(dataloader)

    return avg_rate, avg_distortion


# Example Usage
if __name__ == "__main__":
    # Create Dataset and Dataloaders (Use CIFAR10 as an example)
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the SlimCAE model with different widths (channel numbers)
    widths = [48, 72, 96, 144, 192]
    slimcae = SlimCAE(widths).cuda()

    # Initial λ values (same for all sub-networks at the beginning)
    lambdas = [0.01 for _ in widths]

    # Train SlimCAE with lambda scheduling
    slimcae_trained = lambda_scheduling(slimcae, dataloader_train, dataloader_val, widths, lambdas)
