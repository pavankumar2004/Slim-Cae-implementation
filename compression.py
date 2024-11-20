import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import glob

class CompressionAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):  # Reduced from 32 to 16 for RTX 2050
        super().__init__()
        
        # Smaller encoder for RTX 2050
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Reduced from 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Reduced from 128
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1)
        )
        
        # Smaller decoder for RTX 2050
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def quantize(self, x, training=True):
        if training:
            noise = torch.rand_like(x) - 0.5
            x = x + noise
        else:
            x = torch.round(x)
        return x
        
    def forward(self, x, training=True):
        latent = self.encoder(x)
        quantized = self.quantize(latent, training)
        reconstructed = self.decoder(quantized)
        return reconstructed, quantized

class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # Support multiple image formats
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(glob.glob(os.path.join(root_dir, ext)))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def setup_training(image_folder, batch_size=8):  # Reduced batch size for RTX 2050
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Reduced image size
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    dataset = ImageFolder(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader

def train_model(model, train_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    
    # Training history
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed, quantized = model(data, training=True)
            
            # Compute losses
            distortion_loss = mse_loss(reconstructed, data)
            rate_loss = torch.mean(torch.abs(quantized)) * 0.1
            
            loss = distortion_loss + rate_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # More frequent updates
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
                
            # Save example reconstructions periodically
            if batch_idx % 50 == 0:
                if epoch == num_epochs - 1:  # Check if it's the last epoch
                    save_example_reconstruction(data[0], data[0],  # Save the original image
                                                f'output/reconstruction_epoch{epoch}_batch{batch_idx}.png')
                else:
                    save_example_reconstruction(data[0], reconstructed[0], 
                                                f'output/reconstruction_epoch{epoch}_batch{batch_idx}.png')
        
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f'Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}')  # Print epoch and average loss
    
    return history

def save_example_reconstruction(original, reconstructed, filename):
    """Save original and reconstructed images side by side during training"""
    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(original.cpu().detach().permute(1, 2, 0).clamp(0, 1))
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.cpu().detach().permute(1, 2, 0).clamp(0, 1))
    plt.title('Reconstructed')
    plt.axis('off')
    
    plt.savefig(filename)
    plt.close()

def test_compression(model, test_image_path, device='cuda'):
    """Test the compression model on a single image"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Load and transform image
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # Compress and reconstruct
        reconstructed, _ = model(image_tensor, training=False)
        
        # Save the reconstructed image only
        result = reconstructed.squeeze(0)
        save_reconstructed_image(result, 'compression_test_result.png')

def save_reconstructed_image(reconstructed, filename):
    """Save only the reconstructed image during testing"""
    plt.figure(figsize=(5, 5))
    
    # Reconstructed
    plt.imshow(reconstructed.cpu().detach().permute(1, 2, 0).clamp(0, 1))
    plt.title('Reconstructed')
    plt.axis('off')
    
    plt.savefig(filename)
    plt.close()

def calculate_compression_rate(model, dataloader, device='cuda'):
    """Calculate and return the compression rate"""
    model.eval()
    total_input_size = 0
    total_latent_size = 0
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            total_input_size += images.numel()  # Total number of elements in the input images
            
            # Get the latent representation
            latent = model.encoder(images)
            total_latent_size += latent.numel()  # Total number of elements in the latent representation
            
    # Calculate compression rate
    compression_rate = total_input_size / total_latent_size
    return compression_rate

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"CUDA version in PyTorch: {torch.version.cuda}")

    # Check CUDA availability
    print(torch.cuda.is_available())  # Should return True
    print(torch.version.cuda)  # Should return the CUDA version
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Initialize model
    model = CompressionAutoencoder().to(device)
    
    # Training
    train_folder = "train_images"  # Update this path
    train_loader = setup_training(train_folder)
    
    print("Starting training...")
    history = train_model(model, train_loader, num_epochs=5, device=device)
    
    # Save the trained model
    torch.save(model.state_dict(), 'output/compression_model.pth')
    print("Model saved successfully!")
    
    # Calculate and print compression rate after training
    compression_rate = calculate_compression_rate(model, train_loader, device=device)
    print(f"Compression Rate after training: {compression_rate:.2f}:1")

    # Test the model
    test_image_path = "test_images/pic011.png"  # Update this path
    test_compression(model, test_image_path, device)

if __name__ == "__main__":
    main()
