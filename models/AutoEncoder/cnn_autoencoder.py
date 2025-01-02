import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# AutoEncoder class as provided
class CNN_AutoEncoder(nn.Module):
    def __init__(self, kernel_size=3, num_layers=3, initial_channels=32, optimizer_type='adam'):
        super(CNN_AutoEncoder, self).__init__()
        
        self.padding = (kernel_size - 1) // 2
        self.optimizer_type = optimizer_type
        
        # Encoder layers
        encoder_layers = []
        in_channels = 1
        current_channels = initial_channels
        
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Conv2d(in_channels, current_channels, kernel_size=kernel_size, padding=self.padding),
                nn.BatchNorm2d(current_channels),
                nn.ReLU(),
                nn.Conv2d(current_channels, current_channels, kernel_size=kernel_size, padding=self.padding),
                nn.BatchNorm2d(current_channels),
                nn.ReLU()
            ])
            if i < num_layers - 1:
                encoder_layers.append(nn.MaxPool2d(2, 2))
                in_channels = current_channels
                current_channels *= 2
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        for i in range(num_layers - 1, -1, -1):
            current_out_channels = initial_channels * (2 ** max(i-1, 0))
            current_in_channels = initial_channels * (2 ** i)
            
            if i == num_layers - 1:
                current_in_channels = current_channels
            
            decoder_layers.extend([
                nn.ConvTranspose2d(current_in_channels, current_out_channels, kernel_size=kernel_size, padding=self.padding),
                nn.BatchNorm2d(current_out_channels),
                nn.ReLU()
            ])
            
            if i > 0:
                decoder_layers.append(nn.Upsample(scale_factor=2))
            
            if i == 0:
                decoder_layers.extend([
                    nn.ConvTranspose2d(current_out_channels, 1, kernel_size=kernel_size, padding=self.padding),
                    nn.Sigmoid()
                ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output size check for compatibility with input dimensions
        self.output_size = 28 // (2 ** (num_layers - 1))
        assert self.output_size * (2 ** (num_layers - 1)) == 28, \
            f"Number of layers ({num_layers}) will not maintain 28x28 dimensions"
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def train_model(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        device = next(self.parameters()).device
        criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5) \
                    if self.optimizer_type == 'adam' else optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        train_losses, val_losses = [], []
        
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            
            for data, _ in train_loader:
                data = data.to(device)
                data = data / 255.0 if data.max() > 1 else data
                
                reconstructed, _ = self(data)
                loss = criterion(reconstructed, data)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    data = data / 255.0 if data.max() > 1 else data
                    reconstructed, _ = self(data)
                    val_loss += criterion(reconstructed, data).item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            scheduler.step(avg_val_loss)
        
        return train_losses, val_losses

def visualize_reconstructions(model, test_loader, num_images=5):
    model.eval()
    device = next(model.parameters()).device
    
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    images = images / 255.0 if images.max() > 1 else images
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    fig.suptitle('Original vs Reconstructed Images', fontsize=16)
    
    for i in range(num_images):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i+1}')
        
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Reconstructed {i+1}')
    
    plt.tight_layout()
    plt.show()


