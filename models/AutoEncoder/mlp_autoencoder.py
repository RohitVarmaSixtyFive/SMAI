import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128, 64], optimizer_type='adam'):
        super(MLPAutoEncoder, self).__init__()
        
        self.input_size = input_size
        self.optimizer_type = optimizer_type
        
        # Create encoder layers
        encoder_layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ])
            current_size = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder layers
        decoder_layers = []
        hidden_sizes_reversed = hidden_sizes[::-1]
        
        for i in range(len(hidden_sizes_reversed)):
            current_in_size = hidden_sizes_reversed[i]
            current_out_size = input_size if i == len(hidden_sizes_reversed)-1 else hidden_sizes_reversed[i+1]
            
            if i == len(hidden_sizes_reversed)-1:
                decoder_layers.extend([
                    nn.Linear(current_in_size, current_out_size),
                    nn.Sigmoid()
                ])
            else:
                decoder_layers.extend([
                    nn.Linear(current_in_size, current_out_size),
                    nn.BatchNorm1d(current_out_size),
                    nn.ReLU()
                ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(batch_size, 1, 28, 28)  # Reshape back to image
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
