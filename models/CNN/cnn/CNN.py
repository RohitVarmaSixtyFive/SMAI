import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, task, num_classes, learning_rate, dropout_rate, num_conv_layers, optimizer_name):
        super(CNN, self).__init__()
        self.task = task
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_conv_layers = num_conv_layers
        self.optimizer_name = optimizer_name
        self.current_size = 128
        
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        out_channels = 32
        
        for _ in range(num_conv_layers):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.conv_blocks.append(block)

            in_channels = out_channels
            out_channels *= 2
            self.current_size //= 2

        self.flat_features = in_channels * (self.current_size ** 2)

        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_feature_maps=False):
        feature_maps = []
        
        for block in self.conv_blocks:
            x = block(x)
            if return_feature_maps:
                feature_maps.append(x.detach().clone())
        
        x = x.view(x.size(0), -1)  
        output = self.fc(x)
        
        if return_feature_maps:
            return output, feature_maps
        return output

    def evaluate_metrics(self, data_loader, device):
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                
                # Round predictions to nearest integer for classification
                rounded_preds = torch.round(outputs).clamp(0, 9).long()  # Clamp between 0-9 for Fashion MNIST
                
                all_preds.extend(rounded_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate classification accuracy using rounded predictions
        accuracy = accuracy_score(all_labels, np.array(all_preds).flatten())
        
        return {
            'accuracy': accuracy,
        }

    def train_model(self, train_loader, val_loader, test_loader, num_epochs, device):
        criterion = nn.MSELoss()
        optimizer = self.compile_model(optimizer_name=self.optimizer_name)
        self.to(device)

        train_losses, val_losses = [], []
        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)
                
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).float().view(-1, 1)
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)

            train_metrics = self.evaluate_metrics(train_loader, device)
            val_metrics = self.evaluate_metrics(val_loader, device)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_metrics["accuracy"]:.4f}')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_metrics["accuracy"]:.4f}')
            print('-' * 50)

            # Log training and validation metrics to WandB
            # wandb.log({
            #     "train_loss": avg_train_loss,
            #     "val_loss": avg_val_loss,
            #     "train_accuracy": train_metrics["accuracy"],
            #     "val_accuracy": val_metrics["accuracy"],
            #     "epoch": epoch + 1,
            # })
            
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']

        # Final evaluation on test set
        final_train_metrics = self.evaluate_metrics(train_loader, device)
        final_val_metrics = self.evaluate_metrics(val_loader, device)
        final_test_metrics = self.evaluate_metrics(test_loader, device)

        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'final_train_accuracy': final_train_metrics['accuracy'],
            'final_val_accuracy': final_val_metrics['accuracy'],
            'final_test_accuracy': final_test_metrics['accuracy']
        }

    def compile_model(self, optimizer_name):
        if optimizer_name == 'adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.parameters(), lr=self.learning_rate)