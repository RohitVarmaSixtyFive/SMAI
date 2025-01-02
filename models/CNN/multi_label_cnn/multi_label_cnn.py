import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, hamming_loss
import numpy as np

class Multi_CNN(nn.Module):
    def __init__(self, learning_rate=0.001, dropout_rate=0.2, num_conv_layers=3, optimizer_name='adam', num_classes=33):
        super(Multi_CNN, self).__init__()

        self.num_classes = num_classes  
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_conv_layers = num_conv_layers
        self.optimizer_name = optimizer_name 
        self.current_size = 128

        layers = []
        in_channels = 1  
        out_channels = 33

        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate)
            ]
            in_channels = out_channels
            out_channels *= 2
            self.current_size = self.current_size // 2

        self.conv = nn.Sequential(*layers)

        self.flat_features = in_channels * (self.current_size ** 2)

        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, self.num_classes) 
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

    def compile_model(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Optimizer not supported")

    def train_model(self, train_loader, val_loader, num_epochs=20, device='cpu'):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = self.compile_model()
        self.to(device)

        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # Validation phase
            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).float()
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Final metrics after all epochs
        final_metrics = self.evaluate_metrics(val_loader, device)
        return final_metrics

    def evaluate_metrics(self, data_loader, device):
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = self(images)

                # Apply sigmoid and threshold
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).int()  # Threshold at 0.5
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate Hamming score
        hamming = 1 - hamming_loss(all_labels, all_preds)  # 1 - hamming_loss to get the score

        # Calculate Exact Match Accuracy (Strict accuracy)
        exact_match_accuracy = accuracy_score(all_labels, all_preds)

        return {
            'hamming_score': hamming,
            'exact_match_accuracy': exact_match_accuracy
        }

# Example so that u can run
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Multi_CNN().to(device)

# # Assuming train_loader and val_loader are defined
# metrics = model.train_model(
#     train_loader=train_loader,
#     val_loader=val_loader,
#     device=device
# )
# print(metrics)
# # Display the final metrics
# print("Final Metrics after training:")
# for metric, value in metrics.items():
#     print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")
