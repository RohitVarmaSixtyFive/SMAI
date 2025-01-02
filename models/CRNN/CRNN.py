import torch
from torch import nn
from torch.nn import functional as F

class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_chars):
        super(CRNN, self).__init__()
        # Initial input: [batch, 1, 64, 256]
        self.conv_1 = nn.Conv2d(img_channel, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        # After conv1 + pool1: [batch, 128, 32, 128]
        
        self.conv_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        # After conv2 + pool2: [batch, 256, 16, 64]
        
        self.conv_3 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool_3 = nn.MaxPool2d(kernel_size=(2, 1))  # Note: pooling only height
        # After conv3 + pool3: [batch, 512, 8, 64]
        
        self.conv_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool_4 = nn.MaxPool2d(kernel_size=(2, 1))  # Note: pooling only height
        # After conv4 + pool4: [batch, 512, 4, 64]
        
        self.linear_1 = nn.Linear(512 * 4, 64)  # 512 channels * 4 height
        self.drop_1 = nn.Dropout(0.2)
        # After linear + dropout: [batch, 64, 64]
        
        self.rnn = nn.RNN(64, 128, bidirectional=True, num_layers=2, dropout=0.2, batch_first=True)
        # After RNN: [batch, 64, 256]
        
        self.output = nn.Linear(256, num_chars)
        # Final output: [batch, sequence_length, num_chars]

    def forward(self, images):
        bs, _, _, _ = images.size()
#         print(f"Input shape: {images.shape}")  # [batch, 1, 64, 256]
        
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
#         print(f"After conv1 + pool1: {x.shape}")  # [batch, 128, 32, 128]
        
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
#         print(f"After conv2 + pool2: {x.shape}")  # [batch, 256, 16, 64]
        
        x = F.relu(self.conv_3(x))
        x = self.pool_3(x)
#         print(f"After conv3 + pool3: {x.shape}")  # [batch, 512, 8, 64]
        
        x = F.relu(self.conv_4(x))
        x = self.pool_4(x)
#         print(f"After conv4 + pool4: {x.shape}")  # [batch, 512, 4, 64]
        
        # Reshape for sequence processing
        x = x.permute(0, 3, 1, 2)  # [batch, 64, 512, 4]
        x = x.view(bs, x.size(1), -1)  # [batch, 64, 2048]
        
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
#         print(f"After linear + dropout: {x.shape}")  # [batch, 64, 64]
        
        x, _ = self.rnn(x)
#         print(f"After RNN: {x.shape}")  # [batch, 64, 256]
        
        x = self.output(x)
#         print(f"Final output: {x.shape}")  # [batch, 64, num_chars]
        
        return x

        