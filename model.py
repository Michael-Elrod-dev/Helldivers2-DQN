import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, image_size, num_labels, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the size after convolutions and pooling
        self.pool = nn.MaxPool2d(2, 2)
        conv_output_size = 128 * 3 * 3  # This will need to be adjusted based on your input size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_labels)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, image):
        # Reshape the input to add channel dimension if it's not present
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(1)  # Add channel dimension
            
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(image))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x