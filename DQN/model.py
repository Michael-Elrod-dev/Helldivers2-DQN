import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_labels):
        '''
        in_channels: The number of input channels (depth) of the input tensor, RGB?
        out_channels: The number of output channels (depth) produced by the convolutional layer
        kernel_size: The size of the convolutional filter (kernel) that will be applied to the input
        stride: The step size at which the convolutional filter moves over the input
        '''
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x