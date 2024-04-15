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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    