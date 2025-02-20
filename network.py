import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0")

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
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate size after convolutions
        # For 25x25 input: after 3 pool layers = 3x3
        conv_output_size = 128 * 3 * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        # Input shape should be (batch, 1, 25, 25)
        # Reshape if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class Network():
    def __init__(self, args):
        self.args = args
        self.qnetwork_local = QNetwork(args.image_size, args.num_labels, args.seed).to(device)
        self.qnetwork_target = QNetwork(args.image_size, args.num_labels, args.seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.LR)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                            patience=5, 
                                                            factor=0.5,
                                                            verbose=True)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if state is not None:  # Only proceed if we have a valid state
            loss = self.criterion(self.qnetwork_local(state), 
                                torch.tensor([action]).to(device))
            
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Soft update target network
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.args.TAU)
            
            return loss.item()
        return 0
        
    def act(self, state):
        """Returns prediction for given state"""
        state = state.to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return np.argmax(action_values.cpu().data.numpy())
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_beta(self, interpolation):
        """Update scheduler based on validation loss"""
        self.scheduler.step(interpolation)