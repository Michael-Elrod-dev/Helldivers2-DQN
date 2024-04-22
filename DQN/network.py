import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQN.model import Model
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(time.time_ns())

class Network():
    def __init__(self, args):
        self.args = args
        self.num_labels = args.num_labels
        self.seed = random.seed(args.seed)
        self.BUFFER_SIZE = args.BUFFER_SIZE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.TAU = args.TAU
        self.LR = args.LR
        self.UPDATE_EVERY = args.UPDATE_EVERY
        self.priority_replay = args.priority_replay
        self.prio_e, self.prio_a, self.prio_b = args.priority_replay

        # Q-Network
        self.network_local = Model(args.num_labels).to(device)
        self.network_target = Model(args.num_labels).to(device)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr=self.LR)

        # Replay memory
        self.memory = ReplayBuffer(args.num_labels, args.BUFFER_SIZE, args.BATCH_SIZE, args.seed, self.priority_replay)
        self.t_step = 0
    
    def step(self, image, label, reward):
        # Save experience in replay memory
        self.memory.add(image, label, reward)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            if experiences:  # Ensure experiences were actually retrieved
                loss, accuracy, magnitude, lr = self.learn(experiences)
                return loss, accuracy, magnitude, lr
        return None, None, None, None
   
    def predict(self, image, eps=0):
        image = image.unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            outputs = self.network_local(image)
        self.network_local.train()

        # Epsilon-greedy label selection
        if random.random() > eps:
            _, predicted_label = torch.max(outputs, 1)
            predicted_label = predicted_label.item()
        else:
            predicted_label = random.choice(range(self.num_labels))      
        return predicted_label

    def learn(self, experiences):
        experiences, experience_indexes, priorities = experiences
        importance_weights = (len(self.memory) * priorities) ** (-self.prio_b)
        importance_weights /= importance_weights.max()
        importance_weights = importance_weights.unsqueeze(1).to(device)

        images, labels, rewards = experiences

        # Forward pass
        outputs = self.network_local(images)

        # Compute loss
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss = criterion(outputs, labels.float())

        # Reshape importance_weights to match the dimensions of loss
        importance_weights = importance_weights.expand_as(loss)
        loss = torch.mean(loss * importance_weights)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate priorities and update them
        target_priorities = loss.detach().cpu().numpy()
        if target_priorities.ndim == 0:
            target_priorities = np.array([target_priorities])
        target_priorities += self.prio_e
        self.memory.update_priorities(experience_indexes, target_priorities)

        # Update target network
        self.soft_update(self.network_local, self.network_target, self.TAU)

        # Compute accuracy
        _, predicted_labels = torch.max(outputs, 1)
        accuracy = (predicted_labels == labels).float().mean().item()

        # Compute average gradient magnitude
        avg_grad_magnitude = 0
        for param in self.network_local.parameters():
            if param.grad is not None:
                avg_grad_magnitude += param.grad.abs().mean().item()
        avg_grad_magnitude /= len(list(self.network_local.parameters()))

        return loss.item(), accuracy, avg_grad_magnitude, self.optimizer.param_groups[0]['lr']

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_beta(self, interpolation):
        self.prio_b += (1 - self.prio_b)*interpolation


class ReplayBuffer:
    def __init__(self, num_labels, buffer_size, batch_size, seed, priority_replay):
        self.num_labels = num_labels
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["image", "label", "reward"])
        self.seed = random.seed(seed)
        self.priority_replay = priority_replay
        self.priorities = deque(maxlen=buffer_size)
        self.prio_e, self.prio_a, self.prio_b = priority_replay
    
    def add(self, image, label, reward):
        e = self.experience(image, label, reward)
        self.memory.append(e)
        self.priorities.append(self.prio_e)
        
    def update_priorities(self, priority_indexes, priority_targets):
        for index, priority_index in enumerate(priority_indexes):
            self.priorities[priority_index] = priority_targets[index]
    
    def sample(self):
        adjusted_priorities = np.array(self.priorities)**self.prio_a
        sampling_probabilities = adjusted_priorities/sum(adjusted_priorities)
        experience_indexes = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=sampling_probabilities)
        experiences = [self.memory[index] for index in experience_indexes]
        priorities = torch.from_numpy(np.array([self.priorities[index] for index in experience_indexes])).float().to(device)

        images = torch.stack([e.image for e in experiences if e is not None]).to(device)
        labels = torch.stack([e.label for e in experiences if e is not None]).to(device)
        rewards = torch.tensor([e.reward for e in experiences if e is not None], dtype=torch.float32).to(device)
  
        return (images, labels, rewards), experience_indexes, priorities

    def __len__(self):
        return len(self.memory)
    