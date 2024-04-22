from utils import calculate_eps_decay

class Args():
    def __init__(self):
        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.BATCH_SIZE = 100        # minibatch size
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR = 5e-4               # learning rate 
        self.UPDATE_EVERY = 2        # how often to update the network
    
        self.image_w = 25
        self.image_h = 25
        self.image_size = 25
        
        self.wandb = False
        self.load_policy = True
        self.max_steps = 100000
        self.test_steps = 100
        self.num_labels = 4
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_percentage = 0.98
        self.eps_decay = calculate_eps_decay(self.eps_start, self.eps_end, self.max_steps, self.eps_percentage)
        self.seed = 0
        self.priority_replay = [0.5, 0.5, 0.5]

        self.image_dir = 'ImageProcessing/TestingData/Filtered/'