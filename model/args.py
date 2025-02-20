class Args():
    def __init__(self):
        # Network Hyperparameters
        self.BATCH_SIZE = 32             # Reduced batch size for initial training
        self.LR = 0.0001                 # Reduced learning rate for more stable training
        
        # Model Training Settings
        self.seed = 0                    # Random seed
        self.max_steps = 100000          # Maximum number of training steps
        self.num_labels = 4              # Number of different labels (up, down, left, right)
        
        # Training Parameters
        self.early_stopping_patience = 10  # Number of epochs to wait before early stopping
        self.validation_split = 0.2       # Portion of data to use for validation
        self.min_lr = 1e-6               # Minimum learning rate for scheduler
        self.lr_patience = 5             # Patience for learning rate scheduler
        self.lr_factor = 0.5             # Factor to reduce learning rate by
        
        # Environment and Image Settings
        self.image_dir = 'TestingData/Filtered/'  # Directory containing images
        self.policy_file = 'best_model.pth'       # Where to save the best model
        self.image_w = 25                # Image width after processing
        self.image_h = 25                # Image height after processing
        self.image_size = self.image_w * self.image_h  # Total image size
        
        # Model Architecture Settings
        self.conv_channels = [32, 64, 128]  # Number of channels in each conv layer
        self.fc_units = [256, 128]          # Number of units in fully connected layers
        self.dropout_rate = 0.5             # Dropout rate
        self.use_batch_norm = True          # Whether to use batch normalization
        
        # Miscellaneous Settings
        self.wandb = True                # Enable wandb logging
        self.save_best = True            # Whether to save best model during training
        self.checkpoint_freq = 1000      # How often to save checkpoints
        self.model_save_dir = 'models/'  # Directory to save models