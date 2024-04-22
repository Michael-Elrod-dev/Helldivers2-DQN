import os
import torch
import random
import numpy as np
from PIL import Image, ImageOps


def get_random_image(image_dir):
    # Define the directory labels
    labels = ["Left", "Right", "Up", "Down"]
    
    # Pick a random label from the list
    true_label = random.choice(labels)
    true_label_index = labels.index(true_label)
    
    # Build the path to the selected label directory
    label_directory = os.path.join(image_dir, true_label)
    
    # List all PNG files in the selected directory
    png_files = [file for file in os.listdir(label_directory) if file.endswith('.png')]
    
    # If there are no PNG files, return an error message or handle as needed
    if not png_files:
        return "No PNG files found in the selected directory.", true_label_index
    
    # Pick a random PNG file from the list
    random_png = random.choice(png_files)
    
    # Build the full path to the randomly selected PNG file
    file_path = os.path.join(label_directory, random_png)
    
    return file_path, true_label_index

def calculate_eps_decay(eps_start, eps_end, n_steps, eps_percentage):
    # Calculate the rate epsilon should decay
    effective_steps = n_steps * eps_percentage
    decrement_per_step = (eps_start - eps_end) / effective_steps
    return decrement_per_step

def preprocess_image(image_path, target_size):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img = img.convert("L")
        
        # Resize the image to the target size using the LANCZOS resampling method
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert the PIL image to a numpy array
        img_array = np.array(img)
        
        # Convert the numpy array to a tensor, ensure the data type is float for normalization
        img_tensor = torch.from_numpy(img_array).float()
        
        # Normalize the image tensor to 0-1 by dividing by 255
        img_tensor /= 255.0
        
        # Add a channel dimension: (H, W) -> (1, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Now the shape is [1, target_size, target_size]
        return img_tensor