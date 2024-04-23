import os
import cv2
import torch
import random
import numpy as np
from PIL import Image


def move_file(current_name, base_new_name, target_directory="learned_policy"):
    # Check if the file exists in the current directory
    if os.path.exists(current_name):
        # Ensure the target directory exists; if not, create it
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Calculate the next file number based on the number of files in the directory
        next_file_number = len(os.listdir(target_directory)) + 1

        # Construct the new file name with the number appended
        new_name = f"{base_new_name}_{next_file_number}.pth"

        # Construct the new file path including the directory
        new_path = os.path.join(target_directory, new_name)

        # Rename and move the file
        os.rename(current_name, new_path)
        return new_path
    
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

def preprocess_image(image, target_size, binary):
    # Convert the OpenCV image to a PIL image
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert image to grayscale
    img = img.convert("L")
    
    # Resize the image to the target size using the LANCZOS resampling method
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert the PIL image to a numpy array
    img_array = np.array(img)
    
    # Apply a threshold to convert the image to binary (black and white) if specified
    if binary:
        img_array = np.where(img_array > 128, 255, 0)
    
    # Convert the numpy array to a tensor, ensure the data type is float for normalization
    img_tensor = torch.from_numpy(img_array).float()
    
    # Normalize the image tensor to 0-1 by dividing by 255
    img_tensor /= 255.0
    
    # Add a channel dimension: (H, W) -> (1, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Now the shape is [1, target_size, target_size]
    return img_tensor
    
def env_step(args, predicted_label, true_label, processed_image):
    # If correct next_state = processed_image
    if predicted_label == true_label:
        return processed_image, args.reward, True
    # else it is all 0's
    else:
        processed_image[:] = 0
        return processed_image, 0, False