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

    # Build the path to the selected label directory
    label_directory = os.path.join(image_dir, true_label)

    # List all PNG files in the selected directory
    png_files = [file for file in os.listdir(label_directory) if file.endswith('.png')]

    # If there are no PNG files, return an error message or handle as needed
    if not png_files:
        return "No PNG files found in the selected directory.", true_label

    # Pick a random PNG file from the list
    random_png = random.choice(png_files)

    # Build the full path to the randomly selected PNG file
    file_path = os.path.join(label_directory, random_png)

    return file_path, true_label

def calculate_eps_decay(eps_start, eps_end, n_steps, eps_percentage):
    # Calculate the rate epsilon should decay
    effective_steps = n_steps * eps_percentage
    decrement_per_step = (eps_start - eps_end) / effective_steps
    return decrement_per_step

def preprocess_image(image_path, target_width, target_height):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img = img.convert("L")

        # Get current dimensions
        width, height = img.size

        # Determine the crop area or padding needed
        if width > target_width or height > target_height:
            # Calculate cropping area
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            img = img.crop((left, top, right, bottom))
        else:
            # Calculate padding needed
            horizontal_padding = (target_width - width) / 2
            vertical_padding = (target_height - height) / 2

            # Padding should be added to left/top and right/bottom equally
            left_padding = right_padding = int(horizontal_padding)
            top_padding = bottom_padding = int(vertical_padding)

            # Adjust for odd numbers by adding the extra padding to the right and bottom
            if horizontal_padding != left_padding + right_padding:
                right_padding += 1
            if vertical_padding != top_padding + bottom_padding:
                bottom_padding += 1

            # Apply padding
            img = ImageOps.expand(img, border=(left_padding, top_padding, right_padding, bottom_padding), fill='black')

        # Resize the image using the LANCZOS resampling method
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Convert the PIL image to a numpy array
        img_array = np.array(img)

        # Convert the numpy array to a tensor, ensure the data type is float for normalization
        img_tensor = torch.from_numpy(img_array).float()

        # Normalize the image tensor to 0-1 by dividing by 255
        img_tensor /= 255.0

        # Add a channel dimension: (H, W) -> (1, H, W)
        img_tensor = img_tensor.unsqueeze(0)  # Now the shape is [1, height, width]

        return img_tensor