# Helldivers 2 V.S. Deep Q-Networks (DQN)

This project implements an image classification system using Deep Q-Networks (DQN) with prioritized experience replay. The goal is to train a convolutional neural network (CNN) to classify images into one of four categories: left, right, up, or down and then convert those classifications into in-game actions.

## Project Structure

The project consists of the following files:
- `main.py`: The main script that orchestrates the training and testing process.
- `model.py`: Defines the CNN architecture used for image classification.
- `network.py`: Implements the DQN algorithm with prioritized experience replay.
- `logger.py`: Handles logging of training and testing metrics using Weights and Biases (wandb).
- `utils.py`: Contains utility functions for image preprocessing and epsilon decay calculation.
- `separator.py`: Contains functions for image processing and separating relevant game elements from screenshots.

## Requirements

To run this project, you need the following dependencies:
- Python 3.12.2
- PyTorch
- NumPy
- OpenCV (cv2)
- Weights and Biases (wandb)
- PyAutoGUI
- Screen Info
- Keyboard (Optional for getting training data)
- Pillow

You can install the required packages using pip:
```
pip install torch numpy opencv-python wandb pyautogui screeninfo pillow
```

## Usage

1. Set up your Weights and Biases account and configure the project in `logger.py`.
2. Prepare your dataset by organizing the images into appropriate directories or modify the `get_random_image` function in `utils.py` to load images from your desired location.
3. Run the `main.py` script to start the training process:
   ```
   python main.py
   ```
4. Monitor the training progress using the Weights and Biases dashboard.
5. After training, the trained model checkpoint will be saved as `checkpoint.pth`.
6. To evaluate the trained model, set `load_policy = True` in `main.py` and run the script again:
   ```
   python main.py
   ```
   The testing process will load the trained model from `checkpoint.pth` and evaluate its performance on a set of test images.

## Model Architecture

The CNN architecture used for image classification consists of the following layers:
- Convolutional layer 1: 3 input channels, 16 output channels, 5x5 kernel, stride 2
- Convolutional layer 2: 16 input channels, 32 output channels, 5x5 kernel, stride 2
- Convolutional layer 3: 32 input channels, 32 output channels, 5x5 kernel, stride 2
- Fully connected layer 1: 800 input units, 256 output units
- Fully connected layer 2: 256 input units, 4 output units (corresponding to the four categories)

## Deep Q-Networks (DQN)

The DQN algorithm is used to train the CNN for image classification. The key components of the DQN implementation include:
- Replay Buffer: Stores the experience tuples (image, label) for training.
- Prioritized Experience Replay: Assigns higher priorities to experiences with larger temporal-difference errors.
- Epsilon-Greedy Exploration: Balances exploration and exploitation during training.
- Target Network: Stabilizes the training process by providing target Q-values.

## Logging and Visualization

The project uses Weights and Biases (wandb) for logging and visualizing the training and testing metrics. The logged metrics include:
- Average Accuracy: The average accuracy over the last 50 batches.
- Average Loss: The average loss for the current batch.
- Average Gradient Magnitude: The average magnitude of gradients for the current batch.
- Learning Rate: The current learning rate.
- Priority Alpha and Beta: The parameters for prioritized experience replay.

## Image Preprocessing

The `separator.py` file contains functions for image processing and separating relevant game elements from screenshots:
- `crop_area`: Detects bright yellow objects in the central region of the image, calculates a bounding box around the two largest yellow objects, and crops the image accordingly.
- `separate`: Separates the individual images within the cropped area based on contour detection and stores them in a list.