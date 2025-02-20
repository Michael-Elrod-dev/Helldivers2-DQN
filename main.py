import utils.screengrab as sg
import keyboard
from model.train import test, train_classifier
from model.args import Args
from utils.logger import Logger
import numpy as np
import os
import cv2
from datetime import datetime

def getdata():
    print("Screenshot Taken.")
    screenshot = sg.take_screenshot(
        region_width=400,
        region_height=100,
        x_offset=-10,
        y_offset=0
    )
    
    image_array = np.array(screenshot)
    cropped_image = separator.crop_area_from_array(image_array)
    images = separator.separate(cropped_image)
    return images

def save_images_for_sorting(images):
    base_dir = 'ImageProcessing/UnsortedData'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for i, image in enumerate(images):
        filename = f"{timestamp}_{i}.png"
        filepath = os.path.join(base_dir, filename)
        cv2.imwrite(filepath, image)
    
    print(f"Saved {len(images)} images to {base_dir}")

def on_key_event_test(event):
    if event.name == 'q' and event.event_type == 'down':
        cropped_images = getdata()
        test(cropped_images)

def on_key_event_collect(event):
    if event.name == 'q' and event.event_type == 'down':
        cropped_images = getdata()
        save_images_for_sorting(cropped_images)
    elif event.name == 'esc' and event.event_type == 'down':
        print("Exiting data collection mode...")
        keyboard.unhook_all()

def train_model():
    args = Args()
    # Enable wandb logging
    args.wandb = True
    logger = Logger(args) if args.wandb else None
    
    print("Starting training...")
    scores = train_classifier(args, logger)
    
    if logger:
        logger.close()
    
    print("Training completed!")
    return scores

def main():
    mode = input("Enter mode (1 for test, 2 for data collection, 3 for training): ")
    
    if mode == "1":
        print("Test Mode: Press 'q' to test images. Press 'ESC' to exit.")
        keyboard.hook(on_key_event_test)
        keyboard.wait('esc')
    elif mode == "2":
        print("Data Collection Mode: Press 'q' to capture and save images.")
        print("Press 'ESC' to exit collection mode.")
        keyboard.hook(on_key_event_collect)
        keyboard.wait('esc')
    elif mode == "3":
        print("Training Mode: Starting model training...")
        train_model()
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()