import separator
import screengrab as sg
import cv2
import pyautogui
import keyboard
import random
from main import test_dqn
from args import Args
from logger import Logger

cropped_images = []

def getdata():
    print("Screenshot Taken.")
    sg.take_screenshot()
    cropped_image = separator.crop_area('./screen.png')
    images = separator.separate(cropped_image)
    return images

def on_key_event(event):
    if event.name == 'q' and event.event_type == 'down':
        cropped_images = getdata()
        # Pass the cropped_images to the testing function in the order they were taken
        test_dqn(cropped_images)

def main():
    print("Press 'q' to execute the function. Press 'ESC' to exit.")
    keyboard.hook(on_key_event)
    keyboard.wait('esc')  # Waits until 'esc' is pressed.

if __name__ == "__main__":
    main()