import separator
import screengrab as sg
import cv2
import pyautogui
import random
import keyboard

def getdata():
    sg.take_screenshot()
    cropped_image = separator.crop_area('./screen.png')
    images =separator.separate(cropped_image)
    for idx, img in enumerate(images):
        cv2.imwrite(f'./TestingData/{random.randint(1, 9999999)}/item-{idx+1}.png', img)

def on_key_event(event):
    if event.name == 'q' and event.event_type == 'down':
        getdata()

def main():
    print("Press 'q' to execute the function. Press 'ESC' to exit.")
    keyboard.hook(on_key_event)
    keyboard.wait('esc')  # Waits until 'esc' is pressed.
    

if __name__ == "__main__":
    main()
