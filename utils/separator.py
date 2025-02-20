import cv2
import numpy as np
from PIL import Image


def crop_area_from_array(image_array):    
    return image_array

def separate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    separated_images = []

    bounding_boxes_with_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > 100 and abs(w - h) < max(w, h) * 0.5:
            cropped_image = image[y:y+h, x:x+w]
            bounding_boxes_with_images.append(((x, y, w, h), cropped_image))

    bounding_boxes_with_images.sort(key=lambda b: b[0][0])
    separated_images = [img for _, img in bounding_boxes_with_images]

    return separated_images

def resize(location, outputpath):
    with Image.open(location) as img:
        img = img.resize((80, 80), Image.ANTIALIAS)
        img.save(outputpath, format='PNG')