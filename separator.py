import cv2
import numpy as np
from PIL import Image


def crop_area_from_array(image_array):    
    # height, width = image_array.shape[:2]
    # center_x, center_y = width // 2, height // 2
    
    # width_divisor = 2
    # height_divisor = 2
    
    # cropped_image = image_array[
    #     center_y - height // height_divisor:center_y + height // height_divisor, 
    #     center_x - width // width_divisor:center_x + width // width_divisor
    # ]

    # hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    # hsv = cv2.GaussianBlur(hsv, (25, 25), 0)

    # lower_yellow = np.array([20, 10, 160])
    # upper_yellow = np.array([40, 80, 255])

    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # if len(filtered_contours) >= 2:
    #     filtered_contours.sort(key=cv2.contourArea, reverse=True)
    #     top_contours = filtered_contours[:2]

    #     x_coords = []
    #     y_coords = []
    #     for cnt in top_contours:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         x_coords.extend([x, x + w])
    #         y_coords.extend([y, y + h])

    #     bounding_x1, bounding_x2 = min(x_coords), max(x_coords)
    #     bounding_y1, bounding_y2 = min(y_coords), max(y_coords)
        
    #     cropped_to_bounding_box = cropped_image[bounding_y1:bounding_y2, bounding_x1:bounding_x2]
    # else:
    #     print("Not enough yellow contours found")
    #     cropped_to_bounding_box = cropped_image
    
    # return cropped_to_bounding_box
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