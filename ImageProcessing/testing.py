import separator
import cv2

def main():
    cropped_image = separator.crop_area('./test-images/source3.png')
    separator.separate(cropped_image)

if __name__ == "__main__":
    main()
