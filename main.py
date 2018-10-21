import sys
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

def main():
    sys.argv.pop(0)
    for image_filename in sys.argv:
        analyzeImage(image_filename)


def analyzeImage(image_filename):

    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([70, 70, 70], dtype = "uint16")
    black_image = cv2.inRange(image, lower_black, upper_black)
    
    ret, contours, hierarchy = cv2.findContours(black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(image, contours, -1, (0,255,0), 1)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        rect_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
        displayImage(rect_image)


def displayImage(image):
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()