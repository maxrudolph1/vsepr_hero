import sys
import cv2
import copy
import itertools
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def main():
    sys.argv.pop(0)
    for image_filename in sys.argv:
        analyzeImage(image_filename)

def analyzeImage(image_filename):
    image = cv2.imread(image_filename)

    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([70, 70, 70], dtype = "uint16")
    black_image = cv2.inRange(image, lower_black, upper_black)
    
    ret, contours, hierarchy = cv2.findContours(black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(image, contours, -1, (0,255,0), 1)
    bounding_boxes = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w > 20 and h > 20:
            bounding_boxes.append((x, y, w, h))
            # rect_image = cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
            # displayImage(rect_image)

    findBonds(image, bounding_boxes)

def findBonds(image, bounding_boxes):
    bonds = []
    box_centers = [(x[0] + x[2]/2, x[1] + x[3]/2) for x in bounding_boxes]
    combos = list(itertools.combinations(box_centers, 3))
    for combo in combos:
        x_coors = [x[0] for x in combo]
        y_coors = [x[1] for x in combo]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_coors, y_coors)
        if abs(r_value) > 0.97:
            bonds.append(combo)
            drawRelation(image, combo, slope, intercept, r_value, p_value, std_err)
    

def drawRelation(image, combo, slope, intercept, r_value, p_value, std_err):
    point_image = copy.deepcopy(image)
    for point in combo:
        point_image = cv2.circle(point_image, (int(round(point[0])), int(round(point[1]))), 20, (255, 0, 0), 20)

    top_x = int(round((point_image.shape[0] - intercept) / slope))
    bottom_x = int(round((0 - intercept) / slope))
    point_image = cv2.line(point_image, (bottom_x, 0), (top_x, point_image.shape[0]), (255,0,0), 5)
    print(f"r_value: {r_value}, p_value: {p_value}, std_err: {std_err}")
    displayImage(point_image)

def displayImage(image):
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()