import sys
import cv2
import copy
import itertools
import pytesseract
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def main():
    sys.argv.pop(0)
    for image_filename in sys.argv:
        analyzeImage(image_filename)

def analyzeImage(image_filename):
    raw_image = cv2.imread(image_filename)
    grayscale_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)

    # OTSU disregards thres=0 parameters and computes/returns optimal threshold value
    otsu_thresh, bw_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Grab the black text from the image to form a mask
    lower_black = np.array([0], dtype = "uint16")
    upper_black = np.array([70], dtype = "uint16")
    black_image = cv2.inRange(bw_image, lower_black, upper_black)

    # Find the separate contours for each component, avoiding components within components
    num_contours, contours, hierarchy = cv2.findContours(black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create bounding boxes around the components
    bounding_boxes = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w > 20 and h > 20:
            bounding_boxes.append((x, y, w, h))

    # Identify which of the bounding boxes are letters and annotate them
    letters_bounding_boxes = annotateLetters(bounding_boxes, raw_image)
    
    # Identify which of the bounding boxes are bonds and annotate them
    bonds_bounding_boxes = annotateBonds(bounding_boxes, letters_bounding_boxes, raw_image)

def annotateBonds(bounding_boxes, letters_bounding_boxes, original_image):
    valid_lines = findValidLines(bounding_boxes, original_image)
    letter_box_centers = [(x["bounding_box"][0] + x["bounding_box"][2]/2, x["bounding_box"][1] + x["bounding_box"][3]/2) for x in letters_bounding_boxes]

    annoying_bonds = []
    for valid_line in valid_lines:
        if len(valid_line["data_points"]) >= 6:
            annoying_bonds.append(valid_line)
    for bond in annoying_bonds:
        valid_lines.remove(bond)

    # let's grab the easy bonds: lines that we know contain two letters and a thing between them
    double_letter_bonds = []
    for valid_line in valid_lines:
        matches = []
        for data_point in valid_line["data_points"]:
            if data_point in letter_box_centers:
                matches.append(data_point)
        
        if len(matches) == 2:
            distmatches = ((matches[1][0] - matches[0][0]) ** 2 + (matches[1][1] - matches[0][1]) ** 2) ** 0.5
            if distmatches < 1900:
                # verify that the non-recognized line is in the middle
                remaining_points = list(set(valid_line["data_points"]) - set(matches))
                if len(remaining_points) == 1:
                    remaining_point = remaining_points[0]
                    min_x = min(matches[0][0], matches[1][0]) # left
                    min_y = min(matches[0][1], matches[1][1]) # top
                    width = max(matches[0][0], matches[1][0])-min_x
                    height = max(matches[0][1], matches[1][1])-min_y
                    if remaining_point[0] > min_x and remaining_point[0] < min_x + width and remaining_point[1] > min_y and remaining_point[1] < min_y + height:
                        double_letter_bonds.append(valid_line)
    for bond in double_letter_bonds:
        valid_lines.remove(bond)

    # lets choose a center based on what we have
    common_bonds = []
    for bond in annoying_bonds:
        common_bonds.append(bond)
    for bond in double_letter_bonds:
        common_bonds.append(bond)
    common_denominator = None
    for bond in common_bonds:
        # after this you could even have the annoying_bond's little pieces merged together and run another find_valid_lines algorithm to get rid of the left_overs.

    # for valid_line in valid_lines:
    #     drawRelation(original_image, valid_line["data_points"], valid_line["slope"], valid_line["intercept"], valid_line["r_squared"], 0, 0)
    
    # print("-----")
    # for bond in double_letter_bonds:
    #     drawRelation(original_image, bond["data_points"], bond["slope"], bond["intercept"], bond["r_squared"], 0, 0)

    # print("-----")
    # for bond in annoying_bonds:
    #     drawRelation(original_image, bond["data_points"], bond["slope"], bond["intercept"], bond["r_squared"], 0, 0)

def findValidLines(bounding_boxes, original_image):
    valid_lines = []
    box_centers = [(x[0] + x[2]/2, x[1] + x[3]/2) for x in bounding_boxes]
    combos = list(itertools.combinations(box_centers, 3))
    for combo in combos:
        # remove by points too close together
        dist01 = ((combo[1][0] - combo[0][0]) ** 2 + (combo[1][1] - combo[0][1]) ** 2) ** 0.5
        dist12 = ((combo[2][0] - combo[1][0]) ** 2 + (combo[2][1] - combo[1][1]) ** 2) ** 0.5
        dist02 = ((combo[2][0] - combo[0][0]) ** 2 + (combo[2][1] - combo[0][1]) ** 2) ** 0.5
        if min([dist01, dist02, dist12]) > 300:
            # remove by rsquared value of line
            x_coors = [x[0] for x in combo]
            y_coors = [x[1] for x in combo]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_coors, y_coors)
            if abs(r_value ** 2) > 0.97:
                # remove by lines too similar
                close_match = False
                for valid_line in valid_lines:
                    angle_between = np.arctan2(abs(valid_line["slope"] - slope), abs(1 + valid_line["slope"] * slope))
                    intercept_distance = abs(valid_line["intercept"] - intercept)
                    if angle_between < 0.1 and intercept_distance < 110:
                        close_match = True
                        # let's add data from this matched line to the one already counted
                        valid_line["data_points"] = tuple(set(valid_line["data_points"] + combo))
                if not close_match:
                    valid_lines.append({"data_points": combo, "slope": slope, "intercept": intercept, "r_squared": r_value ** 2})
    
    # print(len(valid_lines))
    # for bond in valid_lines:
    #     drawRelation(original_image, bond["data_points"], bond["slope"], bond["intercept"], bond["r_squared"], 0, 0)
    return valid_lines
    
def drawColinearRelation(image, combo1, slope1, intercept1, combo2, slope2, intercept2):
    point_image = copy.deepcopy(image)
    for point in combo1:
        point_image = cv2.circle(point_image, (int(round(point[0])), int(round(point[1]))), 20, (255, 0, 0), 20)

    for point in combo2:
        point_image = cv2.circle(point_image, (int(round(point[0])), int(round(point[1]))), 20, (0, 0, 255), 20)

    top_x1 = int(round((point_image.shape[0] - intercept1) / slope1))
    bottom_x1 = int(round((0 - intercept1) / slope1))
    point_image = cv2.line(point_image, (bottom_x1, 0), (top_x1, point_image.shape[0]), (255,0,0), 5)

    top_x2 = int(round((point_image.shape[0] - intercept2) / slope2))
    bottom_x2 = int(round((0 - intercept2) / slope2))
    point_image = cv2.line(point_image, (bottom_x2, 0), (top_x2, point_image.shape[0]), (0,0,255), 5)
    
    intercept_distance = abs(intercept1 - intercept2)
    angle_between = np.arctan2(abs(slope1 - slope2), abs(1 + slope1 * slope2))
    print(f"Angle_between: {angle_between}, intercept_distance: {intercept_distance}")
    displayImage(point_image)


def drawRelation(image, combo, slope, intercept, r_value, p_value, std_err):
    point_image = copy.deepcopy(image)
    for point in combo:
        point_image = cv2.circle(point_image, (int(round(point[0])), int(round(point[1]))), 20, (255, 0, 0), 20)

    top_x = int(round((point_image.shape[0] - intercept) / slope))
    bottom_x = int(round((0 - intercept) / slope))
    point_image = cv2.line(point_image, (bottom_x, 0), (top_x, point_image.shape[0]), (255,0,0), 5)
    print(f"r_value: {r_value}, p_value: {p_value}, std_err: {std_err}")
    displayImage(point_image)

def annotateLetters(bounding_boxes, original_image):
    lettered_boxes = []
    for bounding_box in bounding_boxes:
        bounding_box_image = createBoundingBoxImage(bounding_box, original_image)
        parsed_text = pytesseract.image_to_string(bounding_box_image)
        parsed_text = parsed_text.replace(' ', '')
        if len(parsed_text) != 0:
            if parsed_text.isalnum():
                if not parsed_text.startswith('I') and not parsed_text.startswith('l'):
                    lettered_boxes.append({"annotation": parsed_text[0].lower(), "bounding_box": bounding_box})
    return lettered_boxes

def createBoundingBoxImage(bounding_box, whole_image, padding=50):
    bounding_box_image = np.ones((bounding_box[3]+30, bounding_box[2]+30), dtype="uint8")
    bounding_box_image = whole_image[bounding_box[1]-15:bounding_box[1]+bounding_box[3]+15, bounding_box[0]-15:bounding_box[0]+bounding_box[2]+15]
    bounding_box_image = cv2.copyMakeBorder(bounding_box_image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    bounding_box_image = np.append(bounding_box_image, bounding_box_image, axis=1)
    return bounding_box_image

def displayImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()