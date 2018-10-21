import cv2
import numpy as np
import pytesseract
import image_edit as ie

def getString(img):
    if len(img.shape) == 3:
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    img = cv2.erode(img, kernel, iterations = 1)

    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    
    padded = ie.pad_white(img)
    pad = np.append(padded, padded, axis=1)

    std  = pytesseract.image_to_string(pad)

    if len(std) == 0:
        return ''
    else:
        if std.isalnum:
            if std.startswith('I') | std.startswith('l'):
                return ''
            else:
                return std[0]
        else:
            return ''
    return std



# contour_image = cv2.drawContours(raw_image, contours, -1, (255, 0, 0), 10)
# rect_image = cv2.rectangle(raw_image, (x, y), (x + w, y + h), (255,0,0), 2)

# print(len(valid_lines))
#     combos = list(itertools.combinations(valid_lines, 2))
#     for combo in combos:
#         drawColinearRelation(original_image, combo[0]["combo"], combo[0]["slope"], combo[0]["intercept"], combo[1]["combo"], combo[1]["slope"], combo[1]["intercept"])