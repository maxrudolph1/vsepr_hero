import numpy as np
import cv2
def pad_white(img):



    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pad = .25

    r = len(img)
    c = len(img[0])
    newImg = np.zeros((round((1+pad*2)*r),round((1+pad*2)*c)), np.uint8) + 255
    top = round(pad * r)
    left = round(pad * c) 

    for i in range(round(top), top + r-1):
        for k in range(round(left), left + c-1):
            newImg[i, k] = img[i - top, k - left]

    return newImg

def image_overlap(img, box1, box2, thresh):
    if len(img.shape) == 3:
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    newImg = np.zeros(r,c, np.uint8)
    for i in range(box1[1], box1[1] + box1[3]):
        for k in range(box1[0], box1[0] + box1[2]):
            newImg[i][k] = 100
    count = 0
    for i in range(box2[1], box2[1] + box2[3]):
        for k in range(box2[0], box2[0] + box2[2]):
            if newImg[i][k] == 100:
                count = count + 1

    A1 = box1[2]*box1[3]
    A2 = box2[2]* box2[3]
    overlap = 0
    if A1 >= A2:
        overlap =  A2/count
    else:
        overlap =  A1/count

    if overlap > thresh:
        if A1 >= A2:
            return box1
        else:
            return box2
    else:
        return 0





            

    newImg = np.zeros((round((1+pad*2)*r),round((1+pad*2)*c)), np.uint8) + 255