import cv2
import numpy as np
import pytesseract
import text_detection as td
import sys
import copy
import matplotlib.pyplot as plt



def main():
    sys.argv.pop(0)
    for image_filename in sys.argv:
        analyzeImage(image_filename)


def analyzeImage(file_name):

    img = cv2.imread(file_name)
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
    gray_img = img.copy()

    msers, bbox = mser.detectRegions(gray) # bbox is xmin, ymin, xrange, yrange

    portions = []
    blocks = []
    count = 0
    for n in range(0, len(bbox)):
        portions.append(gray[bbox[n][1]:(bbox[n][1] + bbox[n][3]), bbox[n][0]:(bbox[n][0] + bbox[n][2])])
        #cv2.rectangle(img, (bbox[n][0], bbox[n][1]) , (bbox[n][0] + bbox[n][2], bbox[n][1] + bbox[n][3]), (0,255,255), 2 )
        cv2.imshow('Image', portions[n])

        inp = input('character:')
        inp = 'tsta'
        sttr = inp + '_' +str(count+n) + '.bmp'

        cv2.imwrite(sttr, portions[n])
        cv2.waitKey(0)
        if inp = 'i':
            


if __name__ == "__main__":
    main()



