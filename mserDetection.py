import cv2
import numpy as np
import pytesseract
import text_detection as td
class Block:
    def __init__(self, bound, img):
        self.bound = bound
        self.ch = ''
        self.img = img
    
    def getString():
        td.getString(self.img)


#from tesseract import image_to_string

img = cv2.imread('../../../Desktop/ch3.jpg')
mser = cv2.MSER_create()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
gray_img = img.copy()

msers, bbox = mser.detectRegions(gray) # bbox is xmin, ymin, xrange, yrange
print(len(bbox))
portions = []
blocks = []

for n in range(0, len(bbox)):
    portions.append(gray[bbox[n][1]:(bbox[n][1] + bbox[n][3]), bbox[n][0]:(bbox[n][0] + bbox[n][2])])
    cv2.rectangle(img, (bbox[n][0], bbox[n][1]) , (bbox[n][0] + bbox[n][2], bbox[n][1] + bbox[n][3]), (0,255,255), 2 )
    blocks.append(Block(bbox[n,:], portions[n]))



cv2.imshow('Blocked Image', img)
cv2.waitKey(0)
    
for n in range(0, len(blocks)):
    cv2.imshow(blocks[n].getString(), blocks[n].img)
    cv2.waitKey(0)





