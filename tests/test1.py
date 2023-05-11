import cv2
import numpy as np

mask = cv2.imread('test.png')
import pdb; pdb.set_trace()
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)



# contour, _ = cv2.findContours(mask.astype(int), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# moments
M = cv2.moments(mask)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
area = 0#cv2.contourArea(contour)

