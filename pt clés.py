# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:05:21 2021

@author: DELL
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(r'D:\photo numerique\d.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)
plt.imshow(img)