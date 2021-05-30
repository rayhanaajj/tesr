# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:33:08 2021

@author: DELL
"""


import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread(r'D:\photo numerique\f.jpg')
img2 = cv.imread(r'D:\photo numerique\g.jpg')

gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None) 

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(gray1, kp1, gray2, kp2, matches[:50], gray2, flags=2)
plt.imshow(img3),plt.show()
