# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:39:37 2021

@author: DELL
"""

import cv2 
import matplotlib.pyplot as plt

path1=r'D:\photo numerique\d.jpg'
path2=r'D:\photo numerique\e.jpg'

img1 = cv2.imread(path1)  
img2 = cv2.imread(path2) 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

figure, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].imshow(img1, cmap='gray')
ax[1].imshow(img2, cmap='gray')