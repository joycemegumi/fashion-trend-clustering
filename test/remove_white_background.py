# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:14:36 2018

test to remove white background
Inspired by https://stackoverflow.com/questions/29810128/opencv-python-set-background-colour

@author: AI team
"""
import collections
from collections import OrderedDict
import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import utils

img_path = parentdir + '/data/dataset/test/hockeyskatesexample_2.jpg'


#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 5
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
img = cv2.imread(img_path)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection -------------------------------------------------------------------
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area ---------------------------------------------
contour_info = []
_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Previously, for a previous version of cv2, this line was: 
#  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Thanks to notes from commenters, I've updated the code but left this note
for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
#cv2.fillConvexPoly(mask, max_contour[0], (255))
for c in contour_info:
    cv2.fillConvexPoly(mask, c[0], (255))

#-- Smooth mask, then blur it --------------------------------------------------------
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background --------------------------------------
mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
img         = img.astype('float32') / 255.0                 #  for easy blending

masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

# split image into channels
c_red, c_green, c_blue = cv2.split(img)

# merge with mask got on one of a previous steps
img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

# show on screen (optional in jupiter)
plt.imshow(img_a)
plt.show()

#cv2.imshow('img', masked)                                   # Display
#cv2.waitKey()


c_red, c_green, c_blue = cv2.split(img)
mask_normalized = mask.astype('float32') / 255.0
#c_red = np.multiply(c_red, mask_normalized) 
c_red[mask_normalized<0.01]=1
#c_green = np.multiply(c_green, mask_normalized) 
c_green[mask_normalized<0.01]=1
#c_blue = np.multiply(c_blue, mask_normalized) 
c_blue[mask_normalized<0.01]=1

img_b = cv2.merge((c_red, c_green, c_blue))

# display
plt.imshow(img_b)
plt.show()
