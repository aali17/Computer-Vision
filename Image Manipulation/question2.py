# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:14:00 2020

@author: Azam
"""

from skimage import io
import matplotlib.pyplot as plt

fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
I = io.imread('H:\Courses\SP20\Computer Vision\Project0/gigi.jpg')
ax1.imshow(I)
ax1.axis('off')
x,y,c = I.shape
threshold_indices = I < 50
I[threshold_indices] = 50
I = I - 50
ax2.imshow(I)
ax2.axis('off')

ax1.set_title("Original")
ax2.set_title("Darker")

