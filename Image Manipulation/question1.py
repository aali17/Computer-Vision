# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:58:51 2020

@author: Azam
"""

from skimage import io
import time
import numpy as np
import matplotlib.pyplot as plt

A = io.imread('H:\Courses\SP20\Computer Vision\Project0/grizzlypeakg.png')

(m1,n1) = A.shape

# counting time elapsed for provided code
t = time.time()
for i in range(m1):
    for j in range(n1):
        if A[i,j] <= 10 :
            A[i,j] = 0
elapsed_old1 = time.time() - t
print("Time elapsed for provided code"+ "=" + str(elapsed_old1))

#A1 = np.asarray(A)


B = io.imread('H:\Courses\SP20\Computer Vision\Project0/grizzlypeakg.png')
t = time.time()
threshold_indices = B < 10
B[threshold_indices] = 0
elapsed_new1 = time.time() - t
print("Time elapsed for my code"+ "=" + str(elapsed_new1))

factor_1 = elapsed_old1/elapsed_new1
print("speed factor for a single gray image" + "=" + str(factor_1))





# for 10 images typical way
A = io.ImageCollection('H:\Courses\SP20\Computer Vision\Project0/dataset' + '/dat*.tif')
c = io.collection.concatenate_images(A)
t = time.time()
l1,l2,l3 = c.shape
for k in range(l1):
    for i in range(l2):
        for j in range(l3):
            if c[k,i,j] <= 10 :
                c[k,i,j] = 0
elapsed_old10 = time.time() - t
print("Time elapsed for given code for 10 images"+ "=" + str(elapsed_old10))

# 10 images own way
A = io.ImageCollection('H:\Courses\SP20\Computer Vision\Project0/dat' + '/dat*.tif')
d = io.collection.concatenate_images(A)
t = time.time()
threshold_indices = d < 10
d[threshold_indices] = 0
elapsed = time.time() - t
print("Time elapsed for my code for 10 images"+ "=" + str(elapsed))
time_array = np.empty([3])
time_array[0] = elapsed

# 20 images own way
# =============================================================================
# A = io.ImageCollection('H:\Courses\SP20\Computer Vision\Project0/dataset' + '/dat*.tif')
# e = io.collection.concatenate_images(A)
# t = time.time()
# threshold_indices = e < 10
# e[threshold_indices] = 0
# elapsed = time.time() - t
# print("Time elapsed for my code for 20 images"+ "=" + str(elapsed))
# time_array[1] = elapsed
# 
# =============================================================================
# 30 images own way
# =============================================================================
# A = io.ImageCollection('H:\Courses\SP20\Computer Vision\Project0/dataset30' + '/dat*.tif')
# f = io.collection.concatenate_images(A)
# t = time.time()
# threshold_indices = f < 10
# f[threshold_indices] = 0
# elapsed = time.time() - t
# print("Time elapsed for my code for 30 images"+ "=" + str(elapsed))
# time_array[2] = elapsed
# =============================================================================

#print(time_array)


elapsed_old = elapsed_old10*1000/10
elapsed_new = time_array[0]*1000/10
factor_10 = elapsed_old/elapsed_new
print("speed factor for 10 images" + "=" + str(factor_10))

# color image
A = io.imread('H:\Courses\SP20\Computer Vision\Project0/grizzlypeakg.jpg')
l1,l2,l3 = c.shape
t = time.time()
for k in range(l1):
    for i in range(l2):
        for j in range(l3):
            if c[k,i,j] <= 10 :
                c[k,i,j] = 0
elapsed_c = time.time() - t
print("Time elapsed for my code for color image"+ "=" + str(elapsed))

# color image
A = io.imread('H:\Courses\SP20\Computer Vision\Project0/grizzlypeakg.jpg')
t = time.time()
threshold_indices = A < 10
A[threshold_indices] = 0
elapsed_color = time.time() - t
print("Time elapsed for my code for color image"+ "=" + str(elapsed))
# =============================================================================
# plt.imshow(A)
# plt.show()
# =============================================================================


factor_color = elapsed_c/elapsed_color
print("speed factor for a color image" + "=" + str(factor_color))
