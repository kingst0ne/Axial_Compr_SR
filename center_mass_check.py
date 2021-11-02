from scipy import ndimage
from numpy import genfromtxt
import numpy as np
import csv


a = genfromtxt('test.csv', delimiter=',')
points = np.zeros((len(a),4))
for i in range(len(a)):

    points[i,0] = a[i,0]
    points[i,1] = a[i,1]




cen = ndimage.center_of_mass(a)
print(cen)