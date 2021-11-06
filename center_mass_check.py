from scipy import ndimage
from numpy import genfromtxt
import numpy as np
from make_grid_my import main



a = genfromtxt('test.csv', delimiter=',')
points = np.zeros((len(a),4))
for i in range(len(a)):

    points[i,0] = a[i,0]
    points[i,1] = a[i,1]

CM = main('profile.dat')

print(CM)

cen = ndimage.center_of_mass(a)
print(cen)