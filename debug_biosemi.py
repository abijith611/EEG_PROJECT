import scipy.io
import numpy as np

# Load the .mat file
mat = scipy.io.loadmat('biosemi64.mat')
biosemi64 = mat['biosemi64']  # shape (64, 3)

print("First 5 coordinates (x,y,z):")
print(biosemi64[:5])
print("\nMin and max values in each axis:")
print("x: min =", biosemi64[:,0].min(), "max =", biosemi64[:,0].max())
print("y: min =", biosemi64[:,1].min(), "max =", biosemi64[:,1].max())
print("z: min =", biosemi64[:,2].min(), "max =", biosemi64[:,2].max())
print("\nOverall min/max of all coordinates:", biosemi64.min(), biosemi64.max())