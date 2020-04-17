import matplotlib.pyplot as plt
import numpy as np 
img=plt.imread('test2.jpeg')
img2=img.copy()
img2.setflags(write=1)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img2[i,j,0]*=3
        img2[i,j,1]*=3
        img2[i,j,2]*=3
plt.imshow(img2)
plt.show()