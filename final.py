import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
from scipy import fftpack,signal
import cv2
def getimg():
    print('enter filename.extension')#should be in the same directory as the .py file
    name=raw_input()0
    img = plt.imread(name)
    plt.subplot(1,2,1)
    plt.imshow(img)
    return img

def choice():
    print('1.to remove a colour from the image\n2.black and white\
    \n3.fourier transform\n4.invert image\
    \n5.impose 2 images(larger image first)\
    \n6.gaussain filter\
    \n7.edge detection')
    ch=input('enter your choice')
    return ch 


def rmclr(img):
    a=input('which colour would you like to remove?\n0.red\n1.green\n2.blue\n')
    rmc=img.copy()
    rmc.setflags(write=1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rmc[i,j,a]=0
    plt.subplot(1,2,2)
    plt.imshow(rmc)
    cv2.imwrite('rmclr.png',rmc)
    plt.show()


def bw(img):
    gray1= img[...,0]*0.2125+img[...,1]*0.7154+img[...,2]*0.0721
    plt.subplot(1,2,2)
    plt.imshow(gray1)
    cv2.imwrite('gray_image.png',gray1)
    plt.show()


def ft(img):
    img_fft=fftpack.fft2(img)
    plt.subplot(1,2,2)
    imff=np.abs(img_fft)
    plt.imshow(np.abs(img_fft))
    plt.colorbar()
    cv2.imwrite('ft.png',imff)
    plt.show()



def invert(img):
    gray= img[...,0]*0.2125+img[...,1]*0.7154+img[...,2]*0.0721
    copy=np.zeros((gray.shape[0],gray.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            copy[-1-i,-1-j]=gray[i,j]
    plt.subplot(1,2,2)
    plt.imshow(copy)
    cv2.imwrite('invert.png',copy)
    plt.show()


def impose(img1,img2):
    imp=img1.copy()
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            imp[i,j,0]=img2[i,j,0]
            imp[i,j,1]=img2[i,j,1]
            imp[i,j,2]=img2[i,j,2]
    plt.subplot(1,2,2) 
    plt.imshow(imp)
    cv2.imwrite('imposeop.png',imp)
    plt.show()   
    

def gaus(img):
    ker=np.array([1/256.000,4/256.000,6/256.000,4/256.000,1/256.000,4/256.000,16/256.000,24/256.000,16/256.000,4/256.000,6/256.000,24/256.000,36/256.000,24/256.000,6/256.000,4/256.000,16/256.000,24/256.000,16/256.000,4/256.000,1/256.000,4/256.000,6/256.000,4/256.000,1/256.000]).reshape((5,5))
    gray= img[...,0]*0.2125+img[...,1]*0.7154+img[...,2]*0.0721
    gaus=signal.convolve2d(gray,ker)
    plt.subplot(1,2,2)
    plt.imshow(gaus)
    cv2.imwrite('gauss.png',gaus)
    plt.show()

def edge(img):
    ker=np.array([-1,-1,-1,-1,8,-1,-1,-1,-1]).reshape((3,3))
    gray= img[...,0]*0.2125+img[...,1]*0.7154+img[...,2]*0.0721
    gaus=signal.convolve2d(gray,ker)
    plt.subplot(1,2,2)
    plt.imshow(gaus)
    cv2.imwrite('edge.png',gaus)
    plt.show()



ch=choice()
if ch==1:
    rmclr(getimg())
elif ch==2:
    bw(getimg())
elif ch==3:
    ft(getimg())
elif ch==4:
    invert(getimg())
elif ch==5:
    img1=getimg()
    img2=getimg()
    impose(img1,img2)
elif ch==6:
    gaus(getimg())
elif ch==7:
    edge(getimg())