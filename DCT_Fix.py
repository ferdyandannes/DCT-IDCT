import cv2
import os
import math
from PIL import Image
from scipy.misc import imshow
import numpy as np
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib.pyplot as plt

# Load the input image
original_img = cv2.imread("lena.jpg")

# Convert the input image into grayscale value
gray_imgs = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

#gray_img = np.matrix([[8, 5], [3, 4]])
#gray_img = np.matrix([[255, 0], [20, 55]])

print('gambar opencv ')
print(gray_imgs)


# Display the original histogram
img_size = gray_imgs.shape
heights = img_size[0]
widths = img_size[1]

# Define the size of filter
size = 8
j = 0
l = np.zeros((size,size))
tot = np.zeros(size*size)
total = 0

# Cu Cv
#cucv1 = 1/math.sqrt(size)
#cucv2 = 1

# Copy the original image, used to display the new result
#gray_img_copy = gray_imgs.copy()
gray_img_copy2 = gray_imgs.copy()


# Save img to array
gray_img = np.zeros((256,256))
gray_img_copy = np.zeros((256,256))

for a in range(widths):
    for b in range(heights):
        gray_img[a,b] = gray_imgs[a,b]

print('gambar dalam matrix')
print(gray_img)

for a in range(0,heights,size):
    for b in range(0, widths,size):
        #print("a", a, "b", b)
        # Looping for storing the value
        f=0
      
        for c in range(a,a+size):
            g=0
            for d in range(b,b+size):
                l[f,g] = gray_img[c,d]
                g += 1
            f += 1
        #print("nilai L", l)

      
        for n in range(a,a+size):
            for o in range(b,b+size):
                j = 0
                for h in range(0,size):
                    for i in range(0,size):
                        #aw = n-a
                        #print("n", n,"a", a,"aw", aw)
                        tot[j] = l[h,i]*math.cos(((2*h+1)*(n-a)*math.pi)/(2*size))*math.cos(((2*i+1)*(o-b)*math.pi)/(2*size))
                        #print("h", h, "i", i, "tot", tot[i], "Pixel skrg", l[h,i])
                        j += 1

                #print("j", tot[63])

                # PENJUMLAHAN SEMUA
                k = 0
                
                for k in range(0,size*size):
                    total += tot[k]
                #print("k", k)
                
                # Perkalian CuCv
                if n-a == 0 :
                    cucv1 = 1/math.sqrt(2)
                elif n-a > 0 :
                    cucv1 = 1
                if o-b == 0 :
                    cucv2 = 1/math.sqrt(2)
                elif o-b > 0 :
                    cucv2 = 1
                    
                dct_result = (2/size)*cucv1*cucv2*total

                #### PRINT DCT ########
                #print("dct", round(dct_result))
                #print(" ")

                gray_img_copy[n, o] = round(dct_result)

                #print(dct_result)
              

                #Reset
                total = 0
                dct_result = 0
                tot = np.zeros(size*size)
                
        
        # Reset the value
        total = 0
        dct_result = 0
        l = np.zeros(shape=(size,size))
        tot = np.zeros(size*size)
        
        # Main part DCT
        #f[a][b] = 0.5*cucv*cucv*((zz l[f,g]*cos((2*0+1)*0*3.14/4)*cos((2*0+1)*0*3.14/4))zz + yy l[f+1,g]* yy)

for a in range(0,heights):
    for b in range(0, widths):
        if b % size == 0 and a % size == 0:
            #print("ya")
            gray_img_copy2[a, b] = gray_img_copy[a, b]
        else:
            gray_img_copy2[a, b] = 0
            #print("tidak")

print('hohoho')  
print(gray_img_copy)

img = Image.fromarray(gray_img_copy)
if img.mode != 'RGB':
    img = img.convert('RGB')
img.save('hasil.bmp')
img.show()

# ARRAY TO IMG USING PLT
#plt.imshow(gray_img_copy, cmap="gray")
#plt.show()
        
#cv2.imshow('Display the result', gray_img_copy)
#cv2.imshow('awkarin', gray_img_copy2)
#cv2.imwrite("dct2.jpg", gray_img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()



## backup
