import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
original_img = cv2.imread("dct2.JPG")

# Convert the input image into grayscale value
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

#gray_img = np.matrix([[10, 1, 10, 1], [3, 2, 3, 2], [10, 1, 10, 1], [3, 2, 3, 2]])

# Display the original histogram
img_size = gray_img.shape
heights = img_size[0]
widths = img_size[1]

# Define the size of filter
size = 8
j = 0
l = np.zeros(shape=(size,size))
tot = np.zeros(size*size)
total = 0

# Cu Cv
#cucv1 = 1/math.sqrt(size)
#cucv2 = 1

# Copy the original image, used to display the new result
gray_img_copy = gray_img.copy()
gray_img_copy2 = gray_img.copy()

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
                        # Perkalian CuCv
                        if h == 0 :
                            cucv1 = 1/math.sqrt(size)
                        elif h > 0 :
                            cucv1 = 1
                        if i == 0 :
                            cucv2 = 1/math.sqrt(size)
                        elif i > 0 :
                            cucv2 = 1
                        tot[j] = cucv1*cucv2*l[h,i]*math.cos((2*(n-a)+1)*h*3.14/(2*size))*math.cos((2*(o-b)+1)*i*3.14/(2*size))
                        #print("h", h, "i", i, "tot", tot[i], "Pixel skrg", l[h,i])
                        j += 1

                # PENJUMLAHAN SEMUA
                k = 0
                
                for k in range(0,size*size):
                    total += tot[k]
                    
                dct_result = (2/size)*total

                gray_img_copy[n, o] = round(dct_result)

                #print(round(dct_result))

                tot = np.zeros(size*size)
                total = 0
                dct_result = 0

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
            
print(gray_img_copy)
cv2.imshow('Display the result', gray_img_copy)
cv2.imshow('awkarin', gray_img_copy2)
#cv2.imwrite("standar_average.jpg", gray_img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()



## backup
