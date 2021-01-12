import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np


# Read in the image
image = mpimg.imread('repsol.jpg')

plt.imshow(image)
plt.show()


# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')

sobel_z = np.array([[-1, 2, 3, 2, 1],
                   [-2,-3, 0, 3, 2],
                   [-3, 0, 0, 0, 3],
                    [-2,-3,0, 3, 2],
                    [-1,-2,-3,-2,-1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image5 = cv2.filter2D(gray, -1, sobel_z)
plt.imshow(filtered_image5, cmap='gray')
plt.show()



########   ADDING EDGES

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator
s_x = np.array([[ -1, 0, 1], 
                   [ -2, 0, 2], 
                   [ -1, 0, 1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image = cv2.filter2D(gray, -1, sobel_y)
filtered_imag2 = cv2.filter2D(gray, -1, s_x)
filtered = filtered_image + filtered_imag2

plt.imshow(filtered, cmap='gray')
plt.show()
