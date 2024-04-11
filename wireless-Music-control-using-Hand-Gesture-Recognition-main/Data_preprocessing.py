import cv2
import numpy as np

# load the image
image = cv2.imread('hand_gesture.jpg')

# resize the image to 64x64 pixels
image = cv2.resize(image, (64, 64))

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# normalize the pixel values
gray = gray / 255.0

# reshape the image to a 4D tensor with shape (1, 64, 64, 1)
image = np.reshape(gray, (1, 64, 64, 1))

#Saved Grayscale
print('Converted to Grayscale...........!!!!!!!!')
