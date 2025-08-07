import numpy as np
import cv2

img = cv2.imread('exercises/01_color_based_segmentation/input/deer.jpg')  # Load an image
cv2.imshow("Deer", img)  # Display the image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window

R = img[:, :, 2]
G = img[:, :, 1]
B = img[:, :, 0]

cv2.imwrite("deer_red.jpg", R)
cv2.imwrite("deer_green.jpg", G)
cv2.imwrite("deer_blue.jpg", B)