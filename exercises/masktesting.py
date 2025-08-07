import numpy as np
import cv2

img = cv2.imread('exercises/01_color_based_segmentation/input/flower.jpg')  # Load the image

lower_limit = (0, 100, 100)
upper_limit = (100, 255, 255)

mask = cv2.inRange(img, lower_limit, upper_limit)  # Create a mask for the specified color range
cv2.imwrite('exercises/01_color_based_segmentation/flower_yellow_rgb.jpg', mask)  # Save the mask as an image

cv2.imshow("Original Image", img)  # Display the original image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window
