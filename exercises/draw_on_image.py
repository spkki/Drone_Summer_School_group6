import numpy as np
import cv2

img = np.zeros((100, 200, 3), dtype=np.uint8)  # Create a black image

cv2.line(img, (20, 30), (40, 120), (0, 0, 255), 3)  # Draw a red line
cv2.circle(img, (50, 50), 20, (255, 0, 0), -1)  # Draw a filled blue circle
cv2.imwrite('drawn_shapes.jpg', img)  # Save the image with the line